"""Docker drivers (plan D2): the real docker-SDK driver plus an in-memory fake
for unit tests. Both expose the same small surface the manager needs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .spec import ContainerSpec

NAME_PREFIX = "llmsvc-"   # reconcile key: every managed container name starts with this


class DriverError(Exception):
    pass


class DockerSdkDriver:
    """Talks to (possibly remote) dockerd via the docker SDK. Endpoints come
    from registry v2 node.docker (default: local socket). Credentials never
    leave this module (risk memo: no endpoint echo in API responses)."""

    def __init__(self, endpoints: Optional[dict[str, dict]] = None):
        self._endpoints = endpoints or {}
        self._clients: dict[str, object] = {}

    def _client(self, node_id: str):
        import docker
        if node_id not in self._clients:
            cfg = self._endpoints.get(node_id) or {}
            base_url = cfg.get("endpoint") or "unix://var/run/docker.sock"
            self._clients[node_id] = docker.DockerClient(
                base_url=base_url, tls=cfg.get("tls", False) or None,
                timeout=120)
        return self._clients[node_id]

    def ping(self, node_id: str) -> bool:
        try:
            return bool(self._client(node_id).ping())
        except Exception:
            return False

    def pull(self, node_id: str, image: str) -> None:
        self._client(node_id).images.pull(image)

    @staticmethod
    def build_run_kwargs(spec: ContainerSpec) -> dict:
        """containers.run kwargs (pure; unit-testable). GPU injection mode via
        LLMSS_GPU_MODE: 'legacy' (nvidia runtime DeviceRequest, default) or
        'cdi' (CDI-only docker setups reject the legacy path with a bad-driver
        combination inside the container — Error 803)."""
        import os

        from docker.types import DeviceRequest
        kwargs: dict = dict(
            image=spec.image, name=spec.name, detach=True,
            environment=spec.env, shm_size=spec.shm_size,
            ports={f"{p}/tcp": p for p in set(spec.ports.values())},
            labels={"llmsvc": "1"},
        )
        if spec.gpu_indices:
            if os.environ.get("LLMSS_GPU_MODE", "legacy") == "cdi":
                kwargs["device_requests"] = [DeviceRequest(
                    driver="cdi",
                    device_ids=[f"nvidia.com/gpu={i}" for i in spec.gpu_indices])]
            else:
                kwargs["device_requests"] = [DeviceRequest(
                    driver="nvidia",
                    device_ids=[str(i) for i in spec.gpu_indices],
                    capabilities=[["gpu"]])]
        if spec.device_paths:   # NPU character devices (least privilege — no
            kwargs["devices"] = [f"{p}:{p}" for p in spec.device_paths]  # privileged mode)
        if spec.volumes:
            kwargs["volumes"] = {host: {"bind": cont, "mode": "rw"}
                                 for host, cont in spec.volumes.items()}
        if spec.command:
            kwargs["command"] = spec.command
        return kwargs

    def run(self, node_id: str, spec: ContainerSpec) -> str:
        kwargs = self.build_run_kwargs(spec)
        try:
            container = self._client(node_id).containers.run(**kwargs)
        except Exception as e:
            raise DriverError(f"run {spec.name} on {node_id}: {e}") from e
        return container.id

    def stop(self, node_id: str, name: str, timeout_s: int = 30) -> None:
        try:
            self._client(node_id).containers.get(name).stop(timeout=timeout_s)
        except Exception:
            pass  # already gone

    def rm(self, node_id: str, name: str) -> None:
        try:
            self._client(node_id).containers.get(name).remove(force=True)
        except Exception:
            pass

    def list_names(self, node_id: str, prefix: str = NAME_PREFIX) -> list[str]:
        try:
            return [c.name for c in self._client(node_id).containers.list(all=True)
                    if c.name.startswith(prefix)]
        except Exception:
            return []

    def logs(self, node_id: str, name: str, tail: int = 200) -> str:
        try:
            return self._client(node_id).containers.get(name) \
                .logs(tail=tail).decode(errors="replace")
        except Exception as e:
            return f"<logs unavailable: {e}>"

    def log_stream(self, node_id: str, name: str):
        """docker logs --follow iterator (bytes per line)."""
        return self._client(node_id).containers.get(name) \
            .logs(stream=True, follow=True, tail=100)


@dataclass
class FakeDriver:
    """In-memory driver for unit tests: records calls, injectable failures
    (``fail={'pull': 2}`` fails the first 2 pull calls)."""

    fail: dict[str, int] = field(default_factory=dict)
    containers: dict[str, dict] = field(default_factory=dict)  # name -> info
    calls: list[tuple] = field(default_factory=list)

    def _maybe_fail(self, op: str):
        if self.fail.get(op, 0) > 0:
            self.fail[op] -= 1
            raise DriverError(f"injected {op} failure")

    def ping(self, node_id): return True

    def pull(self, node_id, image):
        self.calls.append(("pull", node_id, image))
        self._maybe_fail("pull")

    def run(self, node_id, spec: ContainerSpec) -> str:
        self.calls.append(("run", node_id, spec.name))
        self._maybe_fail("run")
        self.containers[spec.name] = {"node": node_id, "spec": spec,
                                      "state": "running"}
        return "cid-" + spec.name

    def stop(self, node_id, name, timeout_s=30):
        self.calls.append(("stop", node_id, name))
        if name in self.containers:
            self.containers[name]["state"] = "exited"

    def rm(self, node_id, name):
        self.calls.append(("rm", node_id, name))
        self.containers.pop(name, None)

    def list_names(self, node_id, prefix=NAME_PREFIX):
        return [n for n, i in self.containers.items()
                if i["node"] == node_id and n.startswith(prefix)]

    def logs(self, node_id, name, tail=200):
        return f"<fake logs {name}>"

    def log_stream(self, node_id, name):
        return iter([b"fake line 1\n", b"fake line 2\n"])
