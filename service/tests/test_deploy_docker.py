"""D2 docker-marker E2E: the real DockerSdkDriver + DeploymentManager drive a
lightweight container (python http server standing in for vLLM's /health)
through PENDING -> READY -> terminate -> RELEASED on the local dockerd.

Full vLLM-image E2E needs a multi-GB pull and a GPU; this validates the
docker plumbing (pull/run/port publish/health/stop/rm/reconcile) end to end.
"""
from __future__ import annotations

import pytest

from service.deploy.spec import ContainerSpec, DeploymentSpec, HealthPolicy
from service.deploy.state import DeploymentState as S

pytestmark = pytest.mark.docker

PORT = 18099
_SERVER = ("import http.server as h\n"
           "class H(h.BaseHTTPRequestHandler):\n"
           "    def do_GET(self):\n"
           "        self.send_response(200); self.end_headers()\n"
           "        self.wfile.write(b'ok')\n"
           f"h.HTTPServer(('0.0.0.0', {PORT}), H).serve_forever()\n")


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="no local dockerd / docker sdk")
def test_docker_e2e_ready_terminate_released(tmp_path):
    from service.deploy.docker_driver import DockerSdkDriver
    from service.deploy.manager import DeploymentManager
    from service.deploy.store import DeployStore
    from service.inventory.ledger import Ledger
    from service.inventory.registry import ClusterRegistry

    reg = ClusterRegistry.model_validate({"nodes": [
        {"id": "local", "devices": [{"name": "A40", "count": 1, "mem_gb": 48}]}]})
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    ver, _ = ledger.snapshot()
    res = ledger.reserve("t", "dockerjob", ["local/A40/0"], ver)

    store = DeployStore(tmp_path / "l.sqlite")
    driver = DockerSdkDriver()   # local socket
    mgr = DeploymentManager(store, driver, ledger,
                            node_host=lambda n: "localhost", run_async=False)

    spec = DeploymentSpec(
        model="fake/model", served_name="fake",
        containers=[ContainerSpec(
            node_id="local", image="python:3.12-slim",
            device_ids=[], gpu_indices=[],           # no GPU binding
            ports={"api": PORT, "metrics": PORT},
            command=["python", "-c", _SERVER],
            shm_size="256m",
            name="llmsvc-dep-dockertest-0")],
        health=HealthPolicy(interval_s=1.0, timeout_s=60.0, retries=0))

    dep_id = mgr.create(res.id, "t", spec)
    try:
        row = store.get(dep_id)
        assert row.state == S.READY, row.last_error
        assert "llmsvc-dep-dockertest-0" in driver.list_names("local")
        mgr.terminate(dep_id, drain_timeout_s=5)
        assert store.get(dep_id).state == S.RELEASED
        assert "llmsvc-dep-dockertest-0" not in driver.list_names("local")
        assert len(ledger.snapshot()[1]) == 1        # device released
    finally:  # belt and braces cleanup
        driver.stop("local", "llmsvc-dep-dockertest-0", timeout_s=5)
        driver.rm("local", "llmsvc-dep-dockertest-0")
