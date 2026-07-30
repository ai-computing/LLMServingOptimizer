"""DeploymentManager: drives the state machine (plan §4.2, §4.4).

PENDING → PULLING → STARTING → HEALTH_CHECK → READY, with N full-attempt
retries and rollback-on-failure (containers removed, reservation kept).
terminate: DRAINING (queue drain poll) → STOPPED → RELEASED (ledger release).
``recover()`` rebuilds state after a service restart from the store alone;
``reconcile()`` cross-checks the store against actual containers.

Injectables keep unit tests hardware-free: driver (FakeDriver), health_fn,
queue_len_fn, clock/sleep.
"""
from __future__ import annotations

import threading
import time
import urllib.request
from typing import Callable, Optional

from .docker_driver import NAME_PREFIX
from .spec import DeploymentSpec
from .state import DeploymentState as S
from .store import DeployStore


def _default_health_fn(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


class DeploymentManager:
    def __init__(self, store: DeployStore, driver, ledger,
                 health_fn: Callable[[str], bool] = _default_health_fn,
                 queue_len_fn: Optional[Callable[[str], int]] = None,
                 node_host: Optional[Callable[[str], str]] = None,
                 run_async: bool = True, sleep=time.sleep):
        self.store = store
        self.driver = driver
        self.ledger = ledger
        self.health_fn = health_fn
        # in-flight requests for DRAINING; None -> assume empty (P0: no gateway)
        self.queue_len_fn = queue_len_fn or (lambda dep_id: 0)
        # node_id -> reachable hostname for endpoint URLs (default: node id)
        self.node_host = node_host or (lambda node_id: node_id)
        self.run_async = run_async
        self.sleep = sleep

    # -- creation --------------------------------------------------------------
    def create(self, reservation_id: int, tenant: str, spec: DeploymentSpec,
               slo: Optional[dict] = None) -> str:
        dep = self.store.create(reservation_id, tenant, spec.model,
                                spec.model_dump(), slo=slo)
        if self.run_async:
            threading.Thread(target=self._bring_up, args=(dep.id,),
                             daemon=True).start()
        else:
            self._bring_up(dep.id)
        return dep.id

    def _endpoints(self, spec: DeploymentSpec) -> dict:
        c = spec.containers[0]
        host = self.node_host(c.node_id)
        base = f"http://{host}:{c.ports['api']}"
        return {"openai_url": f"{base}/v1", "metrics_url": f"{base}/metrics",
                "health_url": f"{base}/health"}

    def _bring_up(self, dep_id: str) -> None:
        dep = self.store.get(dep_id)
        spec = DeploymentSpec.model_validate(dep.spec)
        attempts = spec.health.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                self.store.transition(dep_id, S.PULLING,
                                      f"attempt {attempt}/{attempts}")
                for c in spec.containers:
                    self.driver.pull(c.node_id, c.image)
                self.store.transition(dep_id, S.STARTING)
                for c in spec.containers:      # head first, then workers (D5)
                    self.driver.run(c.node_id, c)
                self.store.transition(dep_id, S.HEALTH_CHECK)
                eps = self._endpoints(spec)
                deadline = time.time() + spec.health.timeout_s
                while time.time() < deadline:
                    if self.health_fn(eps["health_url"]):
                        self.store.set_endpoints(dep_id, eps)
                        self.store.transition(dep_id, S.READY, "healthy")
                        return
                    self.sleep(spec.health.interval_s)
                raise TimeoutError(f"health check timed out after "
                                   f"{spec.health.timeout_s}s")
            except Exception as e:  # noqa: BLE001 — every failure is retryable
                self._rollback_containers(spec)
                if attempt >= attempts:
                    # reservation is kept: the user chooses retry vs release
                    self.store.transition(dep_id, S.FAILED,
                                          error=f"{type(e).__name__}: {e}")
                    return
                # back to PENDING-equivalent for the next attempt: the state
                # machine forbids *->PULLING except from PENDING, so we record
                # the retry via events on the next PULLING transition
                self._force_pending(dep_id, str(e))

    def _force_pending(self, dep_id: str, why: str) -> None:
        # internal retry rewind — bypasses check_transition on purpose but is
        # still audit-logged through the events table
        import json as _json
        with self.store._conn() as c:  # noqa: SLF001 — same package
            cur = c.execute("SELECT state FROM deployments WHERE id=?",
                            (dep_id,)).fetchone()["state"]
            c.execute("UPDATE deployments SET state=? WHERE id=?",
                      (S.PENDING.value, dep_id))
            c.execute("INSERT INTO deployment_events VALUES (?,?,?,?,?)",
                      (dep_id, time.time(), cur, S.PENDING.value,
                       _json.dumps({"retry_after": why})[:500]))

    def _rollback_containers(self, spec: DeploymentSpec) -> None:
        for c in spec.containers:
            self.driver.stop(c.node_id, c.name, timeout_s=10)
            self.driver.rm(c.node_id, c.name)

    # -- termination ---------------------------------------------------------------
    def terminate(self, dep_id: str, drain_timeout_s: float = 120.0) -> None:
        """Idempotent: already-terminal deployments return unchanged."""
        dep = self.store.get(dep_id)
        if dep.state in (S.STOPPED, S.RELEASED, S.FAILED):
            if dep.state == S.STOPPED:
                self._release(dep_id)
            return
        spec = DeploymentSpec.model_validate(dep.spec)
        self.store.transition(dep_id, S.DRAINING,
                              f"drain_timeout={drain_timeout_s}s")
        deadline = time.time() + drain_timeout_s
        while time.time() < deadline and self.queue_len_fn(dep_id) > 0:
            self.sleep(1.0)
        for c in spec.containers:
            self.driver.stop(c.node_id, c.name, timeout_s=30)
            self.driver.rm(c.node_id, c.name)
        self.store.transition(dep_id, S.STOPPED)
        self._release(dep_id)

    def _release(self, dep_id: str) -> None:
        dep = self.store.get(dep_id)
        # reservation request_id == the service job id that reserved it; the
        # ledger release is idempotent by design
        res = next((r for r in self.ledger.active_reservations()
                    if r.id == dep.reservation_id), None)
        if res is not None:
            self.ledger.release(res.request_id)
        self.store.transition(dep_id, S.RELEASED)

    # -- restart recovery + reconcile ---------------------------------------------
    def recover(self) -> dict[str, list[str]]:
        """After a service restart: READY/DEGRADED stay attached (monitor
        re-attach is D3); mid-flight deployments are rolled back and re-queued."""
        out = {"reattached": [], "requeued": []}
        for dep in self.store.list():
            if dep.state in (S.READY, S.DEGRADED):
                out["reattached"].append(dep.id)
            elif dep.state in (S.PULLING, S.STARTING, S.HEALTH_CHECK, S.PENDING):
                spec = DeploymentSpec.model_validate(dep.spec)
                self._rollback_containers(spec)
                if dep.state != S.PENDING:
                    self._force_pending(dep.id, "service restart")
                out["requeued"].append(dep.id)
                if self.run_async:
                    threading.Thread(target=self._bring_up, args=(dep.id,),
                                     daemon=True).start()
                else:
                    self._bring_up(dep.id)
        return out

    def reconcile(self, node_ids: list[str]) -> dict[str, list[str]]:
        """Two-way ledger↔dockerd check: orphan containers (llmsvc-* with no
        live deployment) are removed; live deployments with missing containers
        are marked FAILED."""
        live = {d.id: d for d in self.store.list(
            states=[S.READY, S.DEGRADED, S.DRAINING])}
        expected: dict[str, str] = {}
        for dep in live.values():
            for c in DeploymentSpec.model_validate(dep.spec).containers:
                expected[c.name] = dep.id
        out = {"orphans_removed": [], "missing_marked_failed": []}
        actual: set[str] = set()
        for node in node_ids:
            for name in self.driver.list_names(node, NAME_PREFIX):
                actual.add(name)
                if name not in expected:
                    self.driver.stop(node, name, timeout_s=10)
                    self.driver.rm(node, name)
                    out["orphans_removed"].append(name)
        for name, dep_id in expected.items():
            if name not in actual and live[dep_id].state != S.DRAINING:
                self.store.transition(dep_id, S.FAILED,
                                      error=f"container {name} disappeared")
                out["missing_marked_failed"].append(dep_id)
        return out
