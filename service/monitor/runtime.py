"""MonitorRuntime: per-deployment 5s polling loops feeding the ring buffers
(plan §4.3). Attached on READY, detached on STOPPED; after a service restart
``attach`` is re-run for every READY row (manager.recover + webapp startup).

Injectables (http_get, power_fn, log_stream, sleep) keep tests fixture-only.
SLO verdicts drive the READY<->DEGRADED transitions through the deploy store.
"""
from __future__ import annotations

import threading
import time
import urllib.request
from typing import Callable, Optional

from ..deploy.spec import DeploymentSpec
from ..deploy.state import DeploymentState as S
from ..deploy.store import DeployStore
from .collector import Collector
from .power import query_nvidia_power
from .slo_checker import SLOChecker
from .store import DeploymentBuffers


def _default_http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode(errors="replace")


class MonitorRuntime:
    def __init__(self, store: DeployStore, driver=None, interval_s: float = 5.0,
                 http_get: Callable[[str], str] = _default_http_get,
                 power_fn: Optional[Callable[[dict], list]] = None,
                 sleep=time.sleep):
        self.store = store
        self.driver = driver          # for log following (optional)
        self.interval_s = interval_s
        self.http_get = http_get
        self.power_fn = power_fn or query_nvidia_power
        self.sleep = sleep
        self.buffers: dict[str, DeploymentBuffers] = {}
        self._stop: dict[str, threading.Event] = {}

    def buf(self, dep_id: str) -> DeploymentBuffers:
        return self.buffers.setdefault(dep_id, DeploymentBuffers(dep_id))

    # -- lifecycle -------------------------------------------------------------
    def attach(self, dep_id: str, block: bool = False) -> None:
        if dep_id in self._stop:      # already attached
            return
        stop = threading.Event()
        self._stop[dep_id] = stop
        if block:
            self._loop(dep_id, stop)
        else:
            threading.Thread(target=self._loop, args=(dep_id, stop),
                             daemon=True).start()

    def detach(self, dep_id: str) -> dict:
        ev = self._stop.pop(dep_id, None)
        if ev:
            ev.set()
        return self.buf(dep_id).summary()

    # -- polling loop --------------------------------------------------------------
    def _loop(self, dep_id: str, stop: threading.Event) -> None:
        dep = self.store.get(dep_id)
        spec = DeploymentSpec.model_validate(dep.spec)
        buf = self.buf(dep_id)
        collector = Collector(dep_id=dep_id)

        def _degraded(d):
            try:
                self.store.transition(d, S.DEGRADED, "SLO violated (3x30s windows)")
            except Exception:
                pass

        def _recovered(d):
            try:
                self.store.transition(d, S.READY, "SLO recovered")
            except Exception:
                pass

        checker = SLOChecker(dep_id=dep_id, targets=dep.slo or {},
                             on_degraded=_degraded, on_recovered=_recovered)
        gpu_map = {i: d for c in spec.containers
                   for i, d in zip(c.gpu_indices, c.device_ids)}
        c0 = spec.containers[0]
        log_iter = None
        if self.driver is not None and hasattr(self.driver, "log_stream"):
            try:
                log_iter = self.driver.log_stream(c0.node_id, c0.name)
            except Exception:
                log_iter = None
        if log_iter is not None:
            threading.Thread(target=self._pump_logs,
                             args=(buf, log_iter, stop), daemon=True).start()

        while not stop.is_set():
            row = self.store.get(dep_id)
            if row.state in (S.STOPPED, S.RELEASED, S.FAILED):
                break
            if row.endpoints:
                try:
                    sample = collector.sample(
                        self.http_get(row.endpoints["metrics_url"]))
                    buf.push_metrics(sample)
                    buf.last_slo = checker.push(sample)  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                samples = self.power_fn(gpu_map)
                if samples:
                    buf.push_power(samples[0].ts,
                                   sum(s.power_w for s in samples))
            except Exception:
                pass
            self.sleep(self.interval_s)
        self._stop.pop(dep_id, None)

    @staticmethod
    def _pump_logs(buf: DeploymentBuffers, lines, stop: threading.Event) -> None:
        try:
            for line in lines:
                if stop.is_set():
                    break
                if isinstance(line, bytes):
                    line = line.decode(errors="replace")
                buf.push_log(line.rstrip("\n"))
        except Exception:
            pass
