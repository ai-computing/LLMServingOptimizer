"""D3 tests: power parser, monitor runtime loop (metrics+power+SLO-driven
READY<->DEGRADED transitions, log pump), and the SSE endpoints — all with
fixtures, no dockerd/GPU."""
from __future__ import annotations

import pytest

from service.deploy.docker_driver import FakeDriver
from service.deploy.manager import DeploymentManager
from service.deploy.state import DeploymentState as S
from service.deploy.store import DeployStore
from service.deploy.vllm_launcher import build_spec
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry
from service.monitor.power import parse_nvidia_csv
from service.monitor.runtime import MonitorRuntime
from service.monitor.store import PowerSample
from service.tests.test_monitor import FIXTURE

MODEL = "meta-llama/Llama-3.1-8B"


def test_power_csv_parser():
    text = "0, 187.5, 92, 21504, 71\n1, 60.2, 0, 512, 45\n2, [N/A], 0, 0, 0\n"
    samples = parse_nvidia_csv(text, {0: "n0/A40/0", 1: "n0/A40/1"}, ts=10.0)
    assert len(samples) == 2                      # index 2 not mapped -> dropped
    assert samples[0].device_id == "n0/A40/0"
    assert samples[0].power_w == 187.5
    assert samples[0].mem_used_gb == pytest.approx(21.0)
    assert samples[1].power_w == 60.2


@pytest.fixture()
def env(tmp_path):
    reg = ClusterRegistry.model_validate({"nodes": [
        {"id": "n0", "devices": [{"name": "A40", "count": 2, "mem_gb": 48}]}]})
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    store = DeployStore(tmp_path / "l.sqlite")
    ver, _ = ledger.snapshot()
    res = ledger.reserve("t", "j1", ["n0/A40/0", "n0/A40/1"], ver)
    driver = FakeDriver()
    mgr = DeploymentManager(store, driver, ledger, health_fn=lambda u: True,
                            run_async=False, sleep=lambda s: None)
    dep_id = mgr.create(res.id, "t",
                        build_spec("dep-mon", MODEL, res.device_ids, tp=2, port=8001),
                        slo={"tpot_ms": 50})      # fixture tpot p95 = 87.5 -> violated
    return store, driver, dep_id, mgr


def _scaled(text: str, k: int) -> str:
    """Multiply every trailing numeric value by k — simulates SUSTAINED load
    for delta-window quantiles (identical counters would mean a quiet window)."""
    import re
    return re.sub(r" ([\d.]+)$",
                  lambda m: f" {float(m.group(1)) * k}", text, flags=re.M)


def _run_loop(store, driver, dep_id, ticks, metrics_text=FIXTURE):
    """Run the monitor loop synchronously for `ticks` iterations."""
    scrapes = {"n": 0}

    def _http_get(url):
        scrapes["n"] += 1
        return _scaled(metrics_text, scrapes["n"])

    rt = MonitorRuntime(
        store, driver=driver, interval_s=0,
        http_get=_http_get,
        power_fn=lambda gpu_map: [PowerSample(ts=float(_run_loop.t), device_id=d,
                                              power_w=150.0)
                                  for d in gpu_map.values()],
        sleep=lambda s: _tick(rt, store, dep_id))
    _run_loop.t = 0.0
    counter = {"n": 0}

    def _tick(rt_, store_, dep_id_):
        counter["n"] += 1
        _run_loop.t += 5.0
        if counter["n"] >= ticks:
            rt_._stop[dep_id_].set()

    rt._stop[dep_id] = __import__("threading").Event()
    rt._loop(dep_id, rt._stop.setdefault(
        dep_id, __import__("threading").Event()))
    return rt


def test_runtime_collects_and_slo_drives_degraded(env):
    store, driver, dep_id, _ = env
    rt = _run_loop(store, driver, dep_id, ticks=4)
    buf = rt.buf(dep_id)
    assert len(buf.metrics) >= 3
    assert buf.metrics[-1].tpot_p95_ms == pytest.approx(87.5)
    assert buf.power and buf.power[-1][1] == 300.0     # 2 devices x 150 W
    # SLO tpot 50 violated 3 consecutive windows -> DEGRADED transition
    assert store.get(dep_id).state == S.DEGRADED
    assert buf.last_slo.verdict == "violated"
    # log pump drained the fake stream
    assert list(buf.logs) == ["fake line 1", "fake line 2"]


def test_runtime_ok_slo_stays_ready(env):
    store, driver, dep_id, _ = env
    good = FIXTURE  # tpot p95 = 87.5
    with store._conn() as c:  # relax the promised SLO on the row
        c.execute("UPDATE deployments SET slo_json='{\"tpot_ms\": 200}' WHERE id=?",
                  (dep_id,))
    _run_loop(store, driver, dep_id, ticks=4, metrics_text=good)
    assert store.get(dep_id).state == S.READY


def test_manager_hooks_attach_detach(env):
    store, driver, _, _ = env
    calls = []
    ledger = Ledger(store.db_path, ClusterRegistry.model_validate(
        {"nodes": [{"id": "n0", "devices": [
            {"name": "A40", "count": 2, "mem_gb": 48}]}]}))
    ledger.release("j1")            # env fixture holds both devices; free one up
    ver, free = ledger.snapshot()
    mgr = DeploymentManager(store, driver, ledger, health_fn=lambda u: True,
                            run_async=False, sleep=lambda s: None,
                            on_ready=lambda d: calls.append(("attach", d)),
                            on_stopped=lambda d: calls.append(("detach", d)))
    res = ledger.reserve("t", "j2", free[:1], ver)
    dep_id = mgr.create(res.id, "t",
                        build_spec("dep-h", MODEL, free[:1], tp=1, port=8002))
    mgr.terminate(dep_id, drain_timeout_s=1)
    assert calls == [("attach", dep_id), ("detach", dep_id)]


def test_sse_metrics_and_logs_endpoints(env):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from service.api.deployment_routes import create_deployment_router
    from service.api.routes import ServiceState

    store, driver, dep_id, mgr = env
    rt = _run_loop(store, driver, dep_id, ticks=3)
    state = ServiceState(registry=ClusterRegistry.model_validate(
        {"nodes": [{"id": "n0", "devices": [
            {"name": "A40", "count": 2, "mem_gb": 48}]}]}),
        ledger=mgr.ledger)
    state.deploy_store = store
    state.deploy_manager = mgr
    state.monitor_runtime = rt
    app = FastAPI()
    app.include_router(create_deployment_router(state))
    client = TestClient(app)

    # terminate so streams end quickly
    mgr.terminate(dep_id, drain_timeout_s=0)
    with client.stream("GET", f"/api/deployments/{dep_id}/metrics") as r:
        text = "".join(r.iter_text())
    assert '"type": "sample"' in text and '"type": "end"' in text
    assert '"energy_wh"' in text

    with client.stream("GET", f"/api/deployments/{dep_id}/logs?tail=10") as r:
        text = "".join(r.iter_text())
    assert "fake line 1" in text and '"type": "end"' in text
