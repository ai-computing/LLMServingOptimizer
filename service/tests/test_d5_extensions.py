"""D5 tests: multi-node Ray launcher golden, NPU (furiosa) container spec,
calibration report + feedback YAML, and the termination history hook."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from service.deploy.vllm_launcher import (
    DEFAULT_IMAGE,
    NPU_IMAGE,
    build_spec,
    build_spec_multinode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "power_calibration_report",
    REPO_ROOT / "scripts" / "power_calibration_report.py")
pcr = importlib.util.module_from_spec(_spec)
sys.modules["power_calibration_report"] = pcr
_spec.loader.exec_module(pcr)

MODEL = "meta-llama/Llama-3.1-8B"


# ---- multi-node launcher golden ------------------------------------------------

def test_multinode_tp2_pp2_head_worker_golden():
    spec = build_spec_multinode(
        "dep-mn", MODEL,
        node_devices=[("n0", ["n0/A40/0", "n0/A40/1"]),
                      ("n1", ["n1/A40/0", "n1/A40/1"])],
        tp=2, port=8001, head_host="10.0.0.1")
    assert spec.engine_args["pipeline-parallel-size"] == 2
    assert spec.engine_args["distributed-executor-backend"] == "ray"
    head, worker = spec.containers
    assert (head.role, worker.role) == ("head", "worker")   # head first (join order)
    assert head.name == "llmsvc-dep-mn-0" and worker.name == "llmsvc-dep-mn-1"
    assert head.env["VLLM_HOST_IP"] == "10.0.0.1"
    assert worker.env["RAY_ADDRESS"] == "10.0.0.1:6379"
    assert "ray start --head" in head.command[-1]
    assert "--distributed-executor-backend ray" in head.command[-1]
    assert "ray start --address=10.0.0.1:6379" in worker.command[-1]
    assert head.ports["ray"] == 6379 and worker.ports == {}
    assert worker.gpu_indices == [0, 1]


def test_multinode_validation():
    with pytest.raises(ValueError, match=">= 2 nodes"):
        build_spec_multinode("d", MODEL, [("n0", ["n0/A40/0"])], tp=1, port=8001)
    with pytest.raises(ValueError, match="!= tp"):
        build_spec_multinode("d", MODEL,
                             [("n0", ["n0/A40/0"]), ("n1", ["n1/A40/0",
                                                            "n1/A40/1"])],
                             tp=1, port=8001)


# ---- NPU path ---------------------------------------------------------------------

def test_npu_container_spec():
    spec = build_spec("dep-npu", MODEL, ["n1/RNGD/0"], tp=1, port=8001)
    c = spec.containers[0]
    assert c.image == NPU_IMAGE
    assert c.gpu_indices == []                       # no nvidia binding
    assert c.device_paths == ["/dev/rngd0"]
    assert c.env["FURIOSA_DEVICES"] == "npu0"
    assert c.ports == {"api": 8001, "metrics": 8001}


def test_gpu_path_unchanged():
    spec = build_spec("dep-g", MODEL, ["n0/A40/0"], tp=1, port=8001)
    assert spec.containers[0].image == DEFAULT_IMAGE
    assert spec.containers[0].device_paths == []


# ---- calibration report ----------------------------------------------------------

def _rec(i, hw="A40", pred=600.0, meas=660.0, tp=2):
    return {"dep_id": f"dep-{i}", "model": MODEL, "hw": hw, "tp": tp,
            "predicted_power_w": pred, "measured_avg_w": meas,
            "energy_wh": 100.0 + i, "span_s": 3600.0}


def test_calibration_report_errors_and_feedback():
    records = [_rec(1, meas=660), _rec(2, meas=540),
               _rec(3, hw="A5000", pred=460, meas=391, tp=2),
               _rec(4, hw="A5000", pred=460, meas=414, tp=2),
               _rec(5, hw="A5000", pred=185, meas=203, tp=1)]
    md, drafts = pcr.build_report(records)
    assert "+10.0%" in md and "-10.0%" in md         # A40 rows
    assert "Mean |err|" in md
    # feedback YAML: A5000 tp2 averaged over the two records
    a5000 = {(r["model"], r["tp"]): r["avg_w"] for r in drafts["A5000"]["measured"]}
    assert a5000[(MODEL, 2)] == pytest.approx((391 + 414) / 2)
    assert a5000[(MODEL, 1)] == 203
    # drafts validate against the power-profile schema fragment
    from planner.power_profiles import MeasuredPoint
    for r in drafts["A5000"]["measured"]:
        MeasuredPoint.model_validate(r)


def test_calibration_cli(tmp_path):
    hist = tmp_path / "h.jsonl"
    hist.write_text("\n".join(json.dumps(_rec(i)) for i in range(5)))
    out = tmp_path / "r.md"
    rc = pcr.main(["--history", str(hist), "--out", str(out),
                   "--yaml-out", str(tmp_path / "drafts")])
    assert rc == 0
    assert "Power calibration report" in out.read_text()
    assert (tmp_path / "drafts" / "A40.measured.yaml").is_file()


# ---- termination history hook ------------------------------------------------------

def test_on_stopped_hook_appends_history(tmp_path):
    from service.api.deployment_routes import make_on_stopped
    from service.api.routes import ServiceState
    from service.deploy.docker_driver import FakeDriver
    from service.deploy.manager import DeploymentManager
    from service.deploy.store import DeployStore
    from service.inventory.ledger import Ledger
    from service.inventory.registry import ClusterRegistry
    from service.monitor.runtime import MonitorRuntime

    reg = ClusterRegistry.model_validate({"nodes": [
        {"id": "n0", "devices": [{"name": "A40", "count": 2, "mem_gb": 48}]}]})
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    store = DeployStore(tmp_path / "l.sqlite")
    state = ServiceState(registry=reg, ledger=ledger)
    state.deploy_store = store
    state.monitor_runtime = MonitorRuntime(store, interval_s=0)
    hist = tmp_path / "hist.jsonl"

    ver, free = ledger.snapshot()
    res = ledger.reserve("t", "j", free[:2], ver)
    mgr = DeploymentManager(store, FakeDriver(), ledger,
                            health_fn=lambda u: True, run_async=False,
                            sleep=lambda s: None,
                            on_stopped=make_on_stopped(state, hist))
    spec = build_spec("dep-cal", MODEL, free[:2], tp=2, port=8001)
    spec.predicted_power_w = 600.0
    dep_id = mgr.create(res.id, "t", spec)
    # feed the energy integral so measured power is nonzero
    buf = state.monitor_runtime.buf(dep_id)
    buf.push_power(0.0, 660.0)
    buf.push_power(3600.0, 660.0)
    mgr.terminate(dep_id, drain_timeout_s=0)

    rec = json.loads(hist.read_text().strip())
    assert rec["dep_id"] == dep_id and rec["hw"] == "A40" and rec["tp"] == 2
    assert rec["predicted_power_w"] == 600.0
    assert rec["measured_avg_w"] == pytest.approx(660.0)
    assert rec["energy_wh"] == pytest.approx(660.0)
