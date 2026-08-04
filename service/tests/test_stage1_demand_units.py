"""Stage-1 demand must be a generation rate, and measured capacity must reach it.

A Qwen-14B AWQ request came back "달성 불가 - demand: 991 > 867 toks/s". Two
independent defects: the service sent sum(in+out) while every ceiling in Stage-1
(_PROXY_TOKS_PER_UNIT, the oracles' thr_toks_s = vLLM output_throughput) is an
output rate, and the ceiling itself ignored the measured capacity curves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.api.routes import ServiceState, create_service_router, measured_capacity
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry

MODEL = "meta-llama/Llama-3.1-8B"
REG = {"nodes": [
    {"id": "node0", "devices": [{"name": "A40", "count": 8, "mem_gb": 48}]}]}


@dataclass
class _FakeResult:
    infeasible_report: object = None
    best: object = None
    candidates: list = field(default_factory=list)


@pytest.fixture()
def captured(tmp_path, monkeypatch):
    """Run the real recommend path but stop at Stage-1's input."""
    import planner.search_orchestrator as orch
    seen: dict = {}

    def fake_run_spec(spec, **kw):
        seen["spec"] = spec
        seen["out_dir"] = kw.get("out_dir")
        return _FakeResult()

    monkeypatch.setattr(orch, "run_spec", fake_run_spec)
    registry = ClusterRegistry.model_validate(REG)
    state = ServiceState(registry=registry,
                         ledger=Ledger(tmp_path / "l.sqlite", registry),
                         out_root=tmp_path / "jobs", run_async=False)
    app = FastAPI()
    app.include_router(create_service_router(state))
    return TestClient(app), seen


def test_stage1_demand_is_the_output_rate_not_input_plus_output(captured):
    client, seen = captured
    r = client.post("/api/serve-requests", json={
        "model": MODEL, "scale": {"req_per_s": 2, "preset": "chat",
                                  "duration_s": 30}})
    assert r.status_code == 200, r.text
    spec = seen["spec"]

    # recompute both rates from the workload the service actually synthesized
    rows = [json.loads(l) for l in
            (Path(seen["out_dir"]) / "workload.jsonl").read_text().splitlines() if l]
    duration_s = 30.0                      # the requested window, as synthesized
    out_toks = sum(x["output_toks"] for x in rows)
    total_toks = out_toks + sum(x["input_toks"] for x in rows)

    demand = spec.requirements.demand.toks_per_s
    # the chat preset is ~1:1, so the wrong unit is ~2x the right one
    assert demand == pytest.approx(out_toks / duration_s, rel=0.02)
    assert demand < 0.65 * (total_toks / duration_s)


def test_measured_capacity_is_handed_to_stage1(captured):
    client, seen = captured
    client.post("/api/serve-requests", json={
        "model": MODEL, "scale": {"req_per_s": 2, "preset": "chat"}})
    cap = seen["spec"].search_space.hw_tp_capacity
    # A40 has a tp1 oracle in this checkout; if oracles are absent the field is
    # simply None and Stage-1 falls back to the proxy
    if cap is None:
        pytest.skip("no measured oracles in this checkout")
    for hw, per_tp in cap.items():
        assert per_tp and all(v > 0 for v in per_tp.values())


def test_measured_capacity_reads_the_oracle_peaks():
    cap = measured_capacity("Qwen/Qwen2.5-14B-Instruct-AWQ", {"A5000": [1, 2]})
    if cap is None:
        pytest.skip("A5000 AWQ oracles not present")
    # the measured curve peaks at c=64: 1100 toks/s at tp1, 1296 at tp2 -- the
    # sublinear TP payoff the analytical proxy cannot express
    assert cap["A5000"][1] == pytest.approx(1100, rel=0.02)
    assert cap["A5000"][2] == pytest.approx(1296, rel=0.02)
    assert cap["A5000"][2] < 2 * cap["A5000"][1]


def test_unmeasured_model_yields_no_capacity_override():
    assert measured_capacity("no/such-model", {"A5000": [1, 2]}) is None
