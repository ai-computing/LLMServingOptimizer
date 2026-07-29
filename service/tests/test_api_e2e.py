"""M9 sim E2E: real pipeline, no planner mock — inventory snapshot → workload
synthesis → fidelity routing → power-min planner → measured-backend Stage-2
(seconds per candidate) → recommendation → confirm → release.

Marked sim (uses the real planner + measured backend in a subprocess); no
hardware and no ASTRA-Sim builds are required, so this stays fast (<2 min).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.api.routes import ServiceState, create_service_router
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry

pytestmark = pytest.mark.sim

MODEL = "meta-llama/Llama-3.1-8B"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    """Real-planner service over a 4xA40 + 2xRNGD registry with full oracle
    coverage (A40 curve copied from the repo, RNGD bucket synthesized)."""
    oracle_root = tmp_path / "oracles"
    a40_dst = oracle_root / "A40" / MODEL
    a40_dst.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "profiles/measured/A40" / MODEL / "tp1.yaml",
                a40_dst / "tp1.yaml")
    rngd_dst = oracle_root / "RNGD" / MODEL
    rngd_dst.mkdir(parents=True)
    (rngd_dst / "tp1.yaml").write_text(yaml.safe_dump({
        "kind": "npu_bucket", "hw": "RNGD", "model": MODEL, "tp": 1,
        "idle_w": 15, "max_concurrency": 16,
        "typical_input_toks": 256, "typical_output_toks": 256,
        "meta": {"stack": "synthetic-fixture", "measured_at": "2026-07-29"},
        "buckets": [
            {"size": 2048, "ttft_ms": 120, "tbt_ms": 12, "avg_w": 150},
            {"size": 4096, "ttft_ms": 160, "tbt_ms": 16, "avg_w": 170},
        ]}))
    monkeypatch.setenv("LLMSS_MEASURED_ORACLES", str(oracle_root))

    registry = ClusterRegistry.model_validate({
        "nodes": [
            {"id": "node0", "host_base_w": 250, "devices": [
                {"name": "A40", "count": 4, "mem_gb": 48}]},
            {"id": "node1", "host_base_w": 200, "devices": [
                {"name": "RNGD", "count": 2, "mem_gb": 48}]},
        ],
        "links": [{"src": "node0", "dst": "node1",
                   "bandwidth": "200Gbps", "latency": "0.0005ms"}],
    })
    ledger = Ledger(tmp_path / "ledger.sqlite", registry)
    state = ServiceState(registry=registry, ledger=ledger,
                         out_root=tmp_path / "jobs", run_async=False)
    app = FastAPI()
    app.include_router(create_service_router(state))
    return TestClient(app), state


def test_npu_cluster_auto_routes_to_measured_and_full_cycle(svc):
    client, state = svc
    r = client.post("/api/serve-requests", json={
        "model": MODEL, "tenant": "tenantA",
        "slo": {"tpot_ms": 200},
        "scale": {"req_per_s": 2, "preset": "chat", "duration_s": 10},
        "num_req_eval": 20,
    })
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["state"] == "done", job
    res = job["result"]
    # P1 DoD: NPU-containing request is auto-routed to the measured backend
    assert res["backend"] == "measured"
    assert res["confidence"] == "high"
    best = res["best"]
    assert best["passed"] and best["power_w"] is not None
    assert res["device_ids"], "winning allocation must map to concrete devices"

    # confirm reserves; a second tenant then plans on the remainder only
    r = client.post(f"/api/serve-requests/{job['id']}/confirm")
    assert r.json()["state"] == "confirmed", r.text
    free_after = client.get("/api/cluster").json()["free_devices"]
    assert len(free_after) == 6 - len(res["device_ids"])

    r2 = client.post("/api/serve-requests", json={
        "model": MODEL, "tenant": "tenantB",
        "scale": {"req_per_s": 1, "preset": "chat", "duration_s": 10},
        "num_req_eval": 10,
    })
    job2 = r2.json()
    assert job2["state"] in ("done", "infeasible")
    if job2["state"] == "done":
        overlap = set(job2["result"]["device_ids"]) & set(res["device_ids"])
        assert not overlap, "tenant B must not be offered tenant A's devices"

    # release restores capacity
    client.post(f"/api/serve-requests/{job['id']}/release")
    assert len(client.get("/api/cluster").json()["free_devices"]) == 6
