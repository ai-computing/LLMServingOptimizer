"""M9 tests: service API — submit/result/confirm/release flow, validation
failures, infeasible reporting, and the confirm-conflict auto-replan path.
Planner is mocked; runs synchronously for determinism."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.api.models import ServeRequestIn
from service.api.routes import Job, ServiceState, create_service_router
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry

MODEL = "meta-llama/Llama-3.1-8B"

REG = {
    "nodes": [
        {"id": "node0", "devices": [{"name": "A40", "count": 4, "mem_gb": 48}]},
        {"id": "node1", "devices": [{"name": "RNGD", "count": 2, "mem_gb": 48}]},
    ],
}


def mock_planner_ok(req: ServeRequestIn, topology: dict, snapshot_ver: int,
                    job: Job) -> dict:
    """Recommends 2x A40 on node0."""
    return {
        "backend": "measured", "confidence": "high", "reason": "mock",
        "snapshot_ver": snapshot_ver, "demand_toks_per_s": 1000.0,
        "best": {"run_id": "cand_1", "hw_summary": "A40x2(tp1)", "passed": True,
                 "power_w": 650.0, "power_source": "sim_energy",
                 "metrics": {"ttft_ms": 100.0}, "violations": []},
        "alternatives": [],
        "_need": {"node0|A40": 2},
    }


def mock_planner_infeasible(req, topology, snapshot_ver, job) -> dict:
    return {
        "backend": "upstream", "confidence": "medium", "reason": "mock",
        "snapshot_ver": snapshot_ver, "best": None, "alternatives": [],
        "device_ids": [],
        "infeasible": {"bottleneck": "demand",
                       "detail": "demand exceeds capacity",
                       "max_achievable_toks_s": 500.0,
                       "suggestions": ["reduce demand"]},
    }


@pytest.fixture()
def svc(tmp_path):
    registry = ClusterRegistry.model_validate(REG)
    ledger = Ledger(tmp_path / "ledger.sqlite", registry)
    state = ServiceState(registry=registry, ledger=ledger,
                         out_root=tmp_path / "jobs",
                         planner_fn=mock_planner_ok, run_async=False)
    app = FastAPI()
    app.include_router(create_service_router(state))
    return TestClient(app), state


def _submit(client, **overrides):
    body = {"model": MODEL, "scale": {"req_per_s": 4, "preset": "chat"}}
    body.update(overrides)
    r = client.post("/api/serve-requests", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_happy_path_submit_confirm_release(svc):
    client, state = svc
    ver0 = client.get("/api/cluster").json()["snapshot_ver"]
    free0 = len(client.get("/api/cluster").json()["free_devices"])

    job = _submit(client)
    assert job["state"] == "done"
    best = job["result"]["best"]
    assert best["power_w"] == 650.0
    assert len(job["result"]["device_ids"]) == 2

    r = client.post(f"/api/serve-requests/{job['id']}/confirm")
    assert r.status_code == 200 and r.json()["state"] == "confirmed"

    cluster = client.get("/api/cluster").json()
    assert cluster["snapshot_ver"] == ver0 + 1
    assert len(cluster["free_devices"]) == free0 - 2
    assert cluster["reservations"][0]["tenant"] == "default"

    r = client.post(f"/api/serve-requests/{job['id']}/release")
    assert r.json()["state"] == "released"
    assert len(client.get("/api/cluster").json()["free_devices"]) == free0


def test_validation_error_bad_request(svc):
    client, _ = svc
    r = client.post("/api/serve-requests",
                    json={"model": MODEL, "scale": {"req_per_s": -1}})
    assert r.status_code == 422


def test_unknown_job_404(svc):
    client, _ = svc
    assert client.get("/api/serve-requests/nope").status_code == 404
    assert client.post("/api/serve-requests/nope/confirm").status_code == 404


def test_infeasible_result_carries_bottleneck(svc):
    client, state = svc
    state.planner_fn = mock_planner_infeasible
    job = _submit(client)
    assert job["state"] == "infeasible"
    inf = job["result"]["infeasible"]
    assert inf["bottleneck"] == "demand"
    assert inf["suggestions"]
    r = client.post(f"/api/serve-requests/{job['id']}/confirm")
    assert r.status_code == 409  # nothing to confirm


def test_tenant_b_plans_on_remaining_resources(svc):
    """P1 DoD: after tenant A reserves, tenant B's planner sees only leftovers."""
    client, state = svc
    seen_topologies = []

    def spy_planner(req, topology, ver, job):
        seen_topologies.append(topology)
        return mock_planner_ok(req, topology, ver, job)

    state.planner_fn = spy_planner
    job_a = _submit(client, tenant="A")
    client.post(f"/api/serve-requests/{job_a['id']}/confirm")
    _submit(client, tenant="B")

    a40_counts = [sum(d["count"] for n in t["nodes"] for d in n["devices"]
                      if d["name"] == "A40") for t in seen_topologies]
    assert a40_counts == [4, 2]  # B planned against 4-2=2 free A40s


def test_confirm_conflict_triggers_single_replan(svc):
    """Two jobs recommended from the same snapshot; the second confirm hits a
    stale snapshot version and must come back 'replanned', not crash."""
    client, state = svc
    job1 = _submit(client, tenant="A")
    job2 = _submit(client, tenant="B")   # same snapshot_ver as job1

    r1 = client.post(f"/api/serve-requests/{job1['id']}/confirm")
    assert r1.json()["state"] == "confirmed"

    r2 = client.post(f"/api/serve-requests/{job2['id']}/confirm")
    body = r2.json()
    assert body["replanned"] is True
    assert body["state"] == "replanned"
    # the re-planned recommendation is confirmable now
    r3 = client.post(f"/api/serve-requests/{job2['id']}/confirm")
    assert r3.json()["state"] == "confirmed"
    # A and B hold disjoint devices
    cluster = client.get("/api/cluster").json()
    held = [set(res["device_ids"]) for res in cluster["reservations"]]
    assert len(held) == 2 and not (held[0] & held[1])


def test_models_endpoint_lists_catalog(svc):
    client, _ = svc
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    # A40 measured oracle (from M7) and/or legacy profiles must appear for 8B
    assert any(MODEL == name for name in models), models.keys()
    assert "A40" in models[MODEL]


def test_sse_events_stream_ends(svc):
    client, _ = svc
    job = _submit(client)
    with client.stream("GET", f"/api/serve-requests/{job['id']}/events") as r:
        text = "".join(chunk for chunk in r.iter_text())
    assert '"type": "end"' in text or '"type": "state"' in text
