"""D2 API tests: confirm -> auto-deploy chain and the /api/deployments REST
surface, all against the fake driver (no dockerd)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.api.deployment_routes import create_deployment_router
from service.api.routes import ServiceState, create_service_router
from service.deploy.docker_driver import FakeDriver
from service.deploy.manager import DeploymentManager
from service.deploy.store import DeployStore
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry

MODEL = "meta-llama/Llama-3.1-8B"


def mock_planner(req, topology, snapshot_ver, job):
    return {"backend": "measured", "confidence": "high", "reason": "mock",
            "snapshot_ver": snapshot_ver,
            "best": {"run_id": "c1", "hw_summary": "A40x1(tp2)", "passed": True,
                     "power_w": 600.0, "power_source": "sim_energy",
                     "metrics": {}, "violations": []},
            "alternatives": [], "_need": {"node0|A40": 2},
            "_groups": [["node0", "A40", 2]]}


@pytest.fixture()
def svc(tmp_path):
    reg = ClusterRegistry.model_validate({"nodes": [
        {"id": "node0", "devices": [{"name": "A40", "count": 4, "mem_gb": 48}]}]})
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    state = ServiceState(registry=reg, ledger=ledger, planner_fn=mock_planner,
                         run_async=False)
    state.deploy_store = DeployStore(tmp_path / "l.sqlite")
    driver = FakeDriver()
    state.deploy_manager = DeploymentManager(
        state.deploy_store, driver, ledger, health_fn=lambda url: True,
        run_async=False, sleep=lambda s: None)
    app = FastAPI()
    app.include_router(create_service_router(state))
    app.include_router(create_deployment_router(state))
    return TestClient(app), state, driver


def test_confirm_auto_deploys_to_ready_and_terminate_releases(svc):
    client, state, driver = svc
    job = client.post("/api/serve-requests", json={
        "model": MODEL, "scale": {"req_per_s": 2, "preset": "chat"}}).json()
    assert job["state"] == "done"

    conf = client.post(f"/api/serve-requests/{job['id']}/confirm").json()
    dep_id = conf["deployment_id"]
    assert conf["state"] == "confirmed" and dep_id

    d = client.get(f"/api/deployments/{dep_id}").json()
    assert d["state"] == "READY"
    assert d["endpoints"]["openai_url"].startswith("http://")
    assert d["device_ids"] == ["node0/A40/0", "node0/A40/1"]
    assert any(e["to_state"] == "READY" for e in d["events"])
    assert driver.containers   # container running on the fake dockerd

    lst = client.get("/api/deployments").json()["deployments"]
    assert [x["id"] for x in lst] == [dep_id]

    r = client.post(f"/api/deployments/{dep_id}/terminate",
                    json={"drain_timeout_s": 1})
    assert r.status_code == 200
    assert client.get(f"/api/deployments/{dep_id}").json()["state"] == "RELEASED"
    assert not driver.containers
    assert len(client.get("/api/cluster").json()["free_devices"]) == 4
    # terminated deployments drop out of the default list
    assert client.get("/api/deployments").json()["deployments"] == []
    assert client.get("/api/deployments?include_terminated=true") \
        .json()["deployments"]


def test_auto_deploy_skipped_for_multi_group(svc):
    client, state, _ = svc
    state.planner_fn = lambda req, t, v, j: {
        **mock_planner(req, t, v, j),
        "_need": {"node0|A40": 2},
        "_groups": [["node0", "A40", 1], ["node0", "A40", 1]]}
    job = client.post("/api/serve-requests", json={
        "model": MODEL, "scale": {"req_per_s": 2, "preset": "chat"}}).json()
    conf = client.post(f"/api/serve-requests/{job['id']}/confirm").json()
    assert conf["state"] == "confirmed"
    assert conf["deployment_id"] is None      # P0 limit, reservation intact


def test_opt_out_auto_deploy(svc):
    client, _, driver = svc
    job = client.post("/api/serve-requests", json={
        "model": MODEL, "auto_deploy": False,
        "scale": {"req_per_s": 2, "preset": "chat"}}).json()
    conf = client.post(f"/api/serve-requests/{job['id']}/confirm").json()
    assert conf["deployment_id"] is None and not driver.containers


def test_unknown_deployment_404(svc):
    client, _, _ = svc
    assert client.get("/api/deployments/dep-nope").status_code == 404
    assert client.post("/api/deployments/dep-nope/terminate").status_code == 404
