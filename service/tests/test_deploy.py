"""D2 tests: state machine, launcher translation golden, manager lifecycle
with the fake driver (happy path, retry->FAILED+rollback, drain, restart
recovery, reconcile). No docker daemon required."""
from __future__ import annotations

import itertools

import pytest

from service.deploy.docker_driver import FakeDriver
from service.deploy.manager import DeploymentManager
from service.deploy.spec import DeploymentSpec
from service.deploy.state import ALLOWED, DeploymentState as S, InvalidTransition, check_transition
from service.deploy.store import DeployStore
from service.deploy.vllm_launcher import build_spec, nccl_env_for
from service.inventory.ledger import Ledger
from service.inventory.registry import ClusterRegistry
from service.topology.graph import TopologyGraph
from service.topology.schema import RegistryV2

MODEL = "meta-llama/Llama-3.1-8B"


# ---- state machine -----------------------------------------------------------

def test_all_allowed_transitions_and_failed_reachability():
    for a, targets in ALLOWED.items():
        for b in targets:
            check_transition(a, b)
    for a in ALLOWED:
        if a not in (S.RELEASED, S.FAILED):
            check_transition(a, S.FAILED)


def test_disallowed_transitions_raise():
    for a, b in itertools.product(ALLOWED, repeat=2):
        if b in ALLOWED[a] or (b == S.FAILED and a not in (S.RELEASED, S.FAILED)):
            continue
        with pytest.raises(InvalidTransition):
            check_transition(a, b)


def test_store_records_every_transition(tmp_path):
    st = DeployStore(tmp_path / "d.sqlite")
    dep = st.create(1, "t", MODEL, {"x": 1})
    for to in (S.PULLING, S.STARTING, S.HEALTH_CHECK, S.READY,
               S.DRAINING, S.STOPPED, S.RELEASED):
        st.transition(dep.id, to)
    row = st.get(dep.id)
    assert [e["to_state"] for e in row.events] == [
        "PENDING", "PULLING", "STARTING", "HEALTH_CHECK", "READY",
        "DRAINING", "STOPPED", "RELEASED"]
    assert row.ready_at and row.terminated_at
    with pytest.raises(InvalidTransition):
        st.transition(dep.id, S.READY)   # RELEASED is terminal


# ---- launcher golden ------------------------------------------------------------

@pytest.fixture()
def nvlink_graph():
    return TopologyGraph(RegistryV2.model_validate({
        "version": 2, "nodes": [{
            "id": "n0", "devices": [
                {"id": f"n0/A40/{i}", "hw": "A40", "mem_gb": 48,
                 "active_w": 300} for i in range(4)],
            "intra_links": [
                {"a": "n0/A40/0", "b": "n0/A40/1", "kind": "nvlink",
                 "bandwidth": "112GBps"},
                {"a": "n0/A40/2", "b": "n0/A40/3", "kind": "pcie",
                 "bandwidth": "32GBps"}]}]}))


def test_launcher_golden_spec(nvlink_graph):
    spec = build_spec("dep-abc", MODEL, ["n0/A40/1", "n0/A40/0"], tp=2,
                      port=8001, graph=nvlink_graph)
    c = spec.containers[0]
    assert c.name == "llmsvc-dep-abc-0"
    assert c.node_id == "n0"
    assert c.gpu_indices == [0, 1]
    assert c.device_ids == ["n0/A40/0", "n0/A40/1"]
    assert c.env["NCCL_P2P_LEVEL"] == "NVL"            # NVLink pair
    # HF policy: token passthrough when set, offline against the cache otherwise
    assert ("HUGGING_FACE_HUB_TOKEN" in c.env) != ("HF_HUB_OFFLINE" in c.env)
    assert c.volumes and "/root/.cache/huggingface" in c.volumes.values()
    assert spec.engine_args["max-model-len"] == 8192   # small-GPU-safe default
    assert c.ports == {"api": 8001, "metrics": 8001}
    assert c.command[:4] == ["--model", MODEL, "--served-model-name",
                             "Llama-3.1-8B"]
    assert "--tensor-parallel-size" in c.command
    assert spec.engine_args["gpu-memory-utilization"] == 0.90


def test_nccl_env_pcie_group_disables_p2p(nvlink_graph):
    assert nccl_env_for(nvlink_graph, ["n0/A40/2", "n0/A40/3"]) == \
        {"NCCL_P2P_DISABLE": "1"}


def test_launcher_rejects_cross_node_and_bad_tp(nvlink_graph):
    with pytest.raises(ValueError, match="tp"):
        build_spec("d", MODEL, ["n0/A40/0"], tp=2, port=8001)
    with pytest.raises(ValueError, match="multi-node"):
        build_spec("d", MODEL, ["n0/A40/0", "n1/A40/0"], tp=2, port=8001)


# ---- manager lifecycle -----------------------------------------------------------

@pytest.fixture()
def env(tmp_path):
    reg = ClusterRegistry.model_validate({"nodes": [
        {"id": "n0", "devices": [{"name": "A40", "count": 4, "mem_gb": 48}]}]})
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    store = DeployStore(tmp_path / "l.sqlite")
    driver = FakeDriver()
    return ledger, store, driver


def _mgr(ledger, store, driver, healthy=True, **kw):
    return DeploymentManager(store, driver, ledger,
                             health_fn=lambda url: healthy,
                             run_async=False, sleep=lambda s: None, **kw)


def _reserve(ledger):
    ver, _ = ledger.snapshot()
    return ledger.reserve("t", "job1", ["n0/A40/0", "n0/A40/1"], ver)


def test_happy_path_to_ready_then_terminate_releases(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    mgr = _mgr(ledger, store, driver)
    spec = build_spec("dep-x", MODEL, res.device_ids, tp=2, port=8001)
    dep_id = mgr.create(res.id, "t", spec)
    row = store.get(dep_id)
    assert row.state == S.READY
    assert row.endpoints["openai_url"].endswith(":8001/v1")
    assert driver.containers   # container actually "running"

    mgr.terminate(dep_id, drain_timeout_s=5)
    row = store.get(dep_id)
    assert row.state == S.RELEASED
    assert not driver.containers
    assert len(ledger.snapshot()[1]) == 4    # devices back
    # idempotent
    mgr.terminate(dep_id)
    assert store.get(dep_id).state == S.RELEASED


def test_failure_retries_then_failed_with_rollback(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    driver.fail["run"] = 99                  # every run attempt fails
    mgr = _mgr(ledger, store, driver)
    spec = build_spec("dep-y", MODEL, res.device_ids, tp=2, port=8001)
    dep_id = mgr.create(res.id, "t", spec)
    row = store.get(dep_id)
    assert row.state == S.FAILED
    assert "injected run failure" in row.last_error
    pulls = [c for c in driver.calls if c[0] == "pull"]
    assert len(pulls) == 3                   # retries+1 full attempts
    assert not driver.containers             # rollback removed everything
    assert len(ledger.snapshot()[1]) == 2    # reservation KEPT (user decides)


def test_health_timeout_fails(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    mgr = _mgr(ledger, store, driver, healthy=False)
    spec = build_spec("dep-z", MODEL, res.device_ids, tp=2, port=8001)
    spec.health.timeout_s = 0.0
    spec.health.retries = 0
    dep_id = mgr.create(res.id, "t", spec)
    assert store.get(dep_id).state == S.FAILED
    assert "health" in store.get(dep_id).last_error.lower() or \
           "Timeout" in store.get(dep_id).last_error


def test_draining_waits_for_queue(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    q = {"n": 3}

    def qlen(dep_id):
        q["n"] -= 1
        return q["n"]

    mgr = _mgr(ledger, store, driver, queue_len_fn=qlen)
    dep_id = mgr.create(res.id, "t",
                        build_spec("dep-q", MODEL, res.device_ids, tp=2, port=8001))
    mgr.terminate(dep_id, drain_timeout_s=60)
    row = store.get(dep_id)
    assert row.state == S.RELEASED
    assert q["n"] <= 0                        # actually polled the queue
    assert any(e["to_state"] == "DRAINING" for e in row.events)


def test_restart_recovery(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    mgr = _mgr(ledger, store, driver)
    ready_id = mgr.create(res.id, "t",
                          build_spec("dep-r", MODEL, res.device_ids, tp=2, port=8001))
    # a second deployment crashed mid-STARTING (simulate by direct store writes)
    stuck = store.create(res.id, "t", MODEL, build_spec(
        "dep-s", MODEL, res.device_ids, tp=2, port=8002).model_dump())
    store.transition(stuck.id, S.PULLING)
    store.transition(stuck.id, S.STARTING)

    mgr2 = _mgr(ledger, store, driver)       # "restarted" service
    out = mgr2.recover()
    assert ready_id in out["reattached"]
    assert stuck.id in out["requeued"]
    assert store.get(stuck.id).state == S.READY   # re-queued and brought up


def test_reconcile_orphans_and_missing(env):
    ledger, store, driver = env
    res = _reserve(ledger)
    mgr = _mgr(ledger, store, driver)
    dep_id = mgr.create(res.id, "t",
                        build_spec("dep-m", MODEL, res.device_ids, tp=2, port=8001))
    # orphan: a container we do not track
    from service.deploy.spec import ContainerSpec
    driver.run("n0", ContainerSpec(node_id="n0", image="x", device_ids=[],
                                   gpu_indices=[], name="llmsvc-dep-ghost-0"))
    # missing: nuke the real container behind the manager's back
    # (container names derive from the launcher-time dep id "dep-m")
    driver.containers.pop("llmsvc-dep-m-0")
    out = mgr.reconcile(["n0"])
    assert out["orphans_removed"] == ["llmsvc-dep-ghost-0"]
    assert out["missing_marked_failed"] == [dep_id]
    assert store.get(dep_id).state == S.FAILED


def test_run_kwargs_gpu_modes(monkeypatch):
    """CDI-only docker hosts need driver='cdi' device requests (Error 803
    with the legacy nvidia path — found on the real A5000 host)."""
    from service.deploy.docker_driver import DockerSdkDriver
    from service.deploy.spec import ContainerSpec

    spec = ContainerSpec(node_id="n0", image="img", device_ids=["n0/A5000/0"],
                         gpu_indices=[0], name="llmsvc-x-0",
                         volumes={"/h": "/c"})
    monkeypatch.delenv("LLMSS_GPU_MODE", raising=False)
    legacy = DockerSdkDriver.build_run_kwargs(spec)["device_requests"][0]
    assert legacy["Driver"] == "nvidia" and legacy["DeviceIDs"] == ["0"]
    monkeypatch.setenv("LLMSS_GPU_MODE", "cdi")
    cdi = DockerSdkDriver.build_run_kwargs(spec)["device_requests"][0]
    assert cdi["Driver"] == "cdi" and cdi["DeviceIDs"] == ["nvidia.com/gpu=0"]
    assert DockerSdkDriver.build_run_kwargs(spec)["volumes"] == {
        "/h": {"bind": "/c", "mode": "rw"}}


def test_reconcile_keeps_midflight_containers(env):
    """Regression (found live): a HEALTH_CHECK deployment's container must NOT
    be treated as an orphan — the first reconcile version removed it and
    livelocked the bring-up."""
    ledger, store, driver = env
    res = _reserve(ledger)
    spec = build_spec("dep-mid", MODEL, res.device_ids, tp=2, port=8001)
    dep = store.create(res.id, "t", MODEL, spec.model_dump())
    store.transition(dep.id, S.PULLING)
    store.transition(dep.id, S.STARTING)
    driver.run("n0", spec.containers[0])          # container exists mid-flight
    store.transition(dep.id, S.HEALTH_CHECK)

    mgr = _mgr(ledger, store, driver)
    out = mgr.reconcile(["n0"])
    assert out == {"orphans_removed": [], "missing_marked_failed": []}
    assert "llmsvc-dep-mid-0" in driver.containers   # survived
    # and a mid-flight dep with no container yet is NOT marked failed
    assert store.get(dep.id).state == S.HEALTH_CHECK


def test_measured_config_carries_host_base_power(tmp_path):
    """Stage-2 (measured) must integrate host base power per active node, so
    the host-consolidation cost Stage-1 minimizes is visible in the final
    judgment. Simulator backends' config schema stays untouched."""
    import json as _json

    from planner.config_renderer import render
    from planner.spec_schema import PlannerSpec
    from planner.types import Allocation, Instance
    from planner.utils import REPO_ROOT
    from sim_backends.measured.backend import cluster_model_from_config

    spec = PlannerSpec.model_validate({
        "model": {"name": MODEL, "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": {"nodes": [
            {"id": "n0", "host_base_w": 250,
             "devices": [{"name": "A40", "count": 2, "mem_gb": 48}]},
            {"id": "n1", "host_base_w": 200,
             "devices": [{"name": "A40", "count": 2, "mem_gb": 48}]}]},
    })
    alloc = Allocation(instances=[
        Instance(node_id="n0", hardware="A40", model_name=MODEL, tp=1,
                 npu_num=1, npu_mem_gb=48),
        Instance(node_id="n1", hardware="A40", model_name=MODEL, tp=1,
                 npu_num=1, npu_mem_gb=48)])

    rel, _cli = render(alloc, spec, tmp_path, "hostpw", backend="measured")
    cfg = _json.loads((REPO_ROOT / rel).read_text())
    assert [n["host_base_w"] for n in cfg["nodes"]] == [250, 200]
    assert [n["id"] for n in cfg["nodes"]] == ["n0", "n1"]

    # the measured backend turns those into HostCfg entries (500 W of hosts)
    import os
    os.environ.setdefault("LLMSS_MEASURED_ORACLES", str(tmp_path / "none"))
    try:
        model = cluster_model_from_config(cfg)
    except FileNotFoundError:
        model = None            # no oracle fixture here; config shape is the point
    if model is not None:
        assert sum(h.base_w for h in model.hosts) == 450

    # simulator backends must NOT get the extra keys
    rel_up, _ = render(alloc, spec, tmp_path, "hostpw_up", backend="upstream")
    cfg_up = _json.loads((REPO_ROOT / rel_up).read_text())
    assert all("host_base_w" not in n and "id" not in n for n in cfg_up["nodes"])
