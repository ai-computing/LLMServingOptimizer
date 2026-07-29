"""M8 tests: registry loading, ledger reserve/release cycles, optimistic
locking, thread-race safety, and planner-topology views."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from planner.spec_schema import PlannerSpec
from service.inventory.ledger import ConflictError, Ledger
from service.inventory.registry import ClusterRegistry, load_registry
from service.inventory.views import available_topology, free_counts, pick_devices

REG = {
    "nodes": [
        {"id": "node0", "host_base_w": 250, "devices": [
            {"name": "A40", "count": 4, "mem_gb": 48}]},
        {"id": "node1", "devices": [
            {"name": "RNGD", "count": 2, "mem_gb": 48}]},
    ],
    "links": [{"src": "node0", "dst": "node1",
               "bandwidth": "200Gbps", "latency": "0.0005ms"}],
    "intra_node_bandwidth": "600GBps",
}


@pytest.fixture()
def registry() -> ClusterRegistry:
    return ClusterRegistry.model_validate(REG)


@pytest.fixture()
def ledger(registry, tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite", registry)


def test_registry_load_and_device_ids(registry, tmp_path):
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(REG))
    assert load_registry(p) == registry
    ids = registry.device_ids()
    assert len(ids) == 6
    assert "node0/A40/3" in ids and "node1/RNGD/1" in ids


def test_reserve_release_cycle_restores_free(ledger):
    ver, free = ledger.snapshot()
    assert len(free) == 6
    res = ledger.reserve("tenantA", "req1", ["node0/A40/0", "node0/A40/1"], ver)
    assert res.state == "reserved"
    ver2, free2 = ledger.snapshot()
    assert ver2 == ver + 1 and len(free2) == 4
    assert "node0/A40/0" not in free2
    assert ledger.release("req1") is True
    ver3, free3 = ledger.snapshot()
    assert ver3 == ver2 + 1 and len(free3) == 6


def test_optimistic_locking_stale_snapshot(ledger):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node0/A40/0"], ver)
    # second reservation still holding the OLD snapshot version -> conflict,
    # even though the devices themselves would be free
    with pytest.raises(ConflictError, match="stale"):
        ledger.reserve("B", "r2", ["node0/A40/1"], ver)
    ver2, _ = ledger.snapshot()
    ledger.reserve("B", "r2", ["node0/A40/1"], ver2)  # fresh snapshot works


def test_device_double_booking_detected(ledger):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node0/A40/0"], ver)
    ver2, _ = ledger.snapshot()
    with pytest.raises(ConflictError, match="already reserved"):
        ledger.reserve("B", "r2", ["node0/A40/0"], ver2)


def test_concurrent_reservation_race_exactly_one_wins(ledger):
    ver, _ = ledger.snapshot()

    def contend(i):
        try:
            ledger.reserve(f"t{i}", f"race{i}", ["node1/RNGD/0"], ver)
            return True
        except ConflictError:
            return False

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(contend, range(10)))
    assert sum(results) == 1


def test_double_release_idempotent(ledger):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node0/A40/0"], ver)
    assert ledger.release("r1") is True
    assert ledger.release("r1") is False
    assert ledger.release("never-existed") is False


def test_duplicate_request_id_rejected(ledger):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node0/A40/0"], ver)
    ver2, _ = ledger.snapshot()
    with pytest.raises(ConflictError, match="already has a"):
        ledger.reserve("A", "r1", ["node0/A40/1"], ver2)


def test_unknown_device_rejected(ledger):
    ver, _ = ledger.snapshot()
    with pytest.raises(ValueError, match="unknown devices"):
        ledger.reserve("A", "r1", ["node9/H100/0"], ver)


def test_partial_reservation_topology_validates_as_planner_spec(ledger, registry):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node0/A40/0", "node0/A40/1", "node0/A40/2"], ver)
    _, free = ledger.snapshot()
    topo = available_topology(registry, free)
    a40 = [d for n in topo["nodes"] if n["id"] == "node0" for d in n["devices"]]
    assert a40 == [{"name": "A40", "count": 1, "mem_gb": 48}]
    assert topo["nodes"][0].get("host_base_w") == 250

    spec = PlannerSpec.model_validate({
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": topo,
    })
    assert spec.topology.nodes[0].devices[0].count == 1


def test_fully_reserved_node_dropped_and_links_pruned(ledger, registry):
    ver, _ = ledger.snapshot()
    ledger.reserve("A", "r1", ["node1/RNGD/0", "node1/RNGD/1"], ver)
    _, free = ledger.snapshot()
    topo = available_topology(registry, free)
    assert [n["id"] for n in topo["nodes"]] == ["node0"]
    assert topo["links"] == []  # cross-node link pruned with node1


def test_pick_devices_deterministic(registry):
    free = registry.device_ids()
    picked = pick_devices(free, {("node0", "A40"): 2, ("node1", "RNGD"): 1})
    assert picked == ["node0/A40/0", "node0/A40/1", "node1/RNGD/0"]
    with pytest.raises(ValueError, match="not enough"):
        pick_devices(free, {("node1", "RNGD"): 3})
    assert free_counts(free)[("node0", "A40")] == 4
