"""D0 tests: registry v2 schema, v1 promotion, graph derivations (affinity,
free_subgraph), planner-view golden compatibility, and the graph endpoint."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from service.inventory.registry import ClusterRegistry
from service.inventory.views import available_topology as inv_topology
from service.topology.graph import TopologyGraph
from service.topology.schema import (
    RegistryV2,
    load_registry_any,
    parse_bandwidth_gbps,
    promote_v1,
)
from service.topology.views import available_topology as topo_topology

EXAMPLE_V1 = Path(__file__).resolve().parents[1] / "cluster_registry.example.yaml"


def _v2(nodes, inter=None):
    return RegistryV2.model_validate({"version": 2, "nodes": nodes,
                                      "inter_links": inter or []})


def _dev(node, hw, i, **kw):
    d = {"id": f"{node}/{hw}/{i}", "hw": hw, "mem_gb": 48, "idle_w": 25,
         "active_w": 300}
    d.update(kw)
    return d


# ---- schema ----------------------------------------------------------------

def test_parse_bandwidth_units():
    assert parse_bandwidth_gbps("112GBps") == 112.0
    assert parse_bandwidth_gbps("100Gbps") == 12.5
    assert parse_bandwidth_gbps("600GBps") == 600.0
    with pytest.raises(ValueError):
        parse_bandwidth_gbps("fast")


def test_v2_schema_violations_rejected():
    with pytest.raises(ValidationError, match="greater than 0"):   # negative power
        _v2([{"id": "n0", "devices": [_dev("n0", "A40", 0, active_w=-5)]}])
    with pytest.raises(ValidationError, match="duplicate"):
        _v2([{"id": "n0", "devices": [_dev("n0", "A40", 0), _dev("n0", "A40", 0)]}])
    with pytest.raises(ValidationError, match="endpoint"):
        _v2([{"id": "n0", "devices": [_dev("n0", "A40", 0)],
              "intra_links": [{"a": "n0/A40/0", "b": "n0/A40/9",
                               "kind": "nvlink", "bandwidth": "112GBps"}]}])
    with pytest.raises(ValidationError, match="must start with"):
        _v2([{"id": "n0", "devices": [_dev("nX", "A40", 0)]}])


# ---- v1 promotion -----------------------------------------------------------

def test_v1_example_promotes_to_expected_graph():
    reg = load_registry_any(EXAMPLE_V1)
    assert isinstance(reg, RegistryV2)
    # v1 example: node0 A40x8, node1 A5000x2, 1 link, intra 600GBps
    assert len(reg.device_ids()) == 10
    assert "node0/A40/7" in reg.device_ids() and "node1/A5000/1" in reg.device_ids()
    g = TopologyGraph(reg)
    node0_devs = g.devices_of_host("node0")
    assert len(node0_devs) == 8
    # full-mesh pcie among 8 devices = C(8,2) = 28 intra edges on node0
    intra0 = [e for e in g.g.edges(data=True)
              if not e[2]["inter"] and e[0].startswith("node0/A40")
              and e[1].startswith("node0/A40")]
    assert len(intra0) == 28
    # one nic-to-nic inter link with the original bandwidth string
    inter = [e for e in g.g.edges(data=True) if e[2]["inter"]]
    assert len(inter) == 1 and inter[0][2]["bandwidth"] == "100Gbps"
    # promoted power comes from the shipped power profiles
    assert g.g.nodes["node0/A40/0"]["active_w"] == 300
    assert g.g.nodes["node0/A40/0"]["idle_w"] == 25


# ---- graph derivations --------------------------------------------------------

@pytest.fixture()
def hetero_links_graph() -> TopologyGraph:
    """One host, 3 groups of 4 A40s with different interconnects:
    clique: NVLink full mesh; split: two NVLink pairs + PCIe mesh between all;
    weak: PCIe mesh only. Same PCIe bandwidth everywhere."""
    devs = [_dev("n0", "A40", i) for i in range(12)]
    ids = [d["id"] for d in devs]
    links = []

    def mesh(group, kind, bw):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                links.append({"a": group[i], "b": group[j], "kind": kind,
                              "bandwidth": bw})
    mesh(ids[0:4], "nvlink", "112GBps")               # clique
    links += [{"a": ids[4], "b": ids[5], "kind": "nvlink", "bandwidth": "112GBps"},
              {"a": ids[6], "b": ids[7], "kind": "nvlink", "bandwidth": "112GBps"}]
    mesh(ids[4:8], "pcie", "32GBps")                  # split: pairs + pcie mesh
    mesh(ids[8:12], "pcie", "32GBps")                 # weak: pcie only
    return TopologyGraph(_v2([{"id": "n0", "devices": devs,
                               "intra_links": links}]))


def test_affinity_ordering_hand_computed(hetero_links_graph):
    g = hetero_links_graph
    ids = [f"n0/A40/{i}" for i in range(12)]
    clique = g.affinity(ids[0:4])   # isolate one: 3x112 = 336
    split = g.affinity(ids[4:8])    # pair-vs-pair: 4x32 = 128
    weak = g.affinity(ids[8:12])    # isolate one: 3x32 = 96
    assert clique == pytest.approx(336.0)
    assert split == pytest.approx(128.0)
    assert weak == pytest.approx(96.0)
    assert clique > split > weak    # the plan's required ordering
    assert g.affinity(ids[:1]) == math.inf
    assert g.affinity([ids[0], ids[8]]) == 0.0   # disconnected pair


def test_free_subgraph_excludes_reserved(hetero_links_graph):
    g = hetero_links_graph
    sub = g.free_subgraph({"n0/A40/0", "n0/A40/1"})
    assert "n0/A40/0" not in sub and "n0/A40/2" in sub
    assert not any(u in ("n0/A40/0", "n0/A40/1") or v in ("n0/A40/0", "n0/A40/1")
                   for u, v in sub.edges())


# ---- planner-view golden compatibility ----------------------------------------

def test_planner_topology_golden_vs_inventory_views():
    """For a v1 registry, topology.views output must equal inventory.views
    output for every reservation pattern tested."""
    v1_data = yaml.safe_load(EXAMPLE_V1.read_text())
    inv_reg = ClusterRegistry.model_validate(v1_data)
    v2_reg = promote_v1(v1_data)
    all_ids = inv_reg.device_ids()
    patterns = [
        all_ids,                                     # nothing reserved
        [d for d in all_ids if d != "node0/A40/0"],  # one reserved
        [d for d in all_ids if not d.startswith("node1/")],  # node1 fully out
        [],                                          # everything reserved
    ]
    for free in patterns:
        assert topo_topology(v2_reg, free) == inv_topology(inv_reg, free)


# ---- API endpoint ---------------------------------------------------------------

def test_cluster_graph_endpoint(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from service.api.routes import ServiceState, create_service_router
    from service.inventory.ledger import Ledger

    reg = ClusterRegistry.model_validate(yaml.safe_load(EXAMPLE_V1.read_text()))
    ledger = Ledger(tmp_path / "l.sqlite", reg)
    ver, _ = ledger.snapshot()
    ledger.reserve("t", "r1", ["node0/A40/0"], ver)

    app = FastAPI()
    app.include_router(create_service_router(ServiceState(registry=reg, ledger=ledger)))
    d = TestClient(app).get("/api/cluster/graph").json()
    assert {h["id"] for h in d["hosts"]} == {"node0", "node1"}
    states = {v["id"]: v.get("state") for v in d["vertices"] if v["type"] == "device"}
    assert states["node0/A40/0"] == "reserved"
    assert states["node0/A40/1"] == "free"
    assert any(e["inter"] for e in d["edges"])


# ---- native v2 registry through the service layer ------------------------------

_V2_DOC = {
    "version": 2,
    "nodes": [
        {"id": "node0", "host_base_w": 200,
         "docker": {"endpoint": "unix://var/run/docker.sock", "tls": False},
         "devices": [{"id": f"node0/A5000/{i}", "hw": "A5000", "mem_gb": 24,
                      "idle_w": 60, "active_w": 230} for i in range(2)],
         "switches": [{"id": "node0/nic0", "kind": "nic"}],
         "intra_links": [{"a": "node0/A5000/0", "b": "node0/A5000/1",
                          "kind": "pcie", "bandwidth": "12GBps"}]},
        {"id": "a40-0", "host_base_w": 250,
         "docker": {"endpoint": "tcp://gpu-a40-0:2376", "tls": True},
         "devices": [{"id": f"a40-0/A40/{i}", "hw": "A40", "mem_gb": 48,
                      "idle_w": 25, "active_w": 300, "numa": 0 if i < 4 else 1}
                     for i in range(8)],
         "switches": [{"id": "a40-0/nic0", "kind": "nic"}],
         "intra_links": [{"a": "a40-0/A40/0", "b": "a40-0/A40/1",
                          "kind": "nvlink", "bandwidth": "112GBps"},
                         {"a": "a40-0/A40/0", "b": "a40-0/A40/2",
                          "kind": "pcie", "bandwidth": "12GBps"}]},
    ],
    "inter_links": [{"a": "node0/nic0", "b": "a40-0/nic0", "kind": "infiniband",
                     "bandwidth": "200Gbps", "latency": "0.0015ms",
                     "rdma": True}],
}


def test_service_state_accepts_native_v2_registry(tmp_path):
    """ServiceState must take a RegistryV2 as-is (no v1 promotion) and expose
    per-node dockerd endpoints so remote deployments never hit the local
    daemon."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from service.api.routes import ServiceState, create_service_router
    from service.inventory.ledger import Ledger

    reg = RegistryV2.model_validate(_V2_DOC)
    ledger = Ledger(tmp_path / "l.sqlite", reg)          # v2 device_ids()
    state = ServiceState(registry=reg, ledger=ledger)
    assert state.topology_graph().registry is reg        # used as-is
    assert state.docker_endpoints() == {
        "node0": {"endpoint": "unix://var/run/docker.sock", "tls": False},
        "a40-0": {"endpoint": "tcp://gpu-a40-0:2376", "tls": True}}

    app = FastAPI()
    app.include_router(create_service_router(state))
    client = TestClient(app)
    assert len(client.get("/api/cluster").json()["free_devices"]) == 10
    graph = client.get("/api/cluster/graph").json()
    assert {h["id"] for h in graph["hosts"]} == {"node0", "a40-0"}
    assert any(e["kind"] == "infiniband" for e in graph["edges"])
    # /api/models must not choke on the v2 device shape (hw, not name)
    models = client.get("/api/models").json()["models"]
    assert any("A40" in per_hw or "A5000" in per_hw for per_hw in models.values())


def test_v2_planner_topology_counts_and_ib_links(tmp_path):
    """Planner view of a v2 registry: per-(node, hw) free counts + inter-node
    IB links preserved with their bandwidth/latency strings."""
    reg = RegistryV2.model_validate(_V2_DOC)
    topo = topo_topology(reg, reg.device_ids())
    by_node = {n["id"]: n for n in topo["nodes"]}
    assert by_node["node0"]["devices"] == [
        {"name": "A5000", "count": 2, "mem_gb": 24}]
    assert by_node["a40-0"]["devices"] == [
        {"name": "A40", "count": 8, "mem_gb": 48}]
    assert by_node["a40-0"]["host_base_w"] == 250
    assert topo["links"] == [{"src": "node0", "dst": "a40-0",
                              "bandwidth": "200Gbps", "latency": "0.0015ms"}]
