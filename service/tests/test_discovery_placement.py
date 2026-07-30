"""D1 tests: nvidia-smi topo parser -> registry draft, and affinity-refined
TP-group placement (NVLink pairs preferred, groups split across pairs)."""
from __future__ import annotations

import pytest

from service.topology.discovery import draft_registry, parse_topo_matrix
from service.topology.graph import TopologyGraph
from service.topology.placement import pick_devices_affinity
from service.topology.schema import RegistryV2

# fixture mimicking `nvidia-smi topo -m` on a 4x GPU box with two NVLink
# pairs (0-1, 2-3), PCIe-switch between 0-2/1-3, host bridge for the rest
TOPO_4GPU = """\
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV4     PIX     PHB     0-15            0
GPU1    NV4      X      PHB     PIX     0-15            0
GPU2    PIX     PHB      X      NV4     16-31           1
GPU3    PHB     PIX     NV4      X      16-31           1

Legend:
  X    = Self
  NV#  = Connection traversing a bonded set of # NVLinks
"""


def test_topo_matrix_parser():
    m = parse_topo_matrix(TOPO_4GPU)
    assert m[(0, 1)] == "NV4" and m[(2, 3)] == "NV4"
    assert m[(0, 2)] == "PIX" and m[(0, 3)] == "PHB"
    assert m[(1, 2)] == "PHB" and m[(1, 3)] == "PIX"
    assert len(m) == 6  # upper triangle only, GPU-GPU cells only


def test_draft_registry_link_kinds_and_validity():
    doc = draft_registry(TOPO_4GPU, node_id="node0", hw="A40", mem_gb=48)
    reg = RegistryV2.model_validate(doc)   # draft must be a valid v2 registry
    assert len(reg.device_ids()) == 4
    links = {(l.a, l.b): (l.kind, l.bandwidth) for l in reg.nodes[0].intra_links}
    assert links[("node0/A40/0", "node0/A40/1")] == ("nvlink", "112GBps")  # NV4
    assert links[("node0/A40/2", "node0/A40/3")] == ("nvlink", "112GBps")
    assert links[("node0/A40/0", "node0/A40/2")] == ("pcie", "32GBps")     # PIX
    assert links[("node0/A40/0", "node0/A40/3")] == ("pcie", "16GBps")     # PHB
    # power fields come from the shipped profiles
    assert reg.nodes[0].devices[0].active_w == 300


@pytest.fixture()
def graph_4gpu() -> TopologyGraph:
    return TopologyGraph(RegistryV2.model_validate(
        draft_registry(TOPO_4GPU, node_id="node0", hw="A40", mem_gb=48)))


def test_tp2_prefers_nvlink_pair(graph_4gpu):
    """One TP2 group with all 4 devices free: must land on an NVLink pair,
    never a PCIe-connected pair."""
    picked = pick_devices_affinity(
        graph_4gpu, [f"node0/A40/{i}" for i in range(4)],
        groups=[("node0", "A40", 2)])
    assert picked in (["node0/A40/0", "node0/A40/1"],
                      ["node0/A40/2", "node0/A40/3"])


def test_two_tp2_groups_split_across_nvlink_pairs(graph_4gpu):
    picked = pick_devices_affinity(
        graph_4gpu, [f"node0/A40/{i}" for i in range(4)],
        groups=[("node0", "A40", 2), ("node0", "A40", 2)])
    g1, g2 = set(picked[:2]), set(picked[2:])
    assert {frozenset(g1), frozenset(g2)} == {
        frozenset({"node0/A40/0", "node0/A40/1"}),
        frozenset({"node0/A40/2", "node0/A40/3"})}


def test_nvlink_pair_avoided_when_partially_reserved(graph_4gpu):
    """GPU1 reserved: the surviving NVLink pair (2,3) must be chosen over the
    PCIe combination (0,2)/(0,3)."""
    free = ["node0/A40/0", "node0/A40/2", "node0/A40/3"]
    picked = pick_devices_affinity(graph_4gpu, free, [("node0", "A40", 2)])
    assert picked == ["node0/A40/2", "node0/A40/3"]


def test_shortage_raises(graph_4gpu):
    with pytest.raises(ValueError, match="not enough"):
        pick_devices_affinity(graph_4gpu, ["node0/A40/0"], [("node0", "A40", 2)])


def test_size1_groups_deterministic(graph_4gpu):
    picked = pick_devices_affinity(
        graph_4gpu, [f"node0/A40/{i}" for i in range(4)],
        groups=[("node0", "A40", 1), ("node0", "A40", 1)])
    assert picked == ["node0/A40/0", "node0/A40/1"]
