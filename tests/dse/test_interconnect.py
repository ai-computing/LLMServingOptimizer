"""Unit tests for webapp.dse.core.interconnect — fabric / catalog resolution."""
from webapp.cluster_builder import ConfigSpec, InstanceSpec, build_cluster_json
from webapp.dse.core.interconnect import (
    _decompose_tp,
    load_fabrics,
    resolve_interconnect,
)

A40_TIERS = [
    {"name": "nvlink", "size": 2, "bw_gbs": 52.8, "latency_ns": 0},
    {"name": "pcie", "size": 2, "bw_gbs": 24.5, "latency_ns": 0},
    {"name": "qpi", "size": 2, "bw_gbs": 21.0, "latency_ns": 0},
]
HW_META = {
    "A40": {"interconnect_bw_gbs": 52.8},
    "A6000": {"interconnect_bw_gbs": 32},
    "H100": {"interconnect_bw_gbs": 900},
}


def _spec(hw, tp, pp=1, dp=1):
    inst = InstanceSpec(hardware=hw, model="m", npu_num=tp * pp,
                        npu_group=pp, pd_type=None)
    return ConfigSpec(label=f"{hw}_tp{tp}", instances=[inst],
                      tp=tp, pp=pp, dp=dp, pd_layout="—")


# ---- _decompose_tp ---------------------------------------------------------

def test_decompose_powers_of_two():
    assert _decompose_tp(2, A40_TIERS) == ([2], [52.8], [0.0])
    assert _decompose_tp(4, A40_TIERS) == ([2, 2], [52.8, 24.5], [0.0, 0.0])
    assert _decompose_tp(8, A40_TIERS) == ([2, 2, 2], [52.8, 24.5, 21.0], [0.0, 0.0, 0.0])


def test_decompose_tp1_is_empty():
    assert _decompose_tp(1, A40_TIERS) == ([], [], [])


def test_decompose_non_factorable_returns_none():
    # TP3 cannot be placed on size-2 tiers.
    assert _decompose_tp(3, A40_TIERS) is None


def test_decompose_tier_larger_than_remaining():
    # Single 8-wide tier; TP4 consumes only the divisible part.
    tiers = [{"size": 8, "bw_gbs": 600.0, "latency_ns": 0}]
    assert _decompose_tp(4, tiers) == ([4], [600.0], [0.0])


# ---- resolve_interconnect: fabric mode ------------------------------------

def test_fabric_tp8_full_hierarchy():
    ic = resolve_interconnect(_spec("A40", 8), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] == [2, 2, 2]
    assert ic["link_bw"] == [52.8, 24.5, 21.0]


def test_fabric_tp4_partial_hierarchy():
    ic = resolve_interconnect(_spec("A40", 4), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] == [2, 2]
    assert ic["link_bw"] == [52.8, 24.5]


def test_fabric_tp1_scalar_slowest_tier():
    # No intra-group collective → slowest (outermost) tier as scalar, no shape.
    ic = resolve_interconnect(_spec("A40", 1), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] is None
    assert ic["link_bw"] == 21.0


def test_fabric_non_factorable_falls_back_to_slowest_scalar():
    ic = resolve_interconnect(_spec("A40", 3), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] is None
    assert ic["link_bw"] == 21.0  # slowest tier


# ---- resolve_interconnect: catalog scalar fallback ------------------------

def test_catalog_fallback_homogeneous():
    ic = resolve_interconnect(_spec("H100", 2), None, HW_META)
    assert ic == {"link_bw": 900.0, "link_latency": 0, "tp_group_shape": None}


def test_catalog_fallback_heterogeneous_uses_min():
    het = ConfigSpec(
        label="het",
        instances=[InstanceSpec("H100", "m", 2, 1, None),
                   InstanceSpec("A6000", "m", 2, 1, None)],
        tp=2, pp=1, dp=2, pd_layout="—",
    )
    ic = resolve_interconnect(het, None, HW_META)
    assert ic["link_bw"] == 32.0  # min(900, 32)


def test_catalog_fallback_unknown_hw_returns_empty():
    ic = resolve_interconnect(_spec("UNKNOWN", 2), None, {})
    assert ic == {}


# ---- end-to-end with build_cluster_json -----------------------------------

def test_build_cluster_json_embeds_tp_group_shape():
    ic = resolve_interconnect(_spec("A40", 8), {"tiers": A40_TIERS}, HW_META)
    cj = build_cluster_json(
        _spec("A40", 8),
        {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        ic["link_bw"], ic["link_latency"],
        tp_group_shape=ic["tp_group_shape"],
    )
    assert cj["tp_group_shape"] == [2, 2, 2]
    assert cj["link_bw"] == [52.8, 24.5, 21.0]


def test_build_cluster_json_no_shape_when_scalar():
    cj = build_cluster_json(
        _spec("H100", 2),
        {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        900.0, 0, tp_group_shape=None,
    )
    assert "tp_group_shape" not in cj
    assert cj["link_bw"] == 900.0


# ---- collective_overhead pass-through -------------------------------------

_COHD = {"enabled": True, "socket_size": 4, "floor_ns": 70000, "per_token_ns": 10000}


def test_fabric_passes_collective_overhead_all_tps():
    fabric = {"tiers": A40_TIERS, "collective_overhead": _COHD}
    for tp in (1, 2, 4, 8):
        ic = resolve_interconnect(_spec("A40", tp), fabric, HW_META)
        assert ic["collective_overhead"] == _COHD


def test_fabric_without_overhead_returns_none():
    ic = resolve_interconnect(_spec("A40", 8), {"tiers": A40_TIERS}, HW_META)
    assert ic.get("collective_overhead") is None


def test_catalog_fallback_has_no_overhead():
    ic = resolve_interconnect(_spec("H100", 2), None, HW_META)
    assert "collective_overhead" not in ic


def test_build_cluster_json_embeds_collective_overhead():
    cj = build_cluster_json(
        _spec("A40", 8), {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        [52.8, 24.5, 21.0], [0, 0, 0],
        tp_group_shape=[2, 2, 2], collective_overhead=_COHD,
    )
    assert cj["collective_overhead"] == _COHD


def test_build_cluster_json_no_overhead_key_when_none():
    cj = build_cluster_json(
        _spec("H100", 2), {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        900.0, 0, collective_overhead=None,
    )
    assert "collective_overhead" not in cj


# ---- fabrics.yaml presets load --------------------------------------------

def test_load_fabrics_has_a40_preset():
    load_fabrics.cache_clear()
    fabs = load_fabrics()
    assert "2socket_8npu_nvlink_bridge_per_2slot" in fabs
    tiers = fabs["2socket_8npu_nvlink_bridge_per_2slot"]["tiers"]
    # 4 tiers: nvlink/pcie/qpi intra-box + cross-node 200G IB (8 GPU/node, up to 2 nodes).
    assert [t["size"] for t in tiers] == [2, 2, 2, 2]
    ib = tiers[-1]
    assert ib["name"] == "ib"
    assert ib["bw_gbs"] == 25.0  # 200 Gbps unidirectional


def test_a40_preset_carries_calibrated_overhead():
    load_fabrics.cache_clear()
    co = load_fabrics()["2socket_8npu_nvlink_bridge_per_2slot"].get("collective_overhead")
    assert co and co["enabled"] and co["socket_size"] == 4
    # cross-node IB collective overhead gates on the node boundary (TP > 8).
    assert co["node_size"] == 8
    assert co["node_floor_ns"] == 105000
    assert co["node_per_token_ns"] == 18000


def test_a40_preset_tp16_crosses_node_on_ib():
    """TP16 on the A40 box spans 2 nodes: outermost TP dim must use the 25 GB/s
    (200 Gbps) IB link, matching cluster_config/a40_16gpu_tp16_70b_4tier.json."""
    load_fabrics.cache_clear()
    fabric = load_fabrics()["2socket_8npu_nvlink_bridge_per_2slot"]
    ic = resolve_interconnect(_spec("A40", 16), fabric, HW_META)
    assert ic["tp_group_shape"] == [2, 2, 2, 2]
    assert ic["link_bw"] == [52.8, 24.5, 21.0, 25.0]
    assert ic["link_bw"][-1] == 25.0  # cross-node 200G IB is the outermost dim


def test_a40_preset_tp8_stays_intra_node():
    """TP8 fits one box: must NOT consume the IB tier (no 25 GB/s dim)."""
    load_fabrics.cache_clear()
    fabric = load_fabrics()["2socket_8npu_nvlink_bridge_per_2slot"]
    ic = resolve_interconnect(_spec("A40", 8), fabric, HW_META)
    assert ic["tp_group_shape"] == [2, 2, 2]
    assert ic["link_bw"] == [52.8, 24.5, 21.0]
    assert 25.0 not in ic["link_bw"]


def test_pcie_only_preset_no_nvlink_tier():
    """PCIe-only 2-socket fabric (for NVLink-less NPUs like RNGD): 4 NPU/socket on
    PCIe, cross-socket QPI, cross-node IB — no NVLink tier."""
    load_fabrics.cache_clear()
    fabric = load_fabrics()["2socket_8npu_pcie_only"]
    assert [t["name"] for t in fabric["tiers"]] == ["pcie", "qpi", "ib"]
    assert [t["size"] for t in fabric["tiers"]] == [4, 2, 2]
    # TP4 stays within one socket (flat PCIe), no cross-socket/cross-node dim.
    assert resolve_interconnect(_spec("RNGD", 4), fabric, HW_META)["link_bw"] == [32.0]
    # TP8 crosses the socket (QPI) but not the node.
    ic8 = resolve_interconnect(_spec("RNGD", 8), fabric, HW_META)
    assert ic8["tp_group_shape"] == [4, 2]
    assert ic8["link_bw"] == [32.0, 21.0]
    # TP16 crosses the node on 200G IB.
    ic16 = resolve_interconnect(_spec("RNGD", 16), fabric, HW_META)
    assert ic16["tp_group_shape"] == [4, 2, 2]
    assert ic16["link_bw"] == [32.0, 21.0, 25.0]
