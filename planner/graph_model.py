"""Cluster topology -> graph, plus unit parsing and cluster_config serialization.

The graph (``networkx.DiGraph``) is the input to the MILP solver's flow
constraints and the source for the ``link_bw`` / ``link_latency`` fields when
rendering a ``cluster_config``.
"""
from __future__ import annotations

import re

import networkx as nx

from .spec_schema import PlannerSpec
from .types import Device

# ---------------------------------------------------------------------------
# Unit parsing -> simulator units (bandwidth: GB/s, latency: ns)
# ---------------------------------------------------------------------------
_BW_RE = re.compile(r"^\s*([\d.]+)\s*([A-Za-z/]+)\s*$")
_LAT_RE = re.compile(r"^\s*([\d.]+)\s*([A-Za-z]*)\s*$")

# SI prefix -> bytes/s multiplier for a *byte* rate ("GBps"/"GB/s")
_BYTE_PREFIX = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}


def parse_bandwidth_gbps_bytes(text: str) -> float:
    """Parse a bandwidth string into GB/s (gigabytes/s), the simulator's link_bw unit.

    Convention: an uppercase 'B' means bytes ("600GBps", "600GB/s"); a lowercase
    'b' means bits ("200Gbps" -> 25 GB/s).
    """
    m = _BW_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse bandwidth: {text!r}")
    value, unit = float(m.group(1)), m.group(2)
    prefix = unit[0].lower()
    if prefix not in _BYTE_PREFIX:
        raise ValueError(f"unknown bandwidth prefix in {text!r}")
    bytes_per_s = value * _BYTE_PREFIX[prefix]
    if "B" not in unit:  # bit rate -> divide by 8
        bytes_per_s /= 8.0
    return bytes_per_s / 1e9  # -> GB/s


def parse_latency_ns(text: str) -> float:
    """Parse a latency string into nanoseconds (the simulator's link_latency unit)."""
    m = _LAT_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse latency: {text!r}")
    value, unit = float(m.group(1)), m.group(2).lower()
    scale = {"ns": 1.0, "us": 1e3, "": 1.0, "ms": 1e6, "s": 1e9}
    if unit not in scale:
        raise ValueError(f"unknown latency unit in {text!r}")
    return value * scale[unit]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(spec: PlannerSpec) -> nx.DiGraph:
    """Build a capacity graph: one node per (node_id) with device attributes,
    edges for inter-node links (bidirectional)."""
    g = nx.DiGraph()
    for node in spec.topology.nodes:
        devices = {d.name: {"count": d.count, "mem_gb": d.mem_gb} for d in node.devices}
        g.add_node(node.id, devices=devices)
    for link in spec.topology.links:
        bw = parse_bandwidth_gbps_bytes(link.bandwidth)
        lat = parse_latency_ns(link.latency)
        g.add_edge(link.src, link.dst, bw_gbps=bw, latency_ns=lat)
        g.add_edge(link.dst, link.src, bw_gbps=bw, latency_ns=lat)
    return g


def device_inventory(graph: nx.DiGraph) -> list[Device]:
    """Flattened device list for the solver."""
    inv: list[Device] = []
    for node_id, attrs in graph.nodes(data=True):
        for hw, d in attrs.get("devices", {}).items():
            inv.append(Device(node_id=node_id, hardware=hw, count=d["count"], mem_gb=d["mem_gb"]))
    return inv


def default_link_params(spec: PlannerSpec) -> tuple[float, float]:
    """Pick a representative (link_bw GB/s, link_latency ns) for the cluster_config.

    Uses the first inter-node link if present, else the intra-node bandwidth, else
    a benign default. Hierarchical fabrics use tp_group_shape + list link params
    (handled by the renderer); this is the scalar fallback.
    """
    if spec.topology.links:
        link = spec.topology.links[0]
        return parse_bandwidth_gbps_bytes(link.bandwidth), parse_latency_ns(link.latency)
    if spec.topology.intra_node_bandwidth:
        return parse_bandwidth_gbps_bytes(spec.topology.intra_node_bandwidth), 0.0
    return 112.0, 0.0  # matches the repo's example configs
