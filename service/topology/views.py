"""Registry-v2 → planner-spec topology serialization (plan D0).

MUST stay byte-compatible with ``service/inventory/views.available_topology``
for v1-promoted registries (guarded by a golden test): same node/device
grouping, same link dicts, same passthrough keys.
"""
from __future__ import annotations

from collections import Counter

from .schema import RegistryV2


def _node_hw(device_id: str) -> tuple[str, str]:
    node_id, hw, _idx = device_id.rsplit("/", 2)
    return node_id, hw


def available_topology(registry: RegistryV2,
                       free_device_ids: list[str]) -> dict:
    counts = Counter(_node_hw(d) for d in free_device_ids)
    nodes = []
    for node in registry.nodes:
        groups: dict[str, dict] = {}
        for dev in node.devices:  # preserve group order of first appearance
            g = groups.setdefault(dev.hw, {"name": dev.hw, "count": 0,
                                           "mem_gb": dev.mem_gb})
        for (nid, hw), free in counts.items():
            if nid == node.id and hw in groups:
                groups[hw]["count"] = free
        devices = [g for g in groups.values() if g["count"] > 0]
        if devices:
            entry: dict = {"id": node.id, "devices": devices}
            if node.host_base_w is not None:
                entry["host_base_w"] = node.host_base_w
            nodes.append(entry)

    kept = {n["id"] for n in nodes}
    links = []
    for l in registry.inter_links:
        src, dst = registry.node_of(l.a), registry.node_of(l.b)
        if src in kept and dst in kept:
            links.append({"src": src, "dst": dst, "bandwidth": l.bandwidth,
                          "latency": l.latency if l.latency is not None else "0ms"})

    topo: dict = {"nodes": nodes, "links": links}
    if registry.intra_node_bandwidth:
        topo["intra_node_bandwidth"] = registry.intra_node_bandwidth
    if registry.tp_group_shape:
        topo["tp_group_shape"] = registry.tp_group_shape
    return topo
