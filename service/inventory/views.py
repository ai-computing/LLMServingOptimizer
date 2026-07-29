"""Available-subtopology views: (registry − reservations) → planner topology.

The planner spec's ``topology`` dict is produced with node/link structure
preserved and per-(node, hardware) counts reduced to what is currently free;
empty device groups and empty nodes are dropped (planner validation requires
count >= 1), and links between dropped nodes are pruned.
"""
from __future__ import annotations

from collections import Counter

from .registry import ClusterRegistry


def _parse_device_id(device_id: str) -> tuple[str, str]:
    node_id, hw, _idx = device_id.rsplit("/", 2)
    return node_id, hw


def free_counts(free_device_ids: list[str]) -> dict[tuple[str, str], int]:
    """{(node_id, hardware): free count}"""
    return Counter(_parse_device_id(d) for d in free_device_ids)


def available_topology(registry: ClusterRegistry,
                       free_device_ids: list[str]) -> dict:
    """Planner-spec ``topology`` dict for the currently free devices."""
    counts = free_counts(free_device_ids)
    nodes = []
    for node in registry.nodes:
        devices = []
        for dev in node.devices:
            free = counts.get((node.id, dev.name), 0)
            if free > 0:
                devices.append({"name": dev.name, "count": free,
                                "mem_gb": dev.mem_gb})
        if devices:
            entry: dict = {"id": node.id, "devices": devices}
            if node.host_base_w is not None:
                entry["host_base_w"] = node.host_base_w
            nodes.append(entry)

    kept = {n["id"] for n in nodes}
    links = [{"src": l.src, "dst": l.dst, "bandwidth": l.bandwidth,
              "latency": l.latency}
             for l in registry.links if l.src in kept and l.dst in kept]

    topo: dict = {"nodes": nodes, "links": links}
    if registry.intra_node_bandwidth:
        topo["intra_node_bandwidth"] = registry.intra_node_bandwidth
    if registry.tp_group_shape:
        topo["tp_group_shape"] = registry.tp_group_shape
    return topo


def pick_devices(free_device_ids: list[str],
                 need: dict[tuple[str, str], int]) -> list[str]:
    """Choose concrete device ids satisfying {(node, hw): count} from the free
    pool (lowest index first, deterministic). Raises ValueError on shortage."""
    by_group: dict[tuple[str, str], list[str]] = {}
    for d in sorted(free_device_ids):
        by_group.setdefault(_parse_device_id(d), []).append(d)
    chosen: list[str] = []
    for key, cnt in need.items():
        pool = by_group.get(key, [])
        if len(pool) < cnt:
            raise ValueError(f"not enough free devices for {key}: "
                             f"need {cnt}, have {len(pool)}")
        chosen.extend(pool[:cnt])
    return chosen
