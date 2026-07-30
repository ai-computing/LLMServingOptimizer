"""Affinity-aware placement refinement (plan §4.1 step 5, D1).

The planner decides hw x count; this module maps that onto concrete device
ids: each TP group goes to the free-device subset with the highest internal
affinity (min-cut bandwidth — NVLink sets beat PCIe-bridged sets), ties break
deterministically (lexicographic). Runs OUTSIDE CP-SAT as pure graph
post-processing (risk memo: no solver modification).
"""
from __future__ import annotations

from itertools import combinations

from .graph import TopologyGraph

#: enumeration cap per group; beyond this fall back to sorted-prefix picking
_MAX_COMBOS = 2000


def pick_devices_affinity(graph: TopologyGraph, free_device_ids: list[str],
                          groups: list[tuple[str, str, int]]) -> list[str]:
    """Choose concrete devices for TP groups.

    ``groups``: one (node_id, hw, group_size) per instance replica. Groups are
    placed largest-first (hardest constraints first); each takes the max-
    affinity combination of the still-free devices of its (node, hw). Raises
    ValueError on shortage.
    """
    pool: dict[tuple[str, str], list[str]] = {}
    for d in sorted(free_device_ids):
        node_id, hw, _ = d.rsplit("/", 2)
        pool.setdefault((node_id, hw), []).append(d)

    chosen: list[str] = []
    order = sorted(range(len(groups)), key=lambda i: -groups[i][2])
    picks: dict[int, list[str]] = {}
    for gi in order:
        node_id, hw, size = groups[gi]
        avail = pool.get((node_id, hw), [])
        if len(avail) < size:
            raise ValueError(f"not enough free devices for ({node_id}, {hw}): "
                             f"need {size}, have {len(avail)}")
        if size == 1:
            best = [avail[0]]
        else:
            n_combos = 1
            for k in range(size):
                n_combos = n_combos * (len(avail) - k) // (k + 1)
            if n_combos > _MAX_COMBOS:
                best = avail[:size]      # documented fallback, deterministic
            else:
                # combinations() yields lexicographic order and max() keeps the
                # first maximum -> ties resolve to the smallest ids (deterministic)
                best = sorted(max(combinations(avail, size),
                                  key=lambda c: graph.affinity(c)))
        picks[gi] = best
        for d in best:
            avail.remove(d)
    for gi in range(len(groups)):
        chosen.extend(picks[gi])
    return chosen
