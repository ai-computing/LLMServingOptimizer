"""TopologyGraph: networkx MultiGraph over registry v2 with the derived
operations the execution layer needs (plan §3.1).

* ``free_subgraph``   — induced subgraph excluding reserved/faulted devices
* ``affinity``        — Stoer–Wagner min-cut bandwidth (GB/s) inside a device
                        set: NVLink cliques score above NVLink pairs bridged by
                        PCIe, which score above PCIe-only meshes
* ``to_ui``           — vertex/edge lists for GET /api/cluster/graph
Planner-view serialization lives in :mod:`service.topology.views`.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import networkx as nx

from .schema import RegistryV2, parse_bandwidth_gbps


class TopologyGraph:
    def __init__(self, registry: RegistryV2):
        self.registry = registry
        g = nx.MultiGraph()
        for node in registry.nodes:
            for d in node.devices:
                g.add_node(d.id, type="device", host=node.id, kind=d.kind,
                           hw=d.hw, mem_gb=d.mem_gb, idle_w=d.idle_w,
                           active_w=d.active_w, numa=d.numa)
            for s in node.switches:
                g.add_node(s.id, type="switch", host=node.id, kind=s.kind)
            for l in node.intra_links:
                g.add_edge(l.a, l.b, kind=l.kind, bandwidth=l.bandwidth,
                           gbps=parse_bandwidth_gbps(l.bandwidth), inter=False)
        for l in registry.inter_links:
            g.add_edge(l.a, l.b, kind=l.kind, bandwidth=l.bandwidth,
                       gbps=parse_bandwidth_gbps(l.bandwidth), inter=True,
                       rdma=l.rdma)
        self.g = g

    # -- derivations ----------------------------------------------------------
    def free_subgraph(self, reserved_device_ids: Iterable[str]) -> nx.MultiGraph:
        """Induced subgraph with reserved device vertices (and their incident
        edges) removed; switches stay."""
        reserved = set(reserved_device_ids)
        keep = [n for n, d in self.g.nodes(data=True)
                if not (d["type"] == "device" and n in reserved)]
        return self.g.subgraph(keep).copy()

    def _collapsed(self, vertices: set[str]) -> nx.Graph:
        """Simple graph on the given vertices; parallel link capacities sum."""
        sg = nx.Graph()
        sg.add_nodes_from(vertices)
        for u, v, d in self.g.subgraph(vertices).edges(data=True):
            if sg.has_edge(u, v):
                sg[u][v]["gbps"] += d["gbps"]
            else:
                sg.add_edge(u, v, gbps=d["gbps"])
        return sg

    def affinity(self, device_ids: Iterable[str]) -> float:
        """Min-cut bandwidth (GB/s) of the device set, switches included when
        they bridge two or more members. Higher = tighter-coupled placement.
        Singleton sets score +inf; disconnected sets score 0."""
        devs = set(device_ids)
        if len(devs) <= 1:
            return math.inf
        # include switches adjacent to >= 2 set members (they carry internal paths)
        switches = set()
        for s, d in self.g.nodes(data=True):
            if d["type"] == "switch":
                nbrs = set(self.g.neighbors(s))
                if len(nbrs & devs) >= 2:
                    switches.add(s)
        sg = self._collapsed(devs | switches)
        if not nx.is_connected(sg):
            return 0.0
        if sg.number_of_nodes() == 1:
            return math.inf
        cut, _ = nx.stoer_wagner(sg, weight="gbps")
        return float(cut)

    def devices_of_host(self, node_id: str) -> list[str]:
        return [n for n, d in self.g.nodes(data=True)
                if d["type"] == "device" and d["host"] == node_id]

    # -- UI serialization ------------------------------------------------------
    def to_ui(self, device_states: Optional[dict[str, str]] = None) -> dict:
        """{hosts, vertices, edges} for the cluster-view UI. ``device_states``
        maps device id -> free|reserved|running|faulted (default free)."""
        states = device_states or {}
        hosts = [{"id": n.id, "hostname": n.hostname,
                  "host_base_w": n.host_base_w} for n in self.registry.nodes]
        vertices = []
        for vid, d in self.g.nodes(data=True):
            v = {"id": vid, "type": d["type"], "host": d["host"],
                 "kind": d.get("kind")}
            if d["type"] == "device":
                v.update(hw=d["hw"], mem_gb=d["mem_gb"], idle_w=d["idle_w"],
                         active_w=d["active_w"],
                         state=states.get(vid, "free"))
            vertices.append(v)
        edges = [{"a": u, "b": v, "kind": d["kind"], "bandwidth": d["bandwidth"],
                  "gbps": round(d["gbps"], 3), "inter": d["inter"]}
                 for u, v, d in self.g.edges(data=True)]
        return {"hosts": hosts, "vertices": vertices, "edges": edges}
