"""Cluster registry: the single source of truth for R1 topology (§4.4).

``cluster_registry.yaml`` describes physical nodes, devices, links, power and
oracle references; the ledger tracks which devices are reserved, and views
turn (registry − reservations) into a planner-spec topology.

.. code-block:: yaml

    nodes:
      - id: node0
        host_base_w: 250
        devices:
          - {name: A40, count: 8, mem_gb: 48}
      - id: node1
        devices:
          - {name: RNGD, count: 2, mem_gb: 48}
    links:
      - {src: node0, dst: node1, bandwidth: "200Gbps", latency: "0.0005ms"}
    intra_node_bandwidth: "600GBps"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class RegistryDevice(BaseModel):
    name: str
    count: int = Field(ge=1)
    mem_gb: float = Field(gt=0)


class RegistryNode(BaseModel):
    id: str
    devices: list[RegistryDevice]
    host_base_w: Optional[float] = Field(default=None, ge=0)


class RegistryLink(BaseModel):
    src: str
    dst: str
    bandwidth: str
    latency: str


class ClusterRegistry(BaseModel):
    nodes: list[RegistryNode]
    links: list[RegistryLink] = Field(default_factory=list)
    intra_node_bandwidth: Optional[str] = None
    tp_group_shape: Optional[list[int]] = None

    def device_ids(self) -> list[str]:
        """Stable individual device identities: ``<node>/<hw>/<index>``."""
        out = []
        for node in self.nodes:
            for dev in node.devices:
                out.extend(f"{node.id}/{dev.name}/{i}" for i in range(dev.count))
        return out

    def node(self, node_id: str) -> RegistryNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"unknown node '{node_id}'")


def load_registry(path: str | Path) -> ClusterRegistry:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ClusterRegistry.model_validate(data)
