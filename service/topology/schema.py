"""Registry v2 schema + v1 auto-promotion (execution-layer plan §3.1).

v2 models nodes down to individual devices, switches, and typed links.
The v1 format (``service/cluster_registry.example.yaml``) keeps working:
``load_registry_any`` promotes it — ``count`` expands to per-device vertices
(ids ``<node>/<hw>/<i>``, identical to the ledger convention), a per-node
``intra_node_bandwidth`` becomes a full-mesh of PCIe IntraLinks, node-level
``links`` become NIC-to-NIC InterLinks.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

_BW_RE = re.compile(r"^\s*([\d.]+)\s*([GMK]?)(B|b)ps\s*$", re.IGNORECASE)


def parse_bandwidth_gbps(s: str) -> float:
    """'112GBps' -> 112.0 GB/s; '100Gbps' -> 12.5 GB/s (bits vs bytes)."""
    m = _BW_RE.match(s)
    if not m:
        raise ValueError(f"unparseable bandwidth {s!r}")
    val, prefix, unit = float(m.group(1)), m.group(2).upper(), m.group(3)
    scale = {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1.0}[prefix]
    gb = val * scale
    return gb / 8.0 if unit == "b" else gb


class DockerCfg(BaseModel):
    endpoint: str                      # tcp://host:2376 | ssh://user@host | unix://...
    tls: bool = False


class DeviceV2(BaseModel):
    id: str                            # <node>/<hw>/<idx> — ledger convention
    kind: Literal["gpu", "npu"] = "gpu"
    hw: str
    mem_gb: float = Field(gt=0)
    idle_w: float = Field(ge=0, default=0.0)
    active_w: Optional[float] = Field(default=None, gt=0)
    pcie: Optional[str] = None         # e.g. gen4x16
    numa: Optional[int] = None
    runtime: Optional[str] = None      # npu: furiosa ...
    device_path: Optional[str] = None


class SwitchV2(BaseModel):
    id: str
    kind: Literal["nvswitch", "pcie_switch", "nic", "tor_switch"]
    bandwidth: Optional[str] = None


class IntraLinkV2(BaseModel):
    a: str
    b: str
    kind: Literal["nvlink", "xgmi", "pcie", "npu_link"]
    bandwidth: str
    latency: Optional[str] = None


class InterLinkV2(BaseModel):
    a: str
    b: str
    kind: Literal["ethernet", "infiniband", "cxl"]
    bandwidth: str
    latency: Optional[str] = None
    rdma: bool = False


class NodeV2(BaseModel):
    id: str
    hostname: Optional[str] = None
    host_base_w: Optional[float] = Field(default=None, ge=0)
    docker: Optional[DockerCfg] = None
    devices: list[DeviceV2]
    switches: list[SwitchV2] = Field(default_factory=list)
    intra_links: list[IntraLinkV2] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        ids = {d.id for d in self.devices} | {s.id for s in self.switches}
        if len(ids) != len(self.devices) + len(self.switches):
            raise ValueError(f"node {self.id}: duplicate device/switch ids")
        for d in self.devices:
            if not d.id.startswith(self.id + "/"):
                raise ValueError(f"device id {d.id!r} must start with '{self.id}/'")
        for l in self.intra_links:
            for end in (l.a, l.b):
                if end not in ids:
                    raise ValueError(f"node {self.id}: intra_link endpoint {end!r} unknown")
            parse_bandwidth_gbps(l.bandwidth)  # fail early on bad units
        return self


class RegistryV2(BaseModel):
    version: Literal[2] = 2
    nodes: list[NodeV2]
    inter_links: list[InterLinkV2] = Field(default_factory=list)
    # planner passthrough (kept from v1)
    intra_node_bandwidth: Optional[str] = None
    tp_group_shape: Optional[list[int]] = None

    @model_validator(mode="after")
    def _check(self):
        all_ids: set[str] = set()
        for n in self.nodes:
            ids = {d.id for d in n.devices} | {s.id for s in n.switches}
            dup = all_ids & ids
            if dup:
                raise ValueError(f"duplicate ids across nodes: {sorted(dup)}")
            all_ids |= ids
        for l in self.inter_links:
            for end in (l.a, l.b):
                if end not in all_ids:
                    raise ValueError(f"inter_link endpoint {end!r} unknown")
            parse_bandwidth_gbps(l.bandwidth)
        return self

    # -- ledger compatibility -------------------------------------------------
    def device_ids(self) -> list[str]:
        return [d.id for n in self.nodes for d in n.devices]

    def node_of(self, element_id: str) -> str:
        return element_id.split("/", 1)[0]

    def node(self, node_id: str) -> NodeV2:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"unknown node '{node_id}'")


# ---------------------------------------------------------------------------
# v1 promotion
# ---------------------------------------------------------------------------
def promote_v1(data: dict) -> RegistryV2:
    """Promote a v1 registry dict (inventory schema) to v2. Device power comes
    from the planner power profiles so v1 files need no new fields."""
    from planner.power_profiles import device_active_w, device_idle_w

    nodes: list[NodeV2] = []
    node_ids_with_links: set[str] = set()
    for l in data.get("links", []):
        node_ids_with_links |= {l["src"], l["dst"]}

    for nd in data.get("nodes", []):
        devices: list[DeviceV2] = []
        for dev in nd.get("devices", []):
            hw = dev["name"]
            for i in range(int(dev["count"])):
                devices.append(DeviceV2(
                    id=f"{nd['id']}/{hw}/{i}", hw=hw, mem_gb=dev["mem_gb"],
                    idle_w=device_idle_w(hw), active_w=device_active_w(hw)[0]))
        intra: list[IntraLinkV2] = []
        bw = data.get("intra_node_bandwidth")
        if bw and len(devices) > 1:
            for i in range(len(devices)):
                for j in range(i + 1, len(devices)):
                    intra.append(IntraLinkV2(a=devices[i].id, b=devices[j].id,
                                             kind="pcie", bandwidth=bw))
        switches = ([SwitchV2(id=f"{nd['id']}/nic0", kind="nic")]
                    if nd["id"] in node_ids_with_links else [])
        nodes.append(NodeV2(id=nd["id"], host_base_w=nd.get("host_base_w"),
                            devices=devices, switches=switches, intra_links=intra))

    inter = [InterLinkV2(a=f"{l['src']}/nic0", b=f"{l['dst']}/nic0",
                         kind="ethernet", bandwidth=l["bandwidth"],
                         latency=l.get("latency"))
             for l in data.get("links", [])]
    return RegistryV2(nodes=nodes, inter_links=inter,
                      intra_node_bandwidth=data.get("intra_node_bandwidth"),
                      tp_group_shape=data.get("tp_group_shape"))


def load_registry_any(path: str | Path) -> RegistryV2:
    """Load a registry YAML of either version; v1 is auto-promoted."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data.get("version") == 2:
        return RegistryV2.model_validate(data)
    return promote_v1(data)
