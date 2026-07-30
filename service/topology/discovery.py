"""Hardware auto-discovery: ``nvidia-smi topo -m`` → registry v2 draft (D1).

Parses the interconnect matrix (NV#/PIX/PXB/PHB/SYS/NODE classes) into typed
IntraLinks with conservative default bandwidths. The output is a DRAFT for
human review — exact link speeds vary by platform; the draft records the
classification faithfully and picks defaults per class:

=====  ==========================================  ==================
class  meaning                                     draft link
=====  ==========================================  ==================
NV#    # NVLink lanes                              nvlink, #x28 GBps
PIX    same PCIe switch                            pcie, 32GBps
PXB    multiple PCIe bridges (no host bridge)      pcie, 24GBps
PHB    through the host bridge (CPU)               pcie, 16GBps
NODE   same NUMA node, cross host bridges          pcie, 12GBps
SYS    cross NUMA (SMP interconnect)               pcie, 8GBps
=====  ==========================================  ==================

CLI: ``python -m service.topology.discovery --node-id node0 --hw A40
--mem-gb 48 [--dry-run]`` (runs nvidia-smi locally; --dry-run prints YAML).
"""
from __future__ import annotations

import re
import subprocess
from typing import Optional

import yaml

_CLASS_BW = {"PIX": "32GBps", "PXB": "24GBps", "PHB": "16GBps",
             "NODE": "12GBps", "SYS": "8GBps"}


def parse_topo_matrix(text: str) -> dict[tuple[int, int], str]:
    """{(i, j): class} for i < j from an ``nvidia-smi topo -m`` dump.
    Only GPU-GPU cells are read; NIC/CPU columns are ignored."""
    # Both the column-header line and each matrix row start with GPU<i>; the
    # header appears first, so the real row simply overwrites its dict slot.
    rows: dict[int, list[str]] = {}
    for ln in text.splitlines():
        toks = ln.split()
        m = re.match(r"^GPU(\d+)$", toks[0]) if toks else None
        if m:
            rows[int(m.group(1))] = toks[1:]
    out: dict[tuple[int, int], str] = {}
    n = len(rows)
    for i, cells in rows.items():
        gpu_cells = cells[:n]  # matrix part; trailing cols are CPU/NUMA affinity
        for j, cls in enumerate(gpu_cells):
            if j <= i:
                continue
            cls = cls.strip()
            if cls and cls != "X":
                out[(i, j)] = cls
    return out


def _link_for(cls: str) -> tuple[str, str]:
    m = re.match(r"^NV(\d+)$", cls)
    if m:
        return "nvlink", f"{int(m.group(1)) * 28}GBps"
    return "pcie", _CLASS_BW.get(cls, "8GBps")


def draft_registry(topo_text: str, node_id: str, hw: str, mem_gb: float,
                   hostname: Optional[str] = None,
                   host_base_w: float = 250.0) -> dict:
    """registry v2 draft dict from a topo matrix dump (single node)."""
    from planner.power_profiles import device_active_w, device_idle_w

    matrix = parse_topo_matrix(topo_text)
    n = max((max(i, j) for i, j in matrix), default=-1) + 1
    devices = [{
        "id": f"{node_id}/{hw}/{i}", "kind": "gpu", "hw": hw,
        "mem_gb": mem_gb, "idle_w": device_idle_w(hw),
        "active_w": device_active_w(hw)[0],
    } for i in range(n)]
    intra = []
    for (i, j), cls in sorted(matrix.items()):
        kind, bw = _link_for(cls)
        intra.append({"a": f"{node_id}/{hw}/{i}", "b": f"{node_id}/{hw}/{j}",
                      "kind": kind, "bandwidth": bw})
    node: dict = {"id": node_id, "host_base_w": host_base_w,
                  "devices": devices, "intra_links": intra}
    if hostname:
        node["hostname"] = hostname
    return {"version": 2, "nodes": [node]}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="service.topology.discovery")
    p.add_argument("--node-id", required=True)
    p.add_argument("--hw", required=True)
    p.add_argument("--mem-gb", type=float, required=True)
    p.add_argument("--hostname", default=None)
    p.add_argument("--out", default=None, help="write YAML here (default stdout)")
    p.add_argument("--dry-run", action="store_true", help="print YAML to stdout")
    args = p.parse_args(argv)

    text = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                          text=True, check=True).stdout
    doc = draft_registry(text, args.node_id, args.hw, args.mem_gb,
                         hostname=args.hostname)
    dumped = yaml.safe_dump(doc, sort_keys=False)
    if args.out and not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(dumped)
        print(f"wrote {args.out}")
    else:
        print(dumped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
