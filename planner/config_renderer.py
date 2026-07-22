"""Render an ``Allocation`` into a ``cluster_config/*.json`` + ``main.py`` CLI args.

Field mapping (verified against ``cluster_config/README.md`` and
``inference_serving/config_builder.py``):

    tp (TP degree)      -> instances[].npu_group
    devices assigned    -> instances[].npu_num   (replicas = npu_num / npu_group)
    role (P/D)          -> instances[].pd_type
    hierarchical fabric -> top-level tp_group_shape (+ list link_bw/link_latency)

Paths are emitted repo-root-relative because ``build_cluster_config`` prepends
``../`` when reading them (main.py runs after ``os.chdir("astra-sim")``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .graph_model import default_link_params
from .spec_schema import PlannerSpec
from .types import Allocation
from .utils import get_logger, stage_path

log = get_logger("planner.render")

# per-hardware NPU memory bandwidth (GB/s) used to fill npu_mem.mem_bw
_HW_MEM_BW = {
    "H100": 3350, "A100": 2039, "A6000": 768, "A40": 696,
    "A40x": 696, "A5000": 768, "RTX3090": 936, "RNGD": 1500, "TPU-v6e-1": 819,
}
# per-hardware power table for optional power modeling (enables toks_per_wh)
_HW_POWER = {
    "H100": dict(idle_power=100, standby_power=250, active_power=700, standby_duration=18),
    "A100": dict(idle_power=50, standby_power=150, active_power=400, standby_duration=18),
    "A6000": dict(idle_power=25, standby_power=115, active_power=300, standby_duration=18),
    "A40": dict(idle_power=25, standby_power=115, active_power=300, standby_duration=18),
    "A40x": dict(idle_power=25, standby_power=115, active_power=300, standby_duration=18),
    "A5000": dict(idle_power=20, standby_power=90, active_power=230, standby_duration=18),
    "RTX3090": dict(idle_power=30, standby_power=120, active_power=350, standby_duration=18),
    "RNGD": dict(idle_power=15, standby_power=60, active_power=150, standby_duration=18),
    "TPU-v6e-1": dict(idle_power=30, standby_power=90, active_power=200, standby_duration=18),
}

_DEFAULT_CPU_MEM = {"mem_size": 128, "mem_bw": 256, "mem_latency": 0}


def _build_power_block(hardwares: set[str]) -> dict:
    npu = {hw: _HW_POWER.get(hw, _HW_POWER["A6000"]) for hw in hardwares}
    return {
        "base_node_power": 60,
        "npu": npu,
        "cpu": {"idle_power": 10, "active_power": 200, "util": 0.15},
        "dram": {"dimm_size": 32, "idle_power": 2.0, "energy_per_bit": 6.0},
        "link": {"num_links": 1, "idle_power": 5, "energy_per_bit": 4.0},
        "nic": {"num_nics": 1, "idle_power": 20},
        "storage": {"num_devices": 2, "idle_power": 5},
    }


def _wants_energy(spec: PlannerSpec) -> bool:
    return any(o.metric == "toks_per_wh" for o in spec.requirements.objectives)


def render(
    allocation: Allocation,
    spec: PlannerSpec,
    out_dir: str | Path,
    run_id: str,
    batch_tokens: Optional[int] = None,
    backend: str = "legacy",
) -> tuple[str, list[str]]:
    """Write a cluster_config JSON and return (repo_relative_config_path, cli_args)."""
    out_dir = Path(out_dir)
    config_path, rel_config = stage_path(out_dir, f"configs/{run_id}.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    link_bw, link_latency = default_link_params(spec)
    include_power = _wants_energy(spec)

    # group instances by node
    by_node: dict[str, list] = {}
    for inst in allocation.instances:
        by_node.setdefault(inst.node_id, []).append(inst)

    nodes_json = []
    for node_id, insts in by_node.items():
        inst_json = []
        for inst in insts:
            # Emit ONE config-instance per replica, each a single TP group:
            #   npu_num = TP degree, npu_group = 1.
            # The simulator derives TP as npu_num // npu_group (verified against
            # cluster_config/{h100_4gpu_tp4,a40_8gpu_tp8}_70b.json), so a single
            # TP-t group is npu_num=t, npu_group=1 — NOT npu_group=t.
            for _ in range(inst.replicas):
                entry = {
                    "model_name": inst.model_name,
                    "hardware": inst.hardware,
                    "npu_mem": {
                        "mem_size": inst.npu_mem_gb,
                        "mem_bw": _HW_MEM_BW.get(inst.hardware, 768),
                        "mem_latency": 0,
                    },
                    "pd_type": inst.pd_type,
                }
                if backend == "upstream":
                    # upstream schema: one TP-t instance = num_npus=t, tp_size=t
                    entry["num_npus"] = inst.tp
                    entry["tp_size"] = inst.tp
                else:
                    entry["npu_num"] = inst.tp
                    entry["npu_group"] = 1
                inst_json.append(entry)
        node_obj = {
            "num_instances": len(inst_json),
            "cpu_mem": dict(_DEFAULT_CPU_MEM),
            "instances": inst_json,
        }
        if include_power:
            node_obj["power"] = _build_power_block({i.hardware for i in insts})
        nodes_json.append(node_obj)

    config = {
        "num_nodes": len(nodes_json),
        "link_bw": link_bw,
        "link_latency": link_latency,
        "nodes": nodes_json,
    }
    if spec.topology.tp_group_shape:
        config["tp_group_shape"] = spec.topology.tp_group_shape

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    bt = batch_tokens if batch_tokens is not None else spec.search_space.batch_tokens_choices[0]
    if backend == "upstream":
        cli_args = [
            "--cluster-config", rel_config,
            "--dtype", "bfloat16" if str(spec.model.fp) == "16" else "float32",
            "--block-size", "16",
            "--dataset", spec.workload.dataset,
            "--num-reqs", str(spec.workload.num_req),
            "--max-num-batched-tokens", str(bt),
            "--request-routing-policy", "RR",
            "--log-interval", "1.0",
            # match legacy scheduling semantics (upstream defaults are ON)
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
        ]
    else:
        cli_args = [
            "--cluster-config", rel_config,
            "--fp", str(spec.model.fp),
            "--block-size", "16",
            "--dataset", spec.workload.dataset,
            "--num-req", str(spec.workload.num_req),
            "--max-num-batched-tokens", str(bt),
            "--request-routing-policy", "RR",
            "--log-interval", "1.0",
        ]
    return rel_config, cli_args
