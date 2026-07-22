"""Build cluster JSON dicts from ConfigSpec dataclasses.

The output JSON shape matches what `inference_serving/config_builder.py`
expects (see config_builder.py:38-122).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from .config import HW_DEFAULTS


@dataclass
class InstanceSpec:
    hardware: str
    model: str
    npu_num: int               # total NPUs for this instance
    npu_group: int             # PP stages (npu_num // npu_group = TP per stage)
    pd_type: Optional[str]     # "prefill", "decode", or None
    npu_mem: dict = field(default_factory=dict)  # mem_size/mem_bw/mem_latency


@dataclass
class ConfigSpec:
    label: str
    instances: list[InstanceSpec]   # each entry = one cluster instance
    tp: int      # display value
    pp: int      # display value
    dp: int      # display value (num_instances per role)
    pd_layout: str  # "—" or "1P+1D" etc.


def _resolve_npu_mem(hardware: str, override: dict) -> dict:
    """Pick npu_mem values: explicit override wins, else HW_DEFAULTS, else hard fallback."""
    base = HW_DEFAULTS.get(hardware, {"mem_size": 40, "mem_bw": 768, "mem_latency": 0})
    return {
        "mem_size":    override.get("mem_size",    base["mem_size"]),
        "mem_bw":      override.get("mem_bw",      base["mem_bw"]),
        "mem_latency": override.get("mem_latency", base["mem_latency"]),
    }


def build_cluster_json(
    spec: ConfigSpec,
    cpu_mem: dict,
    link_bw,
    link_latency,
    *,
    num_nodes: int = 1,
    cpu_mem_per_node: "list[dict] | None" = None,
    instances_per_node: "list[list[InstanceSpec]] | None" = None,
    power_template: "dict | None" = None,
    tp_group_shape: "list[int] | None" = None,
    collective_overhead: "dict | None" = None,
    backend: str = "legacy",
) -> dict:
    """Produce a cluster JSON dict accepted by config_builder.py.

    Backwards compatible: omitting the new kwargs gives the original single-node behaviour.
    When instances_per_node is provided, each list element becomes one node.
    When power_template is provided (a node-level "power" dict from a loaded
    cluster config), it is attached to every generated node so the simulator
    enables power modeling. Each node gets its own deep copy because
    config_builder.py mutates power["npu"][hw]["num_npus"] in place.

    link_bw / link_latency may be a scalar (uniform) or a per-dimension list
    ordered innermost-first; the latter pairs with tp_group_shape to drive the
    hierarchical topology in config_builder._create_network_config. When
    tp_group_shape is given it is written to the top level of the cluster JSON.
    """
    if instances_per_node is None:
        nodes_instances: list[list[InstanceSpec]] = [spec.instances]
        num_nodes = 1
    else:
        nodes_instances = instances_per_node
        assert len(nodes_instances) == num_nodes

    if cpu_mem_per_node is None:
        cpu_mem_per_node = [cpu_mem] * num_nodes

    def _instance_json(inst: InstanceSpec) -> dict:
        base = {
            "model_name": inst.model,
            "hardware":   inst.hardware,
            "npu_mem":    _resolve_npu_mem(inst.hardware, inst.npu_mem),
            "pd_type":    inst.pd_type,
        }
        if backend == "upstream":
            # upstream schema: num_npus + tp_size (tp = npus per group)
            base["num_npus"] = inst.npu_num
            base["tp_size"] = inst.npu_num // inst.npu_group
        else:
            base["npu_num"] = inst.npu_num
            base["npu_group"] = inst.npu_group
        return base

    nodes_json = []
    for node_idx, inst_list in enumerate(nodes_instances):
        instances_json = [_instance_json(inst) for inst in inst_list]
        cm = cpu_mem_per_node[node_idx]
        node_json: dict = {
            "num_instances": len(instances_json),
            "cpu_mem": {
                "mem_size":    cm["mem_size"],
                "mem_bw":      cm["mem_bw"],
                "mem_latency": cm["mem_latency"],
            },
            "instances": instances_json,
        }
        if power_template:
            node_json["power"] = copy.deepcopy(power_template)
        nodes_json.append(node_json)

    cluster: dict = {
        "num_nodes":    num_nodes,
        "link_bw":      link_bw,
        "link_latency": link_latency,
        "nodes": nodes_json,
    }
    if tp_group_shape:
        cluster["tp_group_shape"] = list(tp_group_shape)
    if collective_overhead:
        cluster["collective_overhead"] = collective_overhead
    return cluster


def validate_spec(
    spec: ConfigSpec,
    catalog: dict[tuple[str, str], frozenset[int]],
) -> list[str]:
    """Return a list of validation error messages (empty list = valid).

    Checks:
    - npu_group >= 1
    - npu_num >= npu_group, and npu_num % npu_group == 0
    - npus_per_group (= npu_num / npu_group) is in the profiled TP set
    - pd_type is "prefill", "decode", or None
    - At least one instance is present
    """
    errors: list[str] = []

    if not spec.instances:
        errors.append(f"[{spec.label}] no instances defined")
        return errors

    for idx, inst in enumerate(spec.instances):
        prefix = f"[{spec.label}] instance#{idx} ({inst.hardware}/{inst.model})"

        if inst.npu_group < 1:
            errors.append(f"{prefix}: npu_group ({inst.npu_group}) must be >= 1")
        if inst.npu_num < 1:
            errors.append(f"{prefix}: npu_num ({inst.npu_num}) must be >= 1")
        if inst.npu_group > inst.npu_num:
            errors.append(
                f"{prefix}: npu_group ({inst.npu_group}) > npu_num ({inst.npu_num})"
            )
        elif inst.npu_num % inst.npu_group != 0:
            errors.append(
                f"{prefix}: npu_num ({inst.npu_num}) not divisible by "
                f"npu_group ({inst.npu_group})"
            )
        else:
            npus_per_group = inst.npu_num // inst.npu_group
            tp_options = catalog.get((inst.hardware, inst.model), frozenset())
            if not tp_options:
                errors.append(
                    f"{prefix}: no profile data found for ({inst.hardware}, {inst.model})"
                )
            elif npus_per_group not in tp_options:
                errors.append(
                    f"{prefix}: TP={npus_per_group} not in profiled set "
                    f"{sorted(tp_options)}"
                )

        if inst.pd_type not in (None, "prefill", "decode"):
            errors.append(
                f"{prefix}: invalid pd_type {inst.pd_type!r} "
                f"(must be 'prefill', 'decode', or None)"
            )

    return errors
