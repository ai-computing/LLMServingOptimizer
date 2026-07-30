"""allocation → DeploymentSpec translation (plan §4.2 translation rules).

The planner knows nothing about Docker; this module owns the mapping:
device ids → host-local GPU indices, link kinds → NCCL env, deterministic
container names/ports, vLLM engine args.
"""
from __future__ import annotations

from typing import Optional

from .spec import ContainerSpec, DeploymentSpec, HealthPolicy

DEFAULT_IMAGE = "vllm/vllm-openai:v0.19.0"   # pinned tag (risk memo)
PORT_POOL = range(8001, 8100)


def _gpu_index(device_id: str) -> int:
    return int(device_id.rsplit("/", 1)[1])


def nccl_env_for(graph, device_ids: list[str]) -> dict[str, str]:
    """NCCL knobs from the link kinds inside the TP group: all-NVLink groups
    enable P2P over NVL; PCIe-bridged groups disable P2P (matches the A40
    NVLink-vs-PCIe ablation findings)."""
    if graph is None or len(device_ids) < 2:
        return {}
    kinds = set()
    sub = graph.g.subgraph(device_ids)
    for _u, _v, d in sub.edges(data=True):
        kinds.add(d["kind"])
    if kinds and kinds <= {"nvlink", "xgmi"}:
        return {"NCCL_P2P_LEVEL": "NVL"}
    return {"NCCL_P2P_DISABLE": "1"}


def pick_port(used_ports: set[int]) -> int:
    for p in PORT_POOL:
        if p not in used_ports:
            return p
    raise RuntimeError("port pool 8001-8099 exhausted")


def build_spec(dep_id: str, model: str, device_ids: list[str], tp: int,
               port: int, graph=None, image: str = DEFAULT_IMAGE,
               engine_overrides: Optional[dict] = None,
               served_name: Optional[str] = None) -> DeploymentSpec:
    """Single-node deployment spec (multi-node/Ray lands in D5)."""
    if len(device_ids) != tp:
        raise ValueError(f"device count {len(device_ids)} != tp {tp}")
    nodes = {d.rsplit("/", 2)[0] for d in device_ids}
    if len(nodes) != 1:
        raise ValueError(f"single-node launcher got devices on {sorted(nodes)}; "
                         "multi-node deployment is a D5 feature")
    node_id = nodes.pop()
    served = served_name or model.split("/")[-1]

    engine_args = {"tensor-parallel-size": tp, "dtype": "bfloat16",
                   "gpu-memory-utilization": 0.90, "port": port}
    engine_args.update(engine_overrides or {})

    command = ["--model", model, "--served-model-name", served]
    for k, v in engine_args.items():
        command += [f"--{k}", str(v)]

    container = ContainerSpec(
        node_id=node_id,
        role="standalone",
        image=image,
        device_ids=sorted(device_ids),
        gpu_indices=sorted(_gpu_index(d) for d in device_ids),
        env=nccl_env_for(graph, device_ids),
        ports={"api": port, "metrics": port},
        command=command,
        name=f"llmsvc-{dep_id}-0",   # NAME_PREFIX convention (reconcile key)
    )
    return DeploymentSpec(model=model, served_name=served,
                          engine_args=engine_args, containers=[container],
                          health=HealthPolicy())
