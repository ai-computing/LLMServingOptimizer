"""allocation → DeploymentSpec translation (plan §4.2 translation rules).

The planner knows nothing about Docker; this module owns the mapping:
device ids → host-local GPU indices, link kinds → NCCL env, deterministic
container names/ports, vLLM engine args.
"""
from __future__ import annotations

from typing import Optional

from .spec import ContainerSpec, DeploymentSpec, HealthPolicy

DEFAULT_IMAGE = "vllm/vllm-openai:v0.19.0"   # pinned tag (risk memo)
NPU_IMAGE = "furiosaai/furiosa-llm-serving:2024.2"   # pinned; OpenAI-compatible
NPU_HARDWARE = {"RNGD"}
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

    hw = device_ids[0].rsplit("/", 2)[1]
    is_npu = hw in NPU_HARDWARE or (
        graph is not None and device_ids[0] in graph.g
        and graph.g.nodes[device_ids[0]].get("kind") == "npu")
    if is_npu:
        # NPU (RNGD): furiosa runtime image + character-device mapping; the
        # server stays OpenAI-compatible so endpoints/monitoring are unchanged
        container = ContainerSpec(
            node_id=node_id, role="standalone",
            image=NPU_IMAGE if image == DEFAULT_IMAGE else image,
            device_ids=sorted(device_ids), gpu_indices=[],
            env={"FURIOSA_DEVICES": ",".join(
                f"npu{_gpu_index(d)}" for d in sorted(device_ids))},
            ports={"api": port, "metrics": port},
            command=command,
            name=f"llmsvc-{dep_id}-0",
            device_paths=[f"/dev/rngd{_gpu_index(d)}" for d in sorted(device_ids)],
        )
    else:
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


def build_spec_multinode(dep_id: str, model: str,
                         node_devices: list[tuple[str, list[str]]], tp: int,
                         port: int, graph=None, image: str = DEFAULT_IMAGE,
                         engine_overrides: Optional[dict] = None,
                         served_name: Optional[str] = None,
                         head_host: Optional[str] = None) -> DeploymentSpec:
    """Multi-node deployment via Ray (plan D5, §4.2): first node runs the head
    (ray head + vLLM API server), the rest join as workers before the engine
    starts. TP spans within a node; PP degree = number of nodes."""
    if len(node_devices) < 2:
        raise ValueError("multi-node spec needs >= 2 nodes (use build_spec)")
    served = served_name or model.split("/")[-1]
    pp = len(node_devices)
    head_node = node_devices[0][0]
    head_ip = head_host or head_node
    ray_port = 6379

    engine_args = {"tensor-parallel-size": tp, "pipeline-parallel-size": pp,
                   "distributed-executor-backend": "ray",
                   "dtype": "bfloat16", "gpu-memory-utilization": 0.90,
                   "port": port}
    engine_args.update(engine_overrides or {})

    containers: list[ContainerSpec] = []
    for i, (node_id, dev_ids) in enumerate(node_devices):
        if len(dev_ids) != tp:
            raise ValueError(f"node {node_id}: {len(dev_ids)} devices != tp {tp}")
        role = "head" if i == 0 else "worker"
        env = {"VLLM_HOST_IP": head_ip if i == 0 else node_id,
               "RAY_ADDRESS": f"{head_ip}:{ray_port}"}
        env.update(nccl_env_for(graph, dev_ids))
        if role == "head":
            cmd = ["bash", "-c",
                   f"ray start --head --port={ray_port} && "
                   f"python -m vllm.entrypoints.openai.api_server "
                   f"--model {model} --served-model-name {served} " +
                   " ".join(f"--{k} {v}" for k, v in engine_args.items())]
            ports = {"api": port, "metrics": port, "ray": ray_port}
        else:
            cmd = ["bash", "-c",
                   f"ray start --address={head_ip}:{ray_port} --block"]
            ports = {}
        containers.append(ContainerSpec(
            node_id=node_id, role=role, image=image,
            device_ids=sorted(dev_ids),
            gpu_indices=sorted(_gpu_index(d) for d in dev_ids),
            env=env, ports=ports, command=cmd,
            name=f"llmsvc-{dep_id}-{i}"))
    return DeploymentSpec(model=model, served_name=served,
                          engine_args=engine_args, containers=containers,
                          health=HealthPolicy(timeout_s=600))
