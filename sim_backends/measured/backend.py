"""``MeasuredBackend`` — the third ``SimBackend`` (design doc §5).

Same contract as the legacy/upstream adapters, but evaluation is in-process:
no subprocess, no ASTRA-Sim, no chakra. ``run()`` loads the cluster config,
resolves per-instance oracles from ``profiles/measured/``, runs the event
simulator, writes a backend-schema CSV, and returns a synthetic
``CompletedProcess`` whose stdout matches the summary patterns the existing
parsers scan for. A ``python -m sim_backends.measured`` entry point exists for
callers that must shell out (planner's sim_evaluator).

Cluster-config schema: upstream-style (``num_npus``/``tp_size`` instances),
plus optional measured-only keys per node: ``host_base_w``; top-level:
``routing`` (RR|WEIGHTED), ``kv_link_gbps``, ``kv_bytes_per_token``.
Oracle YAML resolution: ``<oracle_root>/<hw>/<model_name>/tp<N>.yaml`` where
``oracle_root`` is ``$LLMSS_MEASURED_ORACLES`` or ``profiles/measured/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..base import REPO_ROOT, ClusterSpec, ScenarioSpec, SimBackend
from .event_sim import (
    ClusterModel,
    EventSimulator,
    HostCfg,
    InstanceCfg,
    load_jsonl_workload,
    summary_stdout,
    write_csv,
)
from .oracles.capacity_curve import CapacityCurveOracle

DEFAULT_ORACLE_ROOT = REPO_ROOT / "profiles" / "measured"
_ORACLE_ROOT_ENV = "LLMSS_MEASURED_ORACLES"


def oracle_root() -> Path:
    return Path(os.environ.get(_ORACLE_ROOT_ENV, str(DEFAULT_ORACLE_ROOT)))


def load_oracle(hw: str, model: str, tp: int, root: Optional[Path] = None):
    """Resolve + load the oracle YAML for (hw, model, tp).

    Dispatches on the YAML ``kind`` field: capacity_curve (default) or
    npu_bucket (M6)."""
    root = root or oracle_root()
    path = root / hw / model / f"tp{tp}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no measured oracle for ({hw}, {model}, tp{tp}); expected {path}")
    import yaml
    with open(path, encoding="utf-8") as f:
        kind = (yaml.safe_load(f) or {}).get("kind", "capacity_curve")
    if kind == "capacity_curve":
        return CapacityCurveOracle.from_yaml(path)
    if kind == "npu_bucket":
        from .oracles.npu_bucket import NpuBucketOracle
        return NpuBucketOracle.from_yaml(path)
    raise ValueError(f"unknown oracle kind '{kind}' in {path}")


def cluster_model_from_config(config: dict, root: Optional[Path] = None) -> ClusterModel:
    instances: list[InstanceCfg] = []
    hosts: list[HostCfg] = []
    iid = 0
    for ni, node in enumerate(config.get("nodes", [])):
        node_id = node.get("id", f"node{ni}")
        base_w = float(node.get("host_base_w", 0.0))
        if base_w > 0:
            hosts.append(HostCfg(node_id=node_id, base_w=base_w))
        for inst in node.get("instances", []):
            hw = inst["hardware"]
            model = inst["model_name"]
            tp = int(inst.get("tp_size") or inst.get("npu_num") or 1)
            n_npus = int(inst.get("num_npus") or inst.get("npu_num") or tp)
            replicas = max(1, n_npus // tp)
            oracle = load_oracle(hw, model, tp, root=root)
            for _ in range(replicas):
                instances.append(InstanceCfg(
                    inst_id=iid, node_id=node_id, oracle=oracle,
                    role=inst.get("pd_type"),
                    weight=float(inst.get("weight", 1.0)),
                    admit_cap=inst.get("admit_cap")))
                iid += 1
    return ClusterModel(
        instances=instances,
        hosts=hosts,
        routing=str(config.get("routing", "RR")).upper(),
        kv_link_gbps=float(config.get("kv_link_gbps", 10.0)),
        kv_bytes_per_token=float(config.get("kv_bytes_per_token", 131072.0)),
    )


def run_from_files(cluster_config: str | Path, dataset: str | Path,
                   output_csv: str | Path, num_reqs: int = 0,
                   routing: Optional[str] = None) -> str:
    """Shared by ``run()`` and ``__main__``; returns the summary stdout text."""
    with open(cluster_config, encoding="utf-8") as f:
        config = json.load(f)
    if routing:
        config["routing"] = routing
    cluster = cluster_model_from_config(config)
    workload = load_jsonl_workload(dataset, num_reqs=num_reqs)
    out = EventSimulator(cluster).run(workload)
    write_csv(out, output_csv)
    return summary_stdout(out)


class MeasuredBackend(SimBackend):
    name = "measured"

    def __init__(self, root: Optional[Path] = None):
        super().__init__(root=root or REPO_ROOT)

    # -- environment ---------------------------------------------------------
    def python_exe(self) -> str:
        return sys.executable

    def available(self) -> bool:
        return True  # pure python, always runnable

    # -- config / CLI ----------------------------------------------------------
    def build_cluster_config(self, spec: ClusterSpec) -> dict:
        nodes = []
        for ni, node in enumerate(spec.nodes):
            insts = []
            for inst in node.instances:
                insts.append({
                    "model_name": inst.model_name,
                    "hardware": inst.hardware,
                    "num_npus": inst.num_npus,
                    "tp_size": inst.tp_size,
                    "npu_mem": dict(inst.npu_mem),
                    "pd_type": inst.pd_type,
                })
            entry = {"id": f"node{ni}", "num_instances": len(insts),
                     "instances": insts}
            entry.update(node.extra)  # host_base_w etc. pass through
            nodes.append(entry)
        return {"num_nodes": len(nodes), "link_bw": spec.link_bw,
                "link_latency": spec.link_latency, "nodes": nodes}

    def build_cli(self, cluster_config: str, output: str, scenario: ScenarioSpec,
                  run_id: Optional[str] = None) -> list[str]:
        argv = [self.python_exe(), "-m", "sim_backends.measured",
                "--cluster-config", self._path_for_cli(cluster_config),
                "--dataset", self._path_for_cli(scenario.dataset),
                "--output", self._path_for_cli(output),
                "--num-reqs", str(scenario.num_reqs),
                "--request-routing-policy", scenario.request_routing]
        if run_id:
            argv += ["--run-id", run_id]
        return argv

    # -- execution: in-process, subprocess-shaped result -------------------------
    def run(self, cluster_config: str, output: str, scenario: ScenarioSpec,
            run_id: Optional[str] = None, timeout: int = 1800,
            ) -> subprocess.CompletedProcess:
        cmd = self.build_cli(cluster_config, output, scenario, run_id=run_id)
        routing = "WEIGHTED" if scenario.request_routing == "CUSTOM" else "RR"
        try:
            stdout = run_from_files(cluster_config, scenario.dataset, output,
                                    num_reqs=scenario.num_reqs, routing=routing)
        except Exception as e:  # mirror subprocess semantics: fail via returncode
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="",
                                               stderr=f"{type(e).__name__}: {e}")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout,
                                           stderr="")

    # -- profile catalog ----------------------------------------------------------
    def profile_root(self) -> Path:
        return oracle_root()

    def list_hardware(self) -> dict[str, dict[str, list[int]]]:
        """{hw: {"vendor/model": [tp, ...]}} scanned from oracle YAML layout
        <root>/<hw>/<vendor>/<model>/tp<N>.yaml."""
        root = oracle_root()
        out: dict[str, dict[str, list[int]]] = {}
        if not root.is_dir():
            return out
        for hw_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            models: dict[str, list[int]] = {}
            for vendor_dir in sorted(p for p in hw_dir.iterdir() if p.is_dir()):
                for model_dir in sorted(p for p in vendor_dir.iterdir() if p.is_dir()):
                    tps = sorted(int(f.stem[2:]) for f in model_dir.glob("tp*.yaml")
                                 if f.stem[2:].isdigit())
                    if tps:
                        models[f"{vendor_dir.name}/{model_dir.name}"] = tps
            if models:
                out[hw_dir.name] = models
        return out
