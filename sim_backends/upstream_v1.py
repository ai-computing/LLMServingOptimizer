"""Upstream backend — casys-kaist/LLMServingSim v1.1.0+ (serving/ layout).

Entrypoint ``python -m serving`` using the backend-local ``.venv``
(chakra needs protobuf>=7.35 there — do NOT reuse the legacy python).
New-format profiles under ``profiler/perf/``; absolute CLI paths are fine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .base import ClusterSpec, ScenarioSpec, SimBackend

_DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16",
              "float16": "float16", "bfloat16": "bfloat16"}


class UpstreamBackend(SimBackend):
    name = "upstream"

    def python_exe(self) -> str:
        venv_py = self.root / ".venv" / "bin" / "python"
        return str(venv_py) if venv_py.exists() else sys.executable

    def _path_for_cli(self, p: str) -> str:
        # upstream's config_builder ALSO prepends "../" to the cluster-config
        # path after chdir(astra-sim) — same quirk as legacy, so paths must be
        # relative to the backend root (absolute paths break: '..//abs/...').
        import os
        return os.path.relpath(Path(p).resolve(), self.root)

    # -- config -------------------------------------------------------------
    def build_cluster_config(self, spec: ClusterSpec) -> dict:
        nodes = []
        for node in spec.nodes:
            instances = []
            for inst in node.instances:
                instances.append({
                    "model_name": inst.model_name,
                    "hardware": inst.hardware,
                    "npu_mem": inst.npu_mem,
                    "num_npus": inst.num_npus,
                    "tp_size": inst.tp_size,
                    "pd_type": inst.pd_type,
                })
            nodes.append({
                "num_instances": len(instances),
                "cpu_mem": node.cpu_mem,
                "instances": instances,
                **node.extra,
            })
        return {
            "num_nodes": len(nodes),
            "link_bw": spec.link_bw,
            "link_latency": spec.link_latency,
            "nodes": nodes,
        }

    # -- CLI ------------------------------------------------------------------
    def build_cli(self, cluster_config: str, output: str, scenario: ScenarioSpec,
                  run_id: Optional[str] = None) -> list[str]:
        cmd = [
            self.python_exe(), "-m", "serving",
            "--cluster-config", self._path_for_cli(cluster_config),
            "--dtype", _DTYPE_MAP.get(scenario.dtype, "bfloat16"),
            "--block-size", str(scenario.block_size),
            "--dataset", self._path_for_cli(scenario.dataset),
            "--output", self._path_for_cli(output),
            "--num-reqs", str(scenario.num_reqs),   # 0 = all
            "--log-interval", str(scenario.log_interval),
            "--max-num-seqs", str(scenario.max_num_seqs),
            "--max-num-batched-tokens", str(scenario.max_num_batched_tokens),
            "--request-routing-policy", scenario.request_routing,
        ]
        # upstream defaults are ON (vLLM v1 parity); emit --no-* to disable
        if not scenario.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if not scenario.enable_chunked_prefill:
            cmd.append("--no-enable-chunked-prefill")
        if run_id:
            # run-isolated astra-sim inputs -> safe parallel sweeps
            cmd += ["--run-id", run_id]
        cmd += scenario.extra_args
        return cmd

    # -- profiles ------------------------------------------------------------
    def profile_root(self) -> Path:
        return self.root / "profiler" / "perf"

    def list_hardware(self) -> dict[str, dict[str, list[int]]]:
        """Scan profiler/perf/<HW>/<vendor>/<model>/<variant>/tp<N>/."""
        catalog: dict[str, dict[str, list[int]]] = {}
        root = self.profile_root()
        if not root.is_dir():
            return catalog
        for hw in sorted(p for p in root.iterdir() if p.is_dir()):
            models: dict[str, list[int]] = {}
            for vendor in sorted(p for p in hw.iterdir() if p.is_dir()):
                for model in sorted(p for p in vendor.iterdir() if p.is_dir()):
                    for variant in sorted(p for p in model.iterdir() if p.is_dir()):
                        tps = sorted(int(t.name[2:]) for t in variant.glob("tp*")
                                     if t.is_dir() and t.name[2:].isdigit())
                        if tps:
                            key = f"{vendor.name}/{model.name}"
                            models[key] = sorted(set(models.get(key, []) + tps))
            if models:
                catalog[hw.name] = models
        return catalog
