"""Legacy backend — ai-computing/LLMServingSim fork (branch backend-legacy).

Entrypoint ``main.py``; old-format profiles under ``llm_profile/perf_models/``.
IMPORTANT: the fork's ``config_builder.py`` prepends ``../`` to CLI paths
(``main.py`` chdirs into ``astra-sim/``), so every path we pass on the CLI
must be RELATIVE to the backend root — absolute paths would break.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .base import ClusterSpec, ScenarioSpec, SimBackend


class LegacyBackend(SimBackend):
    name = "legacy"

    def python_exe(self) -> str:
        return sys.executable

    def env(self) -> dict:
        # mirrors the fork's script/serve_webapp.sh SIM_ENV: AnalyticalAstra
        # needs libprotobuf.so.23 and graph_generator needs the `python` shim
        e = dict(os.environ)
        e["LD_LIBRARY_PATH"] = ("/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:"
                                + e.get("LD_LIBRARY_PATH", ""))
        e["PATH"] = os.path.expanduser("~/.local/bin") + ":" + e.get("PATH", "")
        return e

    # -- config ----------------------------------------------------------
    def build_cluster_config(self, spec: ClusterSpec) -> dict:
        nodes = []
        for node in spec.nodes:
            instances = []
            for inst in node.instances:
                instances.append({
                    "model_name": inst.model_name,
                    "hardware": inst.hardware,
                    "npu_mem": inst.npu_mem,
                    # fork semantics: npu_num = total NPUs,
                    # npu_group = NPUs per TP group (= TP degree)
                    "npu_num": inst.num_npus,
                    "npu_group": inst.tp_size,
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

    # -- CLI ------------------------------------------------------------
    def _path_for_cli(self, p: str) -> str:
        return os.path.relpath(Path(p).resolve(), self.root)

    def build_cli(self, cluster_config: str, output: str, scenario: ScenarioSpec,
                  run_id: Optional[str] = None) -> list[str]:
        # run_id is unused: legacy has no run isolation flag; callers doing
        # parallel sweeps must clean stale astra-sim inputs themselves
        # (webapp/runner.py already tags & sweeps them by PID).
        cmd = [
            self.python_exe(), "main.py",
            "--cluster-config", self._path_for_cli(cluster_config),
            "--fp", "16",                     # fp16/bf16 are both 2 bytes here
            "--block-size", str(scenario.block_size),
            "--dataset", self._path_for_cli(scenario.dataset),
            "--output", self._path_for_cli(output),
            "--num-req", str(scenario.num_reqs if scenario.num_reqs > 0 else 100000),
            "--log-interval", str(scenario.log_interval),
            "--max-num-batched-tokens", str(scenario.max_num_batched_tokens),
            "--request-routing-policy",
            "RR" if scenario.request_routing == "LOAD" else scenario.request_routing,
        ]
        if scenario.max_num_seqs:
            cmd += ["--max-batch", str(scenario.max_num_seqs)]
        if scenario.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if scenario.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        cmd += scenario.extra_args
        return cmd

    # -- results ----------------------------------------------------------
    def _normalize_output(self, input_toks: int, output_col: int) -> int:
        # legacy CSV records output = input + generated tokens
        return output_col - input_toks

    # -- profiles ----------------------------------------------------------
    def profile_root(self) -> Path:
        return self.root / "llm_profile" / "perf_models"

    def list_hardware(self) -> dict[str, dict[str, list[int]]]:
        """Scan perf_models/<HW>/<vendor>/<model>/tp<N>/ (old format)."""
        catalog: dict[str, dict[str, list[int]]] = {}
        root = self.profile_root()
        if not root.is_dir():
            return catalog
        for hw in sorted(p for p in root.iterdir() if p.is_dir()):
            models: dict[str, list[int]] = {}
            for vendor in sorted(p for p in hw.iterdir() if p.is_dir()):
                for model in sorted(p for p in vendor.iterdir() if p.is_dir()):
                    tps = sorted(int(t.name[2:]) for t in model.glob("tp*")
                                 if t.is_dir() and t.name[2:].isdigit())
                    if tps:
                        models[f"{vendor.name}/{model.name}"] = tps
            if models:
                catalog[hw.name] = models
        return catalog
