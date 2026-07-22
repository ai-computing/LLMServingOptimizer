"""Backend-agnostic interface for running LLMServingSim variants.

Two backends live under ``backends/``:

* ``legacy``   — ai-computing fork (``main.py`` + ``inference_serving/``,
                 old-format profiles under ``llm_profile/perf_models/``)
* ``upstream`` — casys-kaist v1.1.0+ (``python -m serving`` + ``serving/core/``,
                 new-format profiles under ``profiler/perf/``)

A ``SimBackend`` hides every CLI / config-schema / output-semantics
difference between them so webapp/ and planner/ can target either.
Key differences absorbed here (verified against both codebases):

===================  ============================  =============================
item                 legacy                        upstream
===================  ============================  =============================
entrypoint           ``python main.py``            ``.venv/bin/python -m serving``
dtype flag           ``--fp 16``                   ``--dtype float16|bfloat16``
request count        ``--num-req N``               ``--num-reqs N``
cluster instance     ``npu_num``/``npu_group``     ``num_npus``/``tp_size``
CLI paths            RELATIVE to backend root      absolute OK (isabs handled)
                     (config_builder prepends
                     ``../`` after chdir)
CSV ``output`` col   input+output tokens           pure output tokens
parallel isolation   caller-managed                ``--run-id`` + cleanup
===================  ============================  =============================
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_DIR = REPO_ROOT / "backends"


@dataclass
class ScenarioSpec:
    """Backend-neutral run parameters (paths may be absolute)."""
    dataset: str
    num_reqs: int = 0                    # 0 = all requests in dataset
    dtype: str = "fp16"                  # "fp16" | "bf16"
    block_size: int = 16
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    request_routing: str = "RR"          # RR | RAND | LOAD | CUSTOM
    log_interval: float = 1.0
    extra_args: list[str] = field(default_factory=list)


@dataclass
class InstanceSpec:
    """Backend-neutral per-instance spec for cluster-config generation."""
    model_name: str
    hardware: str
    num_npus: int = 1
    tp_size: int = 1
    npu_mem: dict = field(default_factory=lambda: {"mem_size": 24, "mem_bw": 768, "mem_latency": 0})
    pd_type: Optional[str] = None        # "prefill" | "decode" | None


@dataclass
class NodeSpec:
    instances: list[InstanceSpec] = field(default_factory=list)
    cpu_mem: dict = field(default_factory=lambda: {"mem_size": 128, "mem_bw": 256, "mem_latency": 0})
    extra: dict = field(default_factory=dict)   # power, cxl_mem, ... merged verbatim


@dataclass
class ClusterSpec:
    nodes: list[NodeSpec] = field(default_factory=list)
    link_bw: int = 32
    link_latency: int = 0


class SimBackend(ABC):
    """One simulator variant, rooted at ``backends/<name>/``."""

    #: subclass sets: short registry key ("legacy" / "upstream")
    name: str = ""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else BACKENDS_DIR / self.name

    # -- environment ---------------------------------------------------
    @abstractmethod
    def python_exe(self) -> str: ...

    def env(self) -> dict:
        return dict(os.environ)

    def available(self) -> bool:
        """Backend checked out and its ASTRA-Sim binary built?"""
        binary = (self.root / "astra-sim/build/astra_analytical/build"
                  / "AnalyticalAstra/bin/AnalyticalAstra")
        return binary.exists()

    # -- config / CLI ---------------------------------------------------
    @abstractmethod
    def build_cluster_config(self, spec: ClusterSpec) -> dict: ...

    @abstractmethod
    def build_cli(self, cluster_config: str, output: str, scenario: ScenarioSpec,
                  run_id: Optional[str] = None) -> list[str]:
        """Full argv (argv[0] = python executable). Paths may be absolute;
        each backend converts to what its CLI accepts."""

    def _path_for_cli(self, p: str) -> str:
        """Default: absolute path (upstream handles isabs)."""
        return str(Path(p).resolve())

    # -- execution -------------------------------------------------------
    def run(self, cluster_config: str, output: str, scenario: ScenarioSpec,
            run_id: Optional[str] = None, timeout: int = 1800,
            ) -> subprocess.CompletedProcess:
        cmd = self.build_cli(cluster_config, output, scenario, run_id=run_id)
        return subprocess.run(
            cmd, cwd=str(self.root), env=self.env(),
            capture_output=True, text=True, timeout=timeout,
        )

    # -- result parsing ----------------------------------------------------
    #: CSV columns shared by both backends
    CSV_COLUMNS = ["instance id", "request id", "model", "input", "output",
                   "arrival", "end_time", "latency", "queuing_delay",
                   "TTFT", "TPOT", "ITL"]

    def parse_csv(self, path: str) -> list[dict]:
        """Per-request rows with NORMALIZED semantics:
        ``output`` = pure generated tokens (excl. prompt) for every backend."""
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                row: dict[str, Any] = {
                    "instance_id": int(r["instance id"]),
                    "request_id": int(r["request id"]),
                    "model": r["model"],
                    "input": int(r["input"]),
                    "output": self._normalize_output(int(r["input"]), int(r["output"])),
                    "arrival_ns": int(r["arrival"]),
                    "end_time_ns": int(r["end_time"]),
                    "latency_ns": int(r["latency"]),
                    "queuing_delay_ns": int(r["queuing_delay"]),
                    "ttft_ns": int(r["TTFT"]),
                    "tpot_ns": int(r["TPOT"]),
                    "itl_ns": json.loads(r["ITL"]) if r.get("ITL") else [],
                }
                rows.append(row)
        return rows

    def _normalize_output(self, input_toks: int, output_col: int) -> int:
        return output_col  # upstream: already pure output

    _STDOUT_PATTERNS = {
        "ttft_mean_ms": r"Mean TTFT \(ms\):\s*([0-9.]+)",
        "ttft_p99_ms": r"P99 TTFT \(ms\):\s*([0-9.]+)",
        "tpot_mean_ms": r"Mean TPOT \(ms\):\s*([0-9.]+)",
        "tpot_p99_ms": r"P99 TPOT \(ms\):\s*([0-9.]+)",
        "total_token_throughput": r"Total token throughput \(tok/s\):\s*([0-9.]+)",
        "request_throughput": r"Request throughput \(req/s\):\s*([0-9.]+)",
        "total_energy_kj": r"Total energy consumption \(kJ\):\s*([0-9.]+)",
    }

    def parse_stdout(self, text: str) -> dict:
        """Summary metrics from simulator stdout (ANSI codes stripped).
        Missing metrics are simply absent from the dict."""
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        out: dict[str, float] = {}
        for key, pat in self._STDOUT_PATTERNS.items():
            m = re.search(pat, clean)
            if m:
                out[key] = float(m.group(1))
        return out

    # -- profile catalog ---------------------------------------------------
    @abstractmethod
    def profile_root(self) -> Path: ...

    @abstractmethod
    def list_hardware(self) -> dict[str, dict[str, list[int]]]:
        """{hardware: {"vendor/model": [tp, ...]}} scanned from profile_root."""
