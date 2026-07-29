"""Capacity-curve measurement campaign (design doc §5.2, plan M6).

Sweeps per-instance concurrency against a live vLLM server and writes the
capacity-curve oracle YAML. The actual measurement is injected as a callable
so the planning/formatting logic is unit-testable without hardware; the real
runner (``--hw`` path, hw pytest marker) shells out to ``vllm bench serve``
style commands and samples power via ``nvidia-smi`` in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml


@dataclass
class CampaignConfig:
    hw: str
    model: str
    tp: int
    concurrencies: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    base_url: str = "http://localhost:8000"
    dataset: str = "dataset/sharegpt_req100_rate10_llama.jsonl"
    num_prompts: int = 100
    stack: str = ""              # e.g. "vllm-0.19" — recorded in oracle meta
    measured_at: str = ""
    idle_w: float = 0.0
    out_root: str = "profiles/measured"


#: measurement result for one concurrency level
MeasureFn = Callable[[int], dict]  # -> {thr_toks_s, ttft_ms, tpot_ms, itl_p99_ms, avg_w}

_REQUIRED_KEYS = ("thr_toks_s", "ttft_ms", "tpot_ms", "itl_p99_ms", "avg_w")


def build_commands(cfg: CampaignConfig) -> list[list[str]]:
    """The bench command per concurrency level (dry-run output / hw runner)."""
    cmds = []
    for c in cfg.concurrencies:
        cmds.append([
            "vllm", "bench", "serve",
            "--backend", "openai-chat",
            "--base-url", cfg.base_url,
            "--model", cfg.model,
            "--dataset-name", "custom",
            "--dataset-path", cfg.dataset,
            "--num-prompts", str(cfg.num_prompts),
            "--max-concurrency", str(c),
        ])
    return cmds


def oracle_path(cfg: CampaignConfig) -> Path:
    return Path(cfg.out_root) / cfg.hw / cfg.model / f"tp{cfg.tp}.yaml"


def run_campaign(cfg: CampaignConfig, measure_fn: Optional[MeasureFn] = None,
                 dry_run: bool = False) -> Path | list[list[str]]:
    """Run the sweep and write the oracle YAML; with ``dry_run`` just return
    the commands. ``measure_fn(concurrency) -> point dict`` is the injected
    measurement (the hw runner wraps the real bench; tests inject a mock)."""
    if dry_run:
        return build_commands(cfg)
    if measure_fn is None:
        raise ValueError("measure_fn is required unless dry_run=True "
                         "(real measurement needs live hardware)")
    points = []
    for c in cfg.concurrencies:
        m = measure_fn(c)
        missing = [k for k in _REQUIRED_KEYS if k not in m]
        if missing:
            raise ValueError(f"measure_fn result for c={c} missing {missing}")
        points.append({"concurrency": c, **{k: float(m[k]) for k in _REQUIRED_KEYS}})

    doc = {
        "kind": "capacity_curve",
        "hw": cfg.hw, "model": cfg.model, "tp": cfg.tp,
        "idle_w": cfg.idle_w,
        "meta": {"source": "measured", "stack": cfg.stack,
                 "measured_at": cfg.measured_at},
        "points": points,
    }
    path = oracle_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
