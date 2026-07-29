"""NPU bucket measurement campaign (LENS protocol, plan M6 — runner stub).

Per bucket b: two E2E measurements suffice — (1) prefill-only at input=b to
get TTFT_b, (2) long decode inside b to get TBT_b — with power sampled
alongside. The real SDK invocation is hardware-specific (RNGD/Furiosa); this
module ships the command templates, the result parser, and the YAML writer so
the hw runner is a thin shell. Real measurements happen when hardware is
available (hw marker; CI never runs them).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml


@dataclass
class BucketCampaignConfig:
    hw: str
    model: str
    tp: int
    bucket_sizes: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    stack: str = ""              # SDK/compiler version hash — freshness anchor
    measured_at: str = ""
    idle_w: float = 0.0
    max_concurrency: int = 16
    out_root: str = "profiles/measured"


#: (bucket_size) -> {ttft_ms, tbt_ms, avg_w}
BucketMeasureFn = Callable[[int], dict]

_REQUIRED_KEYS = ("ttft_ms", "tbt_ms", "avg_w")


def build_commands(cfg: BucketCampaignConfig) -> list[list[str]]:
    """Furiosa-SDK-shaped command templates: 2 runs per bucket (LENS 2|B|)."""
    cmds = []
    for b in cfg.bucket_sizes:
        common = ["furiosa-bench", cfg.model, "--devices", f"npu:{cfg.tp}"]
        cmds.append(common + ["--mode", "prefill", "--input-len", str(b),
                              "--output-len", "1", "--tag", f"ttft_b{b}"])
        cmds.append(common + ["--mode", "decode", "--input-len", str(b // 2),
                              "--output-len", str(b // 2), "--tag", f"tbt_b{b}"])
    return cmds


def oracle_path(cfg: BucketCampaignConfig) -> Path:
    return Path(cfg.out_root) / cfg.hw / cfg.model / f"tp{cfg.tp}.yaml"


def run_campaign(cfg: BucketCampaignConfig,
                 measure_fn: Optional[BucketMeasureFn] = None,
                 dry_run: bool = False) -> Path | list[list[str]]:
    if dry_run:
        return build_commands(cfg)
    if measure_fn is None:
        raise ValueError("measure_fn is required unless dry_run=True")
    buckets = []
    for size in sorted(cfg.bucket_sizes):
        m = measure_fn(size)
        missing = [k for k in _REQUIRED_KEYS if k not in m]
        if missing:
            raise ValueError(f"measure_fn result for bucket {size} missing {missing}")
        buckets.append({"size": size, **{k: float(m[k]) for k in _REQUIRED_KEYS}})

    doc = {
        "kind": "npu_bucket",
        "hw": cfg.hw, "model": cfg.model, "tp": cfg.tp,
        "idle_w": cfg.idle_w,
        "max_concurrency": cfg.max_concurrency,
        "meta": {"source": "measured", "stack": cfg.stack,
                 "measured_at": cfg.measured_at},
        "buckets": buckets,
    }
    path = oracle_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
