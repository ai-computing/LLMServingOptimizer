"""vLLM /metrics collector (plan §3.4, §4.3): dependency-free prometheus text
parser, histogram percentiles, and MetricSample derivation.

vLLM metric names drift across versions (risk memo): lookups go through
_ALIASES per field; unknown layouts degrade to zeros with a warning flag
instead of crashing.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+([^\s]+)")


def parse_prometheus(text: str) -> dict[tuple[str, tuple], float]:
    """{(metric_name, sorted label items): value} — counters/gauges/buckets."""
    out: dict[tuple[str, tuple], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels_s, val_s = m.groups()
        labels = []
        if labels_s:
            for part in re.findall(r'(\w+)="([^"]*)"', labels_s):
                labels.append(part)
        try:
            out[(name, tuple(sorted(labels)))] = float(val_s)
        except ValueError:
            continue
    return out


def histogram_quantile(samples: dict, base_name: str, q: float) -> float:
    """Prometheus-style quantile from cumulative ``<base>_bucket{le=...}``
    counts (linear interpolation within the winning bucket)."""
    buckets: list[tuple[float, float]] = []
    for (name, labels), val in samples.items():
        if name != f"{base_name}_bucket":
            continue
        le = dict(labels).get("le")
        if le is None:
            continue
        buckets.append((float("inf") if le == "+Inf" else float(le), val))
    if not buckets:
        return float("nan")
    buckets.sort()
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    target = q * total
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in buckets:
        if cum >= target:
            if le == float("inf"):
                return prev_le
            frac = (target - prev_cum) / max(1e-12, cum - prev_cum)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_cum = le, cum
    return buckets[-1][0]


#: field -> candidate metric base names, first hit wins (version drift table)
_ALIASES: dict[str, list[str]] = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv_cache_usage": ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"],
    "gen_toks": ["vllm:generation_tokens_total"],
    "prompt_toks": ["vllm:prompt_tokens_total"],
    "ttft_hist": ["vllm:time_to_first_token_seconds"],
    "tpot_hist": ["vllm:time_per_output_token_seconds"],
}


@dataclass
class MetricSample:
    ts: float
    dep_id: str
    running: int = 0
    waiting: int = 0
    kv_cache_usage: float = 0.0
    gen_toks_per_s: float = 0.0
    prompt_toks_per_s: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    unknown_layout: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _gauge(samples: dict, names: list[str]) -> Optional[float]:
    for n in names:
        for (name, _), v in samples.items():
            if name == n:
                return v
    return None


@dataclass
class Collector:
    """Derives MetricSample from successive /metrics scrapes (rates need the
    previous counter values)."""
    dep_id: str
    _prev: Optional[tuple[float, float, float]] = None  # ts, gen, prompt

    def sample(self, text: str, ts: Optional[float] = None) -> MetricSample:
        ts = ts if ts is not None else time.time()
        s = parse_prometheus(text)
        running = _gauge(s, _ALIASES["running"])
        m = MetricSample(ts=ts, dep_id=self.dep_id,
                         unknown_layout=running is None)
        m.running = int(running or 0)
        m.waiting = int(_gauge(s, _ALIASES["waiting"]) or 0)
        m.kv_cache_usage = float(_gauge(s, _ALIASES["kv_cache_usage"]) or 0.0)
        gen = _gauge(s, _ALIASES["gen_toks"]) or 0.0
        prompt = _gauge(s, _ALIASES["prompt_toks"]) or 0.0
        if self._prev is not None:
            t0, g0, p0 = self._prev
            dt = max(1e-9, ts - t0)
            m.gen_toks_per_s = max(0.0, (gen - g0) / dt)
            m.prompt_toks_per_s = max(0.0, (prompt - p0) / dt)
        self._prev = (ts, gen, prompt)
        for fld, base_key in (("ttft", "ttft_hist"), ("tpot", "tpot_hist")):
            for base in _ALIASES[base_key]:
                p50 = histogram_quantile(s, base, 0.50) * 1000.0
                p95 = histogram_quantile(s, base, 0.95) * 1000.0
                if p50 == p50:  # not NaN
                    setattr(m, f"{fld}_p50_ms", p50)
                    setattr(m, f"{fld}_p95_ms", p95)
                    break
        return m
