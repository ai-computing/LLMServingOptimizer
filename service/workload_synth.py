"""Workload Synthesizer (design doc §4.3, PLAN M3).

Converts a user scale spec (req/s + length distribution + duration) into

* a backend-neutral ``.jsonl`` dataset — same schema as ``dataset/README.md``:
  ``{input_toks, output_toks, arrival_time_ns, input_tok_ids, output_tok_ids}``
  — consumable by both simulator backends and the measured backend, and
* the equivalent hard-demand rate ``demand_toks_per_s`` for the Stage-1
  power-min constraint.

Everything is seeded and deterministic: same ScaleSpec -> byte-identical file.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .presets import LengthDist, get_preset

NS_PER_S = 1_000_000_000
# synthetic vocab range for generated token ids (values are irrelevant to the
# simulators except for prefix-cache matching; distinct ids avoid fake prefix hits)
_VOCAB_LO, _VOCAB_HI = 1000, 120_000


@dataclass
class ScaleSpec:
    """User-facing service scale description."""

    req_per_s: float
    duration_s: float
    preset: Optional[str] = None            # chat | summarize | agentic
    input_dist: Optional[LengthDist] = None  # explicit override of the preset
    output_dist: Optional[LengthDist] = None
    arrival: str = "poisson"                # poisson | uniform | pulse
    seed: int = 0
    # pulse mode: n bursts, each burst_frac*duration long, idle in between
    pulse_bursts: int = 3
    pulse_burst_frac: float = 0.2

    def resolve_dists(self) -> tuple[LengthDist, LengthDist]:
        inp, out = self.input_dist, self.output_dist
        if self.preset is not None:
            p = get_preset(self.preset)
            inp = inp or p.input_dist
            out = out or p.output_dist
        if inp is None or out is None:
            raise ValueError("ScaleSpec needs a preset or explicit input/output dists")
        return inp, out


@dataclass
class SynthResult:
    path: str
    num_requests: int
    duration_s: float
    demand_toks_per_s: float     # sum(in+out)/duration — Stage-1 demand value
    output_toks_per_s: float     # sum(out)/duration (generation-only rate)
    mean_input_toks: float
    mean_output_toks: float
    stats: dict = field(default_factory=dict)


def expected_toks_per_request(preset: Optional[str] = None,
                              input_dist: Optional[LengthDist] = None,
                              output_dist: Optional[LengthDist] = None) -> float:
    """Analytic E[input+output] tokens per request (no sampling)."""
    if preset is not None:
        p = get_preset(preset)
        input_dist = input_dist or p.input_dist
        output_dist = output_dist or p.output_dist
    if input_dist is None or output_dist is None:
        raise ValueError("need a preset or explicit distributions")
    return float(input_dist.mean + output_dist.mean)


def demand_toks_per_s(req_per_s: float, preset: Optional[str] = None,
                      input_dist: Optional[LengthDist] = None,
                      output_dist: Optional[LengthDist] = None) -> float:
    """req/s -> toks/s conversion used to resolve DemandSpec.req_per_s (M4)."""
    return req_per_s * expected_toks_per_request(preset, input_dist, output_dist)


def _arrival_times_s(spec: ScaleSpec, rng: random.Random) -> list[float]:
    """Arrival timestamps in [0, duration_s) for the chosen process."""
    if spec.arrival == "uniform":
        step = 1.0 / spec.req_per_s
        n = int(spec.req_per_s * spec.duration_s)
        return [i * step for i in range(n)]
    if spec.arrival == "poisson":
        out, t = [], 0.0
        while True:
            t += rng.expovariate(spec.req_per_s)
            if t >= spec.duration_s:
                return out
            out.append(t)
    if spec.arrival == "pulse":
        # n bursts at uniform offsets; within a burst, Poisson at a rate that
        # preserves the overall average req_per_s
        n_total = int(round(spec.req_per_s * spec.duration_s))
        per_burst = max(1, n_total // max(1, spec.pulse_bursts))
        burst_len = spec.duration_s * spec.pulse_burst_frac / max(1, spec.pulse_bursts)
        out = []
        for b in range(spec.pulse_bursts):
            start = b * spec.duration_s / spec.pulse_bursts
            rate = per_burst / burst_len
            t = start
            for _ in range(per_burst):
                t += rng.expovariate(rate)
                if t - start >= burst_len:
                    break
                out.append(min(t, spec.duration_s - 1e-9))
        return sorted(out)
    raise ValueError(f"unknown arrival process '{spec.arrival}'")


def synthesize(spec: ScaleSpec, out_path: str | Path) -> SynthResult:
    """Generate the jsonl dataset and return demand stats (deterministic in seed)."""
    if spec.req_per_s <= 0 or spec.duration_s <= 0:
        raise ValueError("req_per_s and duration_s must be > 0")
    input_dist, output_dist = spec.resolve_dists()
    rng = random.Random(spec.seed)

    arrivals = _arrival_times_s(spec, rng)
    if not arrivals:
        raise ValueError("scale spec produced zero requests; increase duration_s")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tot_in = tot_out = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t_s in arrivals:
            n_in = input_dist.sample(rng)
            n_out = output_dist.sample(rng)
            tot_in += n_in
            tot_out += n_out
            row = {
                "input_toks": n_in,
                "output_toks": n_out,
                "arrival_time_ns": int(round(t_s * NS_PER_S)),
                "input_tok_ids": [rng.randrange(_VOCAB_LO, _VOCAB_HI) for _ in range(n_in)],
                "output_tok_ids": [rng.randrange(_VOCAB_LO, _VOCAB_HI) for _ in range(n_out)],
            }
            f.write(json.dumps(row) + "\n")

    n = len(arrivals)
    return SynthResult(
        path=str(out_path),
        num_requests=n,
        duration_s=spec.duration_s,
        demand_toks_per_s=(tot_in + tot_out) / spec.duration_s,
        output_toks_per_s=tot_out / spec.duration_s,
        mean_input_toks=tot_in / n,
        mean_output_toks=tot_out / n,
        stats={
            "arrival": spec.arrival,
            "seed": spec.seed,
            "target_req_per_s": spec.req_per_s,
            "actual_req_per_s": n / spec.duration_s,
            "preset": spec.preset,
        },
    )
