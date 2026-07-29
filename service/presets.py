"""Workload length-distribution presets (design doc §4.3).

Three service archetypes, each an (input, output) token-length distribution:

* ``chat``      — short prompts, medium replies (ShareGPT-like)
* ``summarize`` — long documents in, short summaries out
* ``agentic``   — medium tool-augmented context in, long multi-step output

Lengths are lognormal, parameterized by (mean, p90) and clamped to
[1, max_len], matching how the ShareGPT traces in ``dataset/`` were capped at
2048 tokens.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

_Z90 = 1.2815515655446004  # standard-normal 90th percentile


@dataclass(frozen=True)
class LengthDist:
    """Lognormal token-length distribution with an upper clamp."""

    mean: float
    p90: float
    max_len: int = 2048

    def __post_init__(self):
        if not (1 <= self.mean < self.p90):
            raise ValueError("need 1 <= mean < p90")
        # mean = exp(mu + s^2/2), p90 = exp(mu + z*s)  =>  s^2/2 - z*s + ln(p90/mean) = 0
        ratio = math.log(self.p90 / self.mean)
        disc = _Z90 * _Z90 - 2.0 * ratio
        if disc < 0:
            raise ValueError(f"p90/mean ratio too large for a lognormal: {self.p90}/{self.mean}")
        object.__setattr__(self, "_sigma", _Z90 - math.sqrt(disc))
        object.__setattr__(self, "_mu", math.log(self.mean) - self._sigma ** 2 / 2.0)

    def sample(self, rng: random.Random) -> int:
        n = int(round(rng.lognormvariate(self._mu, self._sigma)))
        return max(1, min(self.max_len, n))


@dataclass(frozen=True)
class WorkloadPreset:
    name: str
    input_dist: LengthDist
    output_dist: LengthDist
    description: str = ""


PRESETS: dict[str, WorkloadPreset] = {
    "chat": WorkloadPreset(
        name="chat",
        input_dist=LengthDist(mean=256, p90=512),
        output_dist=LengthDist(mean=256, p90=560),
        description="interactive chatbot: short prompts, medium replies",
    ),
    "summarize": WorkloadPreset(
        name="summarize",
        input_dist=LengthDist(mean=1400, p90=2000),
        output_dist=LengthDist(mean=150, p90=300),
        description="document summarization: long inputs, short outputs",
    ),
    "agentic": WorkloadPreset(
        name="agentic",
        input_dist=LengthDist(mean=512, p90=1024),
        output_dist=LengthDist(mean=700, p90=1400),
        description="agent loops: tool-augmented context, long generations",
    ),
}


def get_preset(name: str) -> WorkloadPreset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'; choose from {sorted(PRESETS)}")
    return PRESETS[name]
