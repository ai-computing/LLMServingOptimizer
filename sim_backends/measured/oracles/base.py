"""Instance-oracle protocol for the measured backend (design doc §5.1–5.2).

An oracle answers, for ONE serving instance of (hardware, model, tp), measured
once on real hardware: "at steady state with N concurrent requests, what are
the throughput/latency/power?" The event simulator composes any cluster-level
arrangement (replica counts, routing, P/D) from these answers in software, so
measurement cost is linear in (hw x model x tp) and independent of how many
candidates the planner explores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ExtrapolationWarning(UserWarning):
    """Raised (via warnings.warn) when an oracle is queried outside the range
    it was measured on; the nearest measured point is returned."""


@dataclass(frozen=True)
class Envelope:
    """Measured validity range of an oracle (design doc §8: silent
    extrapolation is an accuracy collapse — always recorded, always checked)."""

    max_concurrency: int
    max_input_toks: int = 2048
    max_output_toks: int = 2048


@dataclass(frozen=True)
class SteadyPoint:
    """One steady-state operating point of an instance."""

    concurrency: int
    thr_toks_s: float      # output-token throughput of the whole instance
    ttft_ms: float         # mean time-to-first-token at this concurrency
    tpot_ms: float         # mean time-per-output-token at this concurrency
    itl_p99_ms: float = 0.0
    avg_w: float = 0.0     # whole-instance average power at this point


@dataclass
class OracleMeta:
    """Provenance — the fidelity router and freshness checks depend on these
    fields (plan §5 risk table)."""

    source: str = "measured"
    stack: str = ""            # e.g. "vllm-0.19" / SDK version hash
    measured_at: str = ""
    extra: dict = field(default_factory=dict)


@runtime_checkable
class InstanceOracle(Protocol):
    hw: str
    model: str
    tp: int
    envelope: Envelope
    meta: OracleMeta

    def steady_state(self, concurrency: int) -> SteadyPoint:
        """Interpolated steady-state point at the given concurrency."""
        ...

    def power_w(self, load: float) -> float:
        """Whole-instance power (W) at fractional load in [0, 1]."""
        ...

    def idle_w(self) -> float:
        """Whole-instance idle power (W) when no request is active."""
        ...
