"""Shared dataclasses passed between planner stages.

Kept dependency-free (stdlib only) so every module can import them cheaply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Device:
    """A homogeneous group of accelerators of one hardware type on one node."""

    node_id: str
    hardware: str
    count: int
    mem_gb: float

    def key(self) -> tuple[str, str]:
        return (self.node_id, self.hardware)


@dataclass
class Instance:
    """One serving instance in a candidate allocation.

    ``tp`` is the tensor-parallel degree (serialized to ``npu_group``).
    ``npu_num`` is the total device count assigned to the instance; the number of
    data-parallel replicas is ``npu_num // tp``.
    """

    node_id: str
    hardware: str
    model_name: str
    tp: int
    npu_num: int
    npu_mem_gb: float
    pd_type: Optional[str] = None  # "prefill" | "decode" | None (combined)

    @property
    def replicas(self) -> int:
        return self.npu_num // self.tp


@dataclass
class Allocation:
    """A full candidate placement produced by Stage 1 (the MILP solver)."""

    instances: list[Instance] = field(default_factory=list)
    # coarse Stage-1 proxy score (higher = better); refined by Stage 2
    proxy_score: float = 0.0
    meta: dict = field(default_factory=dict)

    def total_devices(self) -> int:
        return sum(i.npu_num for i in self.instances)

    def signature(self) -> str:
        """Stable string identity used for de-duplication and caching."""
        parts = sorted(
            f"{i.node_id}|{i.hardware}|{i.model_name}|tp{i.tp}|n{i.npu_num}|{i.pd_type}"
            for i in self.instances
        )
        return ";".join(parts)


@dataclass
class Metrics:
    """Aggregated Stage-2 simulation results for one allocation."""

    ttft_ms: float
    tpot_ms: float
    itl_p99_ms: float
    throughput_toks_s: float
    #: Highest generation rate the run actually demonstrated, over a sliding
    #: window (toks/s). ``throughput_toks_s`` cannot answer "did it keep up?":
    #: the trace fixes how many tokens exist, so tokens/span reports the ARRIVAL
    #: rate however fast the server is. The trace is synthesized AT the demanded
    #: rate, so a candidate that never reached that rate in any window did not
    #: sustain the demand.
    peak_gen_toks_s: Optional[float] = None
    # queue wait = arrival to admission. A direct backlog signal, but it only
    # bites once the admission cap binds (the measured backend caps at the
    # oracle's max profiled concurrency, so a short evaluation trace never
    # queues at all) -- hence peak_gen_toks_s above is the primary test.
    queue_p95_ms: Optional[float] = None
    queue_max_ms: Optional[float] = None
    energy_j: Optional[float] = None
    toks_per_wh: Optional[float] = None
    #: generated tokens per joule == tokens/s per watt (same number, both are
    #: "token bandwidth per unit power"); toks_per_wh / 3600
    toks_per_j: Optional[float] = None
    # average cluster power (W): sim energy/wall-clock when the backend reports
    # energy, else a power-profile estimate (raw["power_source"] says which)
    power_w: Optional[float] = None
    num_requests: int = 0
    raw: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class Infeasible:
    """Marker returned when a candidate cannot be evaluated (crash/OOM/timeout)."""

    reason: str


@dataclass
class InfeasibleReport:
    """First-class infeasibility diagnosis (design doc §6).

    ``bottleneck`` names the constraint family whose relaxation makes the model
    SAT: memory | demand | availability | links | structure for Stage-1, plus
    slo | evaluation when Stage-2 rejected every candidate.
    """

    bottleneck: str
    detail: str
    # max achievable demand (toks/s) when bottleneck == "demand"/"availability"
    max_achievable_toks_s: Optional[float] = None
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def is_finite_positive(x: Optional[float]) -> bool:
    return x is not None and math.isfinite(x) and x > 0
