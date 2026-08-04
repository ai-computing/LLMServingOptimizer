"""Hard-constraint checking, scalar scoring, and Pareto-front selection.

Operates on :class:`Metrics` (Stage-2 output) against the spec's
:class:`Requirements`. Missing metrics (e.g. ``toks_per_wh`` when power modeling
was not configured) are treated as unavailable and excluded from scoring rather
than silently counted as zero.
"""
from __future__ import annotations

import math
from typing import Optional

from .spec_schema import Requirements
from .types import Metrics

_OPS = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def _metric_value(metrics: Metrics, name: str) -> Optional[float]:
    return {
        "ttft_ms": metrics.ttft_ms,
        "tpot_ms": metrics.tpot_ms,
        "itl_p99_ms": metrics.itl_p99_ms,
        "throughput": metrics.throughput_toks_s,
        "toks_per_wh": metrics.toks_per_wh,
        "toks_per_j": metrics.toks_per_j,
        "power_w": metrics.power_w,
    }.get(name)


def check_demand_attainment(metrics: Metrics, req: Requirements) -> list[str]:
    """Stage-2 check that the candidate actually served the offered load.

    Stage-1 only compares a demand estimate against a capacity ceiling; here the
    arrival trace has really been played, so falling short is observable.

    It deliberately does NOT compare ``throughput_toks_s`` against the demand:
    the trace fixes how many tokens exist, so tokens/span measures the arrival
    rate, not the server's ceiling -- an idle-ish instance and a saturated one
    both report roughly the offered rate (the A5000 14B AWQ run sat at 266
    toks/s against a 496 toks/s demand while perfectly healthy, purely because
    requests arrived that slowly). Two signals that do carry information:

    * ``peak_gen_toks_s`` -- the trace is synthesized at the demanded rate, so a
      candidate that never once generated at that rate did not keep up.
    * queue wait -- unambiguous when it fires, but it only fires once the
      admission cap binds, which a short evaluation trace never reaches.
    """
    demand = req.demand
    if demand is None:
        return []
    q = metrics.queue_p95_ms
    if demand.max_queue_ms and q is not None and q > demand.max_queue_ms:
        worst = (f", max {metrics.queue_max_ms:.0f}ms"
                 if metrics.queue_max_ms is not None else "")
        return [f"demand not sustained: queue wait p95={q:.0f}ms exceeds "
                f"{demand.max_queue_ms:.0f}ms{worst} — requests are backing up"]
    return []


def demand_shortfall(metrics: Metrics, req: Requirements) -> Optional[str]:
    """Advisory: the run never generated at the demanded rate.

    Deliberately NOT a violation. A short evaluation trace can fail to
    demonstrate a rate the hardware can reach: with the default 50-request
    evaluation a 5 req/s demand only covers ~10 s of arrivals, so concurrency
    never climbs to where the measured curve peaks (A5000 14B AWQ: 710 toks/s at
    c=16 but 1100 at c=64). Failing on that would recreate the false
    "달성 불가" this whole path just got fixed for. Requests backing up IS
    conclusive, and that is what :func:`check_demand_attainment` fails on.
    """
    demand = req.demand
    if demand is None or not demand.toks_per_s:
        return None
    peak, target = metrics.peak_gen_toks_s, demand.toks_per_s
    if peak is None or demand.attainment_tolerance >= 1.0:
        return None
    if peak >= target * (1.0 - demand.attainment_tolerance):
        return None
    return (f"peak generation {peak:.0f} toks/s stayed below the {target:.0f} "
            f"toks/s demanded ({(1 - peak / target):.0%} short) while requests "
            f"never backed up — raise num_req_eval to evaluate the full arrival "
            f"window before trusting this either way")


def check_constraints(metrics: Metrics, req: Requirements) -> tuple[bool, list[str]]:
    """Return (passed, list_of_violation_messages)."""
    violations: list[str] = []
    for name in ("ttft_ms", "tpot_ms", "itl_p99_ms"):
        c = getattr(req, name)
        if c is None:
            continue
        val = _metric_value(metrics, name)
        if val is None or math.isnan(val):
            violations.append(f"{name}: metric unavailable")
            continue
        if not _OPS[c.constraint](val, c.value):
            violations.append(f"{name}={val:.3f} violates {c.constraint} {c.value}")
    violations.extend(check_demand_attainment(metrics, req))
    return (len(violations) == 0, violations)


def score(metrics: Metrics, req: Requirements) -> float:
    """Weighted scalar of the objectives (higher = better).

    Each objective is min/max-normalized only relative to itself is impossible
    here (single point), so we use a direction-signed, weight-scaled raw value.
    This gives a usable tie-breaker; Pareto selection (below) is the primary
    multi-objective tool.
    """
    total = 0.0
    for obj in req.objectives:
        val = _metric_value(metrics, obj.metric)
        if val is None or math.isnan(val):
            continue
        signed = val if obj.direction == "max" else -val
        total += obj.weight * signed
    return total


def _objective_vector(metrics: Metrics, req: Requirements) -> Optional[list[float]]:
    """Direction-normalized vector (all 'higher is better'); None if any missing."""
    vec: list[float] = []
    for obj in req.objectives:
        val = _metric_value(metrics, obj.metric)
        if val is None or math.isnan(val):
            return None
        vec.append(val if obj.direction == "max" else -val)
    return vec


def _dominates(a: list[float], b: list[float]) -> bool:
    """a Pareto-dominates b (all >=, at least one >)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(candidates: list[tuple[object, Metrics]], req: Requirements) -> list[object]:
    """Return the non-dominated candidates.

    ``candidates`` is a list of (tag, Metrics). Candidates whose objective vector
    is incomplete are excluded from dominance comparison but still returned
    (they cannot be proven dominated).
    """
    vectors: list[tuple[object, Optional[list[float]]]] = [
        (tag, _objective_vector(m, req)) for tag, m in candidates
    ]
    front: list[object] = []
    for tag, vec in vectors:
        if vec is None:
            front.append(tag)
            continue
        dominated = any(
            other_vec is not None and _dominates(other_vec, vec)
            for other_tag, other_vec in vectors
            if other_tag is not tag
        )
        if not dominated:
            front.append(tag)
    return front
