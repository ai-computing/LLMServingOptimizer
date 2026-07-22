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
    }.get(name)


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
