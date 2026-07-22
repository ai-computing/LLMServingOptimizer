"""Result Analyzer / Ranker.

Three responsibilities:
  1. SLO filter — flag candidates that violate constraints (don't exclude
     from Pareto, just mark meets_slo=False).
  2. Pareto frontier — N-dimensional dominance check.
  3. Weighted scoring + Top-N — normalize each objective, apply user weights,
     pick highest scores. Diversity: same hw_distribution counts only once.

All inputs/outputs use SimulationResult / RankedResults pydantic models.
"""
from __future__ import annotations

from typing import Any

from .schemas import (
    Constraints,
    ObjectiveWeights,
    RankedResults,
    SimulationResult,
)

# Metric keys we look at, with direction. "min" = lower is better.
# Maps to keys parser.py writes into metrics dict.
_OBJECTIVES = {
    "ttft":       ("p99_ttft_ms",     "min"),
    "tpot":       ("p99_tpot_ms",     "min"),
    "throughput": ("total_token_tp",  "max"),
    "power":      ("total_energy_wh", "min"),   # proxy for "power" objective
    "tokwh":      ("tok_per_wh",      "max"),   # energy efficiency
}


def filter_slo(results: list[SimulationResult], cons: Constraints) -> None:
    """Mark `meets_slo` on each result in-place. Does not remove anything."""
    for r in results:
        if r.state != "done":
            r.meets_slo = False
            continue
        m = r.metrics
        ok = True
        if cons.ttft_p99_ms is not None and (m.get("p99_ttft_ms") is None or m["p99_ttft_ms"] > cons.ttft_p99_ms):
            ok = False
        if ok and cons.tpot_p99_ms is not None and (m.get("p99_tpot_ms") is None or m["p99_tpot_ms"] > cons.tpot_p99_ms):
            ok = False
        if ok and cons.itl_p99_ms is not None and (m.get("p99_itl_ms") is None or m["p99_itl_ms"] > cons.itl_p99_ms):
            ok = False
        if ok and cons.throughput_min_tok_s is not None and (m.get("total_token_tp") is None or m["total_token_tp"] < cons.throughput_min_tok_s):
            ok = False
        if ok and cons.power_max_w is not None and m.get("avg_power_w") is not None and m["avg_power_w"] > cons.power_max_w:
            ok = False
        if ok and cons.energy_max_wh is not None and (m.get("total_energy_wh") is None or m["total_energy_wh"] > cons.energy_max_wh):
            ok = False
        if ok and cons.tokwh_min is not None and (m.get("tok_per_wh") is None or m["tok_per_wh"] < cons.tokwh_min):
            ok = False
        r.meets_slo = ok


def _dominates(a: list[float], b: list[float], directions: list[str]) -> bool:
    """Does point a dominate b in objective space? (assumes both same length)"""
    strictly_better_any = False
    for av, bv, d in zip(a, b, directions):
        if d == "min":
            if av > bv:
                return False
            if av < bv:
                strictly_better_any = True
        else:  # max
            if av < bv:
                return False
            if av > bv:
                strictly_better_any = True
    return strictly_better_any


def pareto_frontier(results: list[SimulationResult],
                    objective_keys: list[tuple[str, str]] | None = None) -> list[int]:
    """Return indices of Pareto-optimal results among `state=done && meets_slo`.

    objective_keys: list of (metric_key, direction) tuples. None → defaults.
    """
    if objective_keys is None:
        objective_keys = [
            ("p99_ttft_ms",     "min"),
            ("total_token_tp",  "max"),
            ("total_energy_wh", "min"),
        ]

    # Eligible candidates (done + meets_slo)
    elig = [i for i, r in enumerate(results)
            if r.state == "done" and r.meets_slo]

    # Drop objective dims that are entirely missing across eligible candidates,
    # rather than rejecting the candidates. This way a single missing metric
    # (e.g. total_energy_wh on a sim with no power block) doesn't empty Pareto.
    active_objectives = []
    for k, d in objective_keys:
        if any(results[i].metrics.get(k) is not None for i in elig):
            active_objectives.append((k, d))
    if not active_objectives:
        return []

    valid: list[tuple[int, list[float]]] = []
    for i in elig:
        pt = []
        ok = True
        for k, _ in active_objectives:
            v = results[i].metrics.get(k)
            if v is None:
                ok = False
                break
            pt.append(float(v))
        if ok:
            valid.append((i, pt))

    directions = [d for _, d in active_objectives]
    keep_indices: list[int] = []
    for i, pt_i in valid:
        dominated = False
        for j, pt_j in valid:
            if i == j:
                continue
            if _dominates(pt_j, pt_i, directions):
                dominated = True
                break
        if not dominated:
            keep_indices.append(i)
    return keep_indices


def _normalize_minmax(values: list[float], direction: str) -> list[float]:
    """Map values to [0, 1] where 1 = best (direction-aware)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [1.0] * len(values)
    if direction == "min":
        return [(hi - v) / span for v in values]
    else:  # max
        return [(v - lo) / span for v in values]


def compute_scores(results: list[SimulationResult],
                   weights: ObjectiveWeights) -> None:
    """Set `score` on each result in-place. SLO-failing results get score=None."""
    # Per-objective normalized arrays
    elig_indices = [i for i, r in enumerate(results)
                    if r.state == "done" and r.meets_slo]

    norm_arrays: dict[str, list[float]] = {}
    for obj_name, (metric_key, direction) in _OBJECTIVES.items():
        raw = []
        for i in elig_indices:
            v = results[i].metrics.get(metric_key)
            raw.append(float(v) if v is not None else None)
        # Skip objective if any candidate missing the metric
        if any(v is None for v in raw):
            norm_arrays[obj_name] = [0.0] * len(elig_indices)
        else:
            norm_arrays[obj_name] = _normalize_minmax(raw, direction)

    weight_map = {
        "ttft":       weights.ttft,
        "tpot":       weights.tpot,
        "throughput": weights.throughput,
        "power":      weights.power,
        "tokwh":      weights.tokwh,
    }

    # Reset scores
    for r in results:
        r.score = None

    for local_idx, global_idx in enumerate(elig_indices):
        score = sum(
            weight_map[obj] * norm_arrays[obj][local_idx]
            for obj in _OBJECTIVES
        )
        results[global_idx].score = round(score, 4)


def _hw_signature(result: SimulationResult, candidates_by_label: dict[str, Any]) -> tuple:
    """Diversity key: stable hw-distribution tuple."""
    c = candidates_by_label.get(result.label)
    if c is None:
        return ()
    return tuple(sorted(c.hw_distribution.items()))


def top_n(results: list[SimulationResult], n: int,
          candidates_by_label: dict[str, Any],
          diversity: bool = True) -> list[int]:
    """Pick best-N indices (by score, desc). Optional diversity: same
    hw_distribution counted only once.
    """
    # Filter to scored candidates
    scored = [(i, r) for i, r in enumerate(results) if r.score is not None]
    scored.sort(key=lambda t: t[1].score, reverse=True)

    if not diversity:
        return [i for i, _ in scored[:n]]

    seen: set = set()
    picked: list[int] = []
    for i, r in scored:
        sig = _hw_signature(r, candidates_by_label)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(i)
        if len(picked) >= n:
            break
    return picked


def rank_candidates(
    results: list[SimulationResult],
    constraints: Constraints,
    weights: ObjectiveWeights,
    top_n_count: int,
    candidates_by_label: dict[str, Any] | None = None,
    diversity: bool = True,
) -> RankedResults:
    """Full pipeline: SLO filter → Pareto → score → Top-N."""
    filter_slo(results, constraints)
    pareto_idx = pareto_frontier(results)
    for idx in pareto_idx:
        results[idx].on_pareto = True
    compute_scores(results, weights)

    if candidates_by_label is None:
        candidates_by_label = {}

    top_idx = top_n(results, top_n_count, candidates_by_label, diversity=diversity)

    return RankedResults(
        all_results=results,
        pareto_indices=pareto_idx,
        top_n_indices=top_idx,
        weights_used=weights,
    )
