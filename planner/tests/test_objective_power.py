"""M2 tests: power_w metric in constraint checking / scoring — among SLO
passers the lower-power candidate ranks first; SLO violators lose regardless
of power."""
from __future__ import annotations

from planner.objective import check_constraints, score
from planner.spec_schema import Requirements
from planner.types import Metrics


def _metrics(ttft=100.0, power=500.0):
    return Metrics(ttft_ms=ttft, tpot_ms=20.0, itl_p99_ms=30.0,
                   throughput_toks_s=1000.0, power_w=power, num_requests=10)


_REQ = Requirements.model_validate({
    "ttft_ms": {"constraint": "<=", "value": 500},
    "demand": {"toks_per_s": 100},
    "objectives": [{"metric": "power_w", "direction": "min", "weight": 1.0}],
})


def test_lower_power_wins_among_slo_passers():
    low = _metrics(ttft=100, power=500)
    high = _metrics(ttft=100, power=800)
    assert check_constraints(low, _REQ) == (True, [])
    assert check_constraints(high, _REQ) == (True, [])
    assert score(low, _REQ) > score(high, _REQ)  # min direction: -power


def test_slo_violator_excluded_regardless_of_power():
    violator = _metrics(ttft=9999, power=1.0)  # tiny power, blown TTFT
    passer = _metrics(ttft=100, power=800.0)
    ok, violations = check_constraints(violator, _REQ)
    assert not ok and violations
    assert check_constraints(passer, _REQ)[0]
    # orchestrator ranks only passers, so the violator never competes;
    # its score being higher must not matter
    assert score(violator, _REQ) > score(passer, _REQ)


def test_missing_power_metric_not_counted_as_zero():
    m = _metrics()
    m.power_w = None
    assert score(m, _REQ) == 0.0  # excluded from scoring, not treated as 0 W
