from planner.objective import check_constraints, pareto_front, score
from planner.spec_schema import Constraint, Objective, Requirements
from planner.types import Metrics


def _m(ttft=100, tpot=20, itl=40, thpt=1000, tpw=None):
    return Metrics(ttft_ms=ttft, tpot_ms=tpot, itl_p99_ms=itl,
                   throughput_toks_s=thpt, toks_per_wh=tpw, num_requests=10)


def test_constraint_pass_and_fail():
    req = Requirements(ttft_ms=Constraint(constraint="<=", value=500),
                       tpot_ms=Constraint(constraint="<=", value=50))
    ok, viol = check_constraints(_m(ttft=100, tpot=20), req)
    assert ok and not viol

    ok, viol = check_constraints(_m(ttft=600, tpot=20), req)
    assert not ok and len(viol) == 1


def test_missing_metric_is_violation():
    req = Requirements(itl_p99_ms=Constraint(constraint="<=", value=50))
    m = _m(itl=float("nan"))
    ok, viol = check_constraints(m, req)
    assert not ok


def test_score_prefers_higher_throughput():
    req = Requirements(objectives=[Objective(metric="throughput", direction="max", weight=1.0)])
    assert score(_m(thpt=2000), req) > score(_m(thpt=1000), req)


def test_pareto_front_dominance():
    req = Requirements(objectives=[
        Objective(metric="throughput", direction="max", weight=1.0),
        Objective(metric="toks_per_wh", direction="max", weight=1.0),
    ])
    # a dominates b (higher on both); c trades off (non-dominated vs a)
    a = ("a", _m(thpt=2000, tpw=100))
    b = ("b", _m(thpt=1000, tpw=50))
    c = ("c", _m(thpt=1500, tpw=200))
    front = set(pareto_front([a, b, c], req))
    assert "b" not in front
    assert "a" in front and "c" in front
