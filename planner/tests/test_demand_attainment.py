"""Stage-2 must notice when a candidate cannot sustain the offered load.

Stage-1 only compares a demand estimate against a capacity ceiling; nothing
re-checked it once the trace had actually been played. Neither obvious signal
works alone:

* ``throughput_toks_s`` is not capacity. The trace fixes the token count, so
  tokens/span reports the ARRIVAL rate however fast the server is -- the A5000
  14B AWQ run read 266 toks/s against a 496 toks/s demand while perfectly
  healthy, purely because requests arrived that slowly.
* a peak-generation shortfall is not proof either: a 50-request evaluation of a
  5 req/s demand covers ~10 s of arrivals, so concurrency never reaches where
  the measured curve peaks.

So backlog (queue wait) fails a candidate, and a peak-rate shortfall is reported
as an advisory instead.
"""
from __future__ import annotations

import pytest

from planner.objective import (
    check_constraints,
    check_demand_attainment,
    demand_shortfall,
)
from planner.spec_schema import Requirements
from planner.types import Metrics


DEMAND = 496.0                 # the real chat-preset generation demand


def _req(**demand_kw) -> Requirements:
    slo = demand_kw.pop("slo", {})
    return Requirements.model_validate(
        {"demand": {"toks_per_s": DEMAND, **demand_kw}, **slo})


def _metrics(peak=None, queue_p95=None, queue_max=None, thr=266.0) -> Metrics:
    return Metrics(ttft_ms=280.0, tpot_ms=17.0, itl_p99_ms=18.0,
                   throughput_toks_s=thr, peak_gen_toks_s=peak,
                   queue_p95_ms=queue_p95, queue_max_ms=queue_max)


def test_healthy_candidate_passes_though_throughput_is_below_demand():
    """The real A5000 AWQ recommendation: throughput_toks_s 266 against a 496
    toks/s demand, but it demonstrably generated at 536 toks/s and never queued.
    A naive throughput >= demand test would have rejected it."""
    m = _metrics(peak=536.0, queue_p95=0.0, queue_max=0.0, thr=266.0)
    assert check_demand_attainment(m, _req()) == []
    passed, violations = check_constraints(m, _req())
    assert passed and violations == []


def test_short_peak_without_backlog_advises_but_does_not_fail():
    """The 5 req/s candidates peaked at 914-978 toks/s against a 1154 toks/s
    demand with zero queueing and TPOT 14-17ms — no saturation, just a
    50-request trace too short to reach the concurrency where the measured curve
    peaks. Failing that would recreate the false 달성 불가."""
    m = _metrics(peak=914.0, queue_p95=0.0)
    req = Requirements.model_validate({"demand": {"toks_per_s": 1154.0}})
    assert check_demand_attainment(m, req) == []          # not a violation
    passed, _ = check_constraints(m, req)
    assert passed
    note = demand_shortfall(m, req)
    assert note and "914" in note and "1154" in note and "num_req_eval" in note


def test_no_advisory_when_the_demanded_rate_was_reached():
    assert demand_shortfall(_metrics(peak=536.0), _req()) is None
    assert demand_shortfall(_metrics(peak=DEMAND * 0.93), _req()) is None  # 10% tol
    assert demand_shortfall(_metrics(peak=DEMAND * 0.93),
                            _req(attainment_tolerance=0.05)) is not None
    assert demand_shortfall(_metrics(peak=1.0),
                            _req(attainment_tolerance=1.0)) is None      # disabled


def test_backlog_is_conclusive_and_fails_the_candidate():
    """Requests waiting for a slot means the admission cap bound: unambiguous."""
    m = _metrics(peak=DEMAND * 1.1, queue_p95=4200.0, queue_max=9000.0)
    out = check_demand_attainment(m, _req())
    assert len(out) == 1 and "queue wait" in out[0]
    assert "4200ms" in out[0] and "9000ms" in out[0]
    passed, violations = check_constraints(m, _req())
    assert not passed and violations == out
    assert check_demand_attainment(m, _req(max_queue_ms=0)) == []       # disabled


def test_missing_demand_or_metrics_is_not_a_violation():
    """A backend that reports neither signal must not fail everything."""
    assert check_demand_attainment(_metrics(), _req()) == []
    no_demand = Requirements.model_validate({})
    assert check_demand_attainment(_metrics(peak=1.0, queue_p95=9999.0),
                                   no_demand) == []
    assert demand_shortfall(_metrics(peak=1.0), no_demand) is None


def test_demand_violation_coexists_with_slo_violations():
    m = _metrics(peak=100.0, queue_p95=5000.0)
    req = _req(slo={"ttft_ms": {"constraint": "<=", "value": 100.0}})
    passed, violations = check_constraints(m, req)
    assert not passed
    assert any("ttft_ms" in v for v in violations)
    assert any("demand not sustained" in v for v in violations)


def test_defaults():
    d = _req().demand
    assert d.attainment_tolerance == 0.10 and d.max_queue_ms == 1000.0


# ---- the metric itself ------------------------------------------------------

def _trace(tmp_path, n, out_toks, tpot_ns, end_s, arrival_s=0):
    import csv
    ns = 1_000_000_000
    p = tmp_path / "out.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance id", "request id", "model", "input", "output",
                    "arrival", "end_time", "latency", "queuing_delay",
                    "TTFT", "TPOT", "ITL"])
        for i in range(n):
            w.writerow([0, i, "m", 10, out_toks, arrival_s * ns, end_s * ns,
                        (end_s - arrival_s) * ns, 0, ns // 10, tpot_ns, "[]"])
    return p


def test_peak_generation_rate_sees_what_throughput_cannot(tmp_path):
    """Ten requests each generating 500 tokens at 10 ms/token decode the same
    5 s stretch and finish at t=30s: the cluster really produced 1000 toks/s,
    while tokens/span (throughput_toks_s) reports 167 because the trace spans
    30 s. That gap is the whole reason this metric exists."""
    from planner.sim_evaluator import parse_metrics_csv

    m = parse_metrics_csv(_trace(tmp_path, n=10, out_toks=500,
                                 tpot_ns=10_000_000, end_s=30))
    assert m.peak_gen_toks_s == pytest.approx(1000, rel=0.05)
    assert m.throughput_toks_s == pytest.approx(5000 / 30, rel=0.05)
    assert m.queue_p95_ms == 0.0


def test_peak_generation_rate_is_a_sustained_average_not_a_spike(tmp_path):
    """The window is 5 s on purpose: a burst shorter than that is averaged down,
    so a momentary spike cannot be mistaken for capacity."""
    from planner.sim_evaluator import _GEN_WINDOW_S, parse_metrics_csv

    m = parse_metrics_csv(_trace(tmp_path, n=10, out_toks=100,
                                 tpot_ns=10_000_000, end_s=10))
    # 1000 tokens generated inside a single second -> 1000/5 over the window
    assert m.peak_gen_toks_s == pytest.approx(1000 / _GEN_WINDOW_S, rel=0.05)
