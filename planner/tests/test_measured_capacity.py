"""Stage-1 must filter with measured capacity when we have it.

A Qwen-14B AWQ request on 2x A5000 came back "달성 불가 - demand": the proxy
put the ceiling at 867 toks/s while the measured oracles show 1100 toks/s from
a single card. The proxy scales token rate by parameter count alone, so it gives
quantization no credit and assumes TP scales linearly (measured: 1100 at tp1 ->
1296 at tp2, not 2200).
"""
from __future__ import annotations

import pytest

from planner.milp_solver import _rel_throughput, _PROXY_TOKS_PER_UNIT, solve_with_report
from planner.spec_schema import PlannerSpec

UNIT = 582.0            # proxy_toks_per_unit for a ~13.75B checkpoint
MEASURED = {"A5000": {1: 1100.0, 2: 1296.0}}


def _spec(demand_toks_s: float, capacity=None, tps=(1, 2)) -> PlannerSpec:
    search: dict = {"tp_choices": list(tps), "hw_tp_choices": {"A5000": list(tps)}}
    if capacity is not None:
        search["hw_tp_capacity"] = capacity
    return PlannerSpec.model_validate({
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl",
                     "num_req": 10},
        "topology": {"nodes": [{"id": "n0", "host_base_w": 200, "devices": [
            {"name": "A5000", "count": 2, "mem_gb": 24}]}]},
        "requirements": {
            "demand": {"toks_per_s": demand_toks_s, "proxy_toks_per_unit": UNIT},
            "objectives": [{"metric": "power_w", "direction": "min", "weight": 1.0}]},
        "search_space": search,
        "solver": {"top_k": 3, "time_limit_sec": 15, "pareto_epsilon_steps": 2},
    })


# ---- the unitless conversion ------------------------------------------------

def test_measured_capacity_reproduces_the_measured_rate():
    """rel_throughput * tp * unit is what every downstream expression uses, so
    the conversion has to round-trip the measured toks/s exactly."""
    for tp, want in ((1, 1100.0), (2, 1296.0)):
        rel = _rel_throughput("A5000", tp, MEASURED, UNIT)
        assert rel * tp * UNIT == pytest.approx(want)


def test_falls_back_to_the_hardware_constant_without_a_measurement():
    rel = _rel_throughput("A5000", 4, MEASURED, UNIT)          # tp4 not measured
    assert rel == pytest.approx(_rel_throughput("A5000", 4, {}, UNIT))
    assert _rel_throughput("no-such-hw", 1, {}, UNIT) == 1.0


def test_measured_capacity_is_not_assumed_linear_in_tp():
    """The proxy's tp2 = 2x tp1 is exactly what the measurement contradicts."""
    one = _rel_throughput("A5000", 1, MEASURED, UNIT) * 1 * UNIT
    two = _rel_throughput("A5000", 2, MEASURED, UNIT) * 2 * UNIT
    assert two / one == pytest.approx(1296 / 1100, rel=1e-6)
    assert two < 2 * one


# ---- the reported bug -------------------------------------------------------

def test_demand_above_the_proxy_ceiling_is_feasible_with_measured_capacity():
    """~990 toks/s: over the proxy's ceiling for two A5000s, well under what one
    card was measured doing."""
    demand = 990.0
    allocs, report = solve_with_report(_spec(demand))
    assert report is not None and report.bottleneck == "demand"   # the bug
    assert report.max_achievable_toks_s < demand

    allocs, report = solve_with_report(_spec(demand, capacity=MEASURED))
    assert report is None, report                                  # fixed
    assert allocs
    # one card is enough at 1100 toks/s measured; no need to burn the second
    assert min(sum(i.tp for i in a.instances) for a in allocs) == 1


def test_measured_capacity_still_refuses_impossible_demand():
    """The filter must stay a filter: 5000 toks/s is beyond both cards."""
    allocs, report = solve_with_report(_spec(5000.0, capacity=MEASURED))
    assert report is not None and report.bottleneck == "demand"
    assert report.max_achievable_toks_s == pytest.approx(2 * 1100.0, rel=0.02)
