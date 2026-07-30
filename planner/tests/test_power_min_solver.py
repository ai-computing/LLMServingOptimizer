"""M2 tests: power-min Stage-1 mode — manual optimum, host-power effect,
demand infeasibility reporting, headroom sweep, and mode regression.

The toy inventory exploits the shipped constants directly:
H100 rel_throughput 6.0 / 700 W, A5000 rel_throughput 0.8 / 230 W.
With demand.proxy_toks_per_unit = 1.0, demand values are in proxy units.
"""
from __future__ import annotations

import pytest

from planner.milp_solver import solve_with_report
from planner.spec_schema import PlannerSpec

MODEL = {"name": "meta-llama/Llama-3.1-8B", "fp": 16}
WORKLOAD = {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl", "num_req": 10}


def _spec(nodes, demand_toks_s, steps=1, top_k=8, host_base_w=None, tp=(1,)):
    for nd in nodes:
        if host_base_w is not None:
            nd["host_base_w"] = host_base_w
    return PlannerSpec.model_validate({
        "model": MODEL,
        "workload": WORKLOAD,
        "topology": {"nodes": nodes, "links": []},
        "requirements": {
            "demand": {"toks_per_s": demand_toks_s, "proxy_toks_per_unit": 1.0},
            "objectives": [{"metric": "power_w", "direction": "min", "weight": 1.0}],
        },
        "search_space": {"tp_choices": list(tp)},
        "solver": {"top_k": top_k, "time_limit_sec": 10, "pareto_epsilon_steps": steps},
    })


def _hw_counts(alloc):
    out: dict[str, int] = {}
    for inst in alloc.instances:
        out[inst.hardware] = out.get(inst.hardware, 0) + inst.npu_num
    return out


def test_manual_optimum_prefers_two_a5000_over_one_h100():
    """demand 1.6 proxy units: A5000x2 = 1.6 thr @ 460 W beats H100x1 = 6.0 thr
    @ 700 W. A power-min solver that picks the 'fastest' device is wrong."""
    nodes = [{"id": "node0", "devices": [
        {"name": "H100", "count": 1, "mem_gb": 80},
        {"name": "A5000", "count": 4, "mem_gb": 24},
    ]}]
    allocations, report = solve_with_report(_spec(nodes, demand_toks_s=1.6))
    assert report is None and allocations
    best = allocations[0]
    assert _hw_counts(best) == {"A5000": 2}
    assert best.meta["device_power_w"] == 460
    assert best.meta["mode"] == "power_min"


def test_host_power_flips_optimum_to_h100():
    """Same demand, but the A5000s sit one per host and every host costs 500 W:
    A5000x2 = 460 + 2x500 = 1460 W vs H100x1 = 700 + 500 = 1200 W -> H100 wins.
    Proves the used[host] activation variable works."""
    nodes = [
        {"id": "node0", "devices": [{"name": "H100", "count": 1, "mem_gb": 80}]},
        {"id": "node1", "devices": [{"name": "A5000", "count": 1, "mem_gb": 24}]},
        {"id": "node2", "devices": [{"name": "A5000", "count": 1, "mem_gb": 24}]},
    ]
    allocations, report = solve_with_report(
        _spec(nodes, demand_toks_s=1.6, host_base_w=500))
    assert report is None and allocations
    best = allocations[0]
    assert _hw_counts(best) == {"H100": 1}
    assert best.meta["power_proxy_w"] == 700 + 500


def test_demand_beyond_capacity_yields_infeasible_report():
    nodes = [{"id": "node0", "devices": [
        {"name": "H100", "count": 1, "mem_gb": 80},
        {"name": "A5000", "count": 4, "mem_gb": 24},
    ]}]
    # capacity = 6.0 + 4*0.8 = 9.2 proxy units << demand 100
    allocations, report = solve_with_report(_spec(nodes, demand_toks_s=100.0))
    assert allocations == []
    assert report is not None
    assert report.bottleneck == "demand"
    assert report.max_achievable_toks_s == pytest.approx(9.2, rel=1e-6)
    assert report.suggestions


def test_headroom_sweep_thr_monotonically_nondecreasing():
    """steps=3 -> floors demand*(1.0, 1.1, 1.2); recorded candidates' throughput
    proxy must not decrease along the sweep."""
    nodes = [{"id": "node0", "devices": [{"name": "A5000", "count": 4, "mem_gb": 24}]}]
    allocations, report = solve_with_report(_spec(nodes, demand_toks_s=1.6, steps=3))
    assert report is None and allocations
    thrs = [a.meta["thr_proxy_units"] for a in allocations]
    assert all(b >= a for a, b in zip(thrs, thrs[1:]))
    assert all(t >= 1.6 for t in thrs)
    # 1.76-floor forces a third A5000 -> a genuinely different candidate exists
    assert len(allocations) >= 2


def test_max_throughput_mode_unaffected(spec):
    """Regression: specs without power_w still take the epsilon-sweep path and
    return (allocations, None)."""
    allocations, report = solve_with_report(spec)
    assert report is None
    assert allocations
    assert all(a.meta.get("mode") != "power_min" for a in allocations)


def test_power_w_objective_requires_demand():
    with pytest.raises(ValueError, match="demand"):
        PlannerSpec.model_validate({
            "model": MODEL, "workload": WORKLOAD,
            "topology": {"nodes": [{"id": "n0", "devices": [
                {"name": "A5000", "count": 1, "mem_gb": 24}]}]},
            "requirements": {"objectives": [
                {"metric": "power_w", "direction": "min", "weight": 1.0}]},
        })


def test_power_w_objective_rejects_max_direction():
    with pytest.raises(ValueError, match="min"):
        PlannerSpec.model_validate({
            "model": MODEL, "workload": WORKLOAD,
            "topology": {"nodes": [{"id": "n0", "devices": [
                {"name": "A5000", "count": 1, "mem_gb": 24}]}]},
            "requirements": {
                "demand": {"toks_per_s": 100},
                "objectives": [{"metric": "power_w", "direction": "max", "weight": 1.0}],
            },
        })


def test_hw_tp_choices_restricts_templates():
    """A40 restricted to tp1 while A5000 may use tp1/tp2: no candidate may
    contain an A40 tp2 instance (found live: Stage-1 proposed A40 tp2 with
    only a tp1 measured oracle -> unevaluable candidate)."""
    nodes = [{"id": "node0", "devices": [
        {"name": "A40", "count": 4, "mem_gb": 48},
        {"name": "A5000", "count": 4, "mem_gb": 24},
    ]}]
    s = _spec(nodes, demand_toks_s=2.0, steps=3, tp=(1, 2))
    s.search_space.hw_tp_choices = {"A40": [1], "A5000": [1, 2]}
    allocations, report = solve_with_report(s)
    assert report is None and allocations
    for a in allocations:
        for inst in a.instances:
            if inst.hardware == "A40":
                assert inst.tp == 1, a.signature()


def test_diversity_candidates_cover_each_combo():
    """The SLO-blind sweep would only ever propose the cheap hardware; the
    per-combo diversity pass must still surface an H100-only candidate so
    Stage-2 can pick it when it is the only SLO-passing option."""
    nodes = [{"id": "node0", "devices": [
        {"name": "H100", "count": 2, "mem_gb": 80},
        {"name": "A5000", "count": 8, "mem_gb": 24},
    ]}]
    allocations, report = solve_with_report(_spec(nodes, demand_toks_s=1.6, top_k=2))
    assert report is None
    hw_sets = [{i.hardware for i in a.instances} for a in allocations]
    assert {"A5000"} in hw_sets     # global min-power pick
    assert {"H100"} in hw_sets      # diversity pick despite higher power
    div = [a for a in allocations if a.meta.get("diversity_combo")]
    assert all(a.meta["thr_proxy_units"] >= 1.6 for a in div)


def test_unresolved_req_per_s_demand_raises():
    s = _spec([{"id": "n0", "devices": [{"name": "A5000", "count": 1, "mem_gb": 24}]}],
              demand_toks_s=1.0)
    s.requirements.demand.toks_per_s = None
    s.requirements.demand.req_per_s = 5.0
    s.requirements.demand.preset = "chat"
    with pytest.raises(ValueError, match="toks_per_s"):
        solve_with_report(s)
