"""Regression: per-model TP options must come from the fidelity ladder.

A 70B request on an A40+A5000 cluster reported bottleneck="memory": the
service derived TP options from the measured-oracle catalog only and fell back
to a hardcoded [1, 2], so Stage-1 only ever tried tp<=2 — where 141 GB of
weights cannot fit — even though A40 tp4/tp8 profiles exist and the A40 70B
validation booted tp4 at 35 GB/GPU on real hardware.
"""
from __future__ import annotations

import pytest

from service.api.routes import (
    _drop_hardware,
    model_tp_catalog,
    preferred_tp_options,
    proxy_toks_per_unit,
)

M8 = "meta-llama/Llama-3.1-8B"
M70 = "meta-llama/Llama-3.1-70B"

CATALOG = {
    "measured": {"A40": [1], "A5000": [1, 2]},
    "upstream": {"A40": [1, 2, 4, 8], "A5000": [1, 2, 4]},
    "legacy": {"A40": [1, 2, 4, 8, 16], "RNGD": [1]},
}


# ---- pure resolution -------------------------------------------------------

def test_prefers_highest_fidelity_source_per_hardware():
    hw_tps, unsupported = preferred_tp_options(CATALOG, {"A40", "A5000", "RNGD"})
    assert hw_tps == {"A40": [1], "A5000": [1, 2], "RNGD": [1]}
    assert unsupported == []


def test_falls_through_ladder_when_no_oracle():
    cat = {"measured": {}, "upstream": {"A40": [1, 4, 8]}, "legacy": {"A40": [4, 8]}}
    hw_tps, unsupported = preferred_tp_options(cat, {"A40"})
    assert hw_tps == {"A40": [1, 4, 8]}          # upstream, not legacy
    assert unsupported == []


def test_hardware_without_any_profile_is_reported_unsupported():
    cat = {"measured": {}, "upstream": {"A40": [1, 4, 8]}, "legacy": {}}
    hw_tps, unsupported = preferred_tp_options(cat, {"A40", "A5000"})
    assert hw_tps == {"A40": [1, 4, 8]} and unsupported == ["A5000"]
    # never invent TPs for hardware that cannot run the model
    assert "A5000" not in hw_tps


def test_drop_hardware_prunes_devices_and_empty_nodes():
    topo = {"nodes": [
        {"id": "n0", "devices": [{"name": "A40", "count": 8, "mem_gb": 48},
                                 {"name": "A5000", "count": 2, "mem_gb": 24}]},
        {"id": "n1", "devices": [{"name": "A5000", "count": 2, "mem_gb": 24}]}],
        "links": []}
    out = _drop_hardware(topo, ["A5000"])
    assert [n["id"] for n in out["nodes"]] == ["n0"]
    assert out["nodes"][0]["devices"] == [{"name": "A40", "count": 8, "mem_gb": 48}]
    assert topo["nodes"][1]["devices"]          # input untouched (deep-copied)


def test_demand_proxy_scales_with_model_size():
    """The Stage-1 proxy constant is calibrated for 8B; a 70B demand would be
    ~9x over-optimistic without scaling."""
    s8, s70 = proxy_toks_per_unit(M8), proxy_toks_per_unit(M70)
    assert s8 == pytest.approx(1000, rel=0.05)   # 8B: unchanged behaviour
    assert s70 is not None and 80 < s70 < 200
    assert s8 / s70 == pytest.approx(70 / 8, rel=0.25)
    assert proxy_toks_per_unit("no/such-model") is None


# ---- the reported bug, end to end through Stage-1 ---------------------------

def test_70b_is_feasible_on_a40_with_ladder_tp_options():
    """Stage-1 must find A40 tp>=4 templates for 70B (not memory-infeasible)."""
    cat = model_tp_catalog(M70)
    if not any(cat.get(s, {}).get("A40") for s in ("upstream", "legacy")):
        pytest.skip("no A40 70B profile in this checkout (submodules/setup.sh)")

    from planner.milp_solver import solve_with_report
    from planner.spec_schema import PlannerSpec

    hw_tps, unsupported = preferred_tp_options(cat, {"A40", "A5000"})
    assert "A5000" in unsupported          # 24 GB cards cannot host 70B here
    assert max(hw_tps["A40"]) >= 4

    spec = PlannerSpec.model_validate({
        "model": {"name": M70, "fp": 16},
        "workload": {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl",
                     "num_req": 10},
        "topology": {"nodes": [
            {"id": "a40-0", "host_base_w": 250,
             "devices": [{"name": "A40", "count": 8, "mem_gb": 48}]}]},
        "requirements": {
            "demand": {"toks_per_s": 300,
                       "proxy_toks_per_unit": proxy_toks_per_unit(M70)},
            "objectives": [{"metric": "power_w", "direction": "min",
                            "weight": 1.0}]},
        "search_space": {"tp_choices": hw_tps["A40"],
                         "hw_tp_choices": {"A40": hw_tps["A40"]}},
        "solver": {"top_k": 4, "time_limit_sec": 15, "pareto_epsilon_steps": 2},
    })
    allocations, report = solve_with_report(spec)
    assert report is None, report            # <- was bottleneck="memory"
    assert allocations
    for a in allocations:
        for inst in a.instances:
            assert inst.tp >= 4, a.signature()   # 70B never fits at tp<4 on 48 GB


def test_70b_with_only_small_gpus_reports_profiles_not_memory():
    """A cluster of 24 GB cards has no 70B profile at all -> the diagnosis is
    'profiles', surfaced before Stage-1 ever runs."""
    cat = model_tp_catalog(M70)
    hw_tps, unsupported = preferred_tp_options(cat, {"A5000"})
    assert hw_tps == {} and unsupported == ["A5000"]


def test_eval_jobs_shrinks_for_large_models():
    """Stage-2 fan-out: 70B candidates need far more memory per simulation."""
    from service.api.routes import eval_jobs
    assert eval_jobs(M8) == 4
    assert eval_jobs(M70) == 2
    assert eval_jobs("no/such-model") == 4       # unknown -> default
