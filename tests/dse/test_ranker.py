"""Unit tests for webapp.dse.core.ranker."""
from types import SimpleNamespace

from webapp.dse.core.ranker import (
    _dominates,
    _normalize_minmax,
    compute_scores,
    filter_slo,
    pareto_frontier,
    rank_candidates,
    top_n,
)
from webapp.dse.core.schemas import (
    Constraints,
    ObjectiveWeights,
    SimulationResult,
)


def _mk(label, ttft, tp, energy=None, tpot=None, itl=None, state="done"):
    m = {"p99_ttft_ms": ttft, "total_token_tp": tp}
    if energy is not None: m["total_energy_wh"] = energy
    if tpot   is not None: m["p99_tpot_ms"]    = tpot
    if itl    is not None: m["p99_itl_ms"]     = itl
    return SimulationResult(candidate_id=label, label=label, state=state, elapsed_s=10, metrics=m)


# ------- helpers -------------------------------------------------------------

def test_dominates_basic():
    # a=[10, 100] (min, max). b=[20, 50]. a dominates b.
    assert _dominates([10, 100], [20, 50], ["min", "max"]) is True
    assert _dominates([20, 50], [10, 100], ["min", "max"]) is False


def test_dominates_equal_not_strict():
    # All equal → no dominance (strict improvement required)
    assert _dominates([5, 5], [5, 5], ["min", "min"]) is False


def test_normalize_minmax_min():
    # Lower is better → smaller value gets 1.0
    out = _normalize_minmax([10, 20, 30], "min")
    assert out == [1.0, 0.5, 0.0]


def test_normalize_minmax_max():
    out = _normalize_minmax([10, 20, 30], "max")
    assert out == [0.0, 0.5, 1.0]


def test_normalize_zero_span():
    out = _normalize_minmax([5, 5, 5], "min")
    assert out == [1.0, 1.0, 1.0]


# ------- SLO filter ----------------------------------------------------------

def test_filter_slo_passes_when_no_constraints():
    rs = [_mk("a", 50, 100, energy=1.0)]
    filter_slo(rs, Constraints())
    assert rs[0].meets_slo is True


def test_filter_slo_rejects_failed_state():
    rs = [_mk("a", 50, 100, state="failed")]
    filter_slo(rs, Constraints(ttft_p99_ms=100))
    assert rs[0].meets_slo is False


def test_filter_slo_ttft_violation():
    rs = [_mk("a", 50, 100), _mk("b", 200, 100)]
    filter_slo(rs, Constraints(ttft_p99_ms=100))
    assert rs[0].meets_slo is True
    assert rs[1].meets_slo is False


def test_filter_slo_throughput_floor():
    rs = [_mk("a", 50, 50), _mk("b", 50, 500)]
    filter_slo(rs, Constraints(throughput_min_tok_s=100))
    assert rs[0].meets_slo is False
    assert rs[1].meets_slo is True


# ------- Pareto --------------------------------------------------------------

def test_pareto_single_dim():
    rs = [_mk("a", 30, 100, energy=2.0),
          _mk("b", 50, 200, energy=1.0),
          _mk("c", 40, 150, energy=1.5)]
    for r in rs: r.meets_slo = True
    keep = pareto_frontier(rs)
    # All three are Pareto optimal (each best in some dim):
    # a best ttft, b best throughput, c best in none → c is dominated
    labels = {rs[i].label for i in keep}
    assert "a" in labels and "b" in labels
    # c (50, 150, 1.5) vs (30, 100, 2.0): c is worse on ttft+energy but better on tp.
    # Actually c (40, 150, 1.5) vs a (30, 100, 2.0): c better in tp+energy, worse in ttft. Both Pareto.


def test_pareto_drops_missing_dim_globally():
    # No candidate has energy → that dim should be dropped, not the candidates
    rs = [_mk("a", 30, 100), _mk("b", 50, 200)]
    for r in rs: r.meets_slo = True
    keep = pareto_frontier(rs)
    assert len(keep) == 2  # both Pareto on (ttft, tp)


def test_pareto_excludes_slo_violators():
    rs = [_mk("a", 30, 100), _mk("b", 50, 200)]
    rs[0].meets_slo = False
    rs[1].meets_slo = True
    keep = pareto_frontier(rs)
    assert keep == [1]


# ------- Scoring -------------------------------------------------------------

def test_compute_scores_weight_zero_throughput():
    """When weight=[0, 0, 1.0, 0] (throughput-only), result with higher tp wins."""
    rs = [_mk("a", 30, 100, energy=2.0, tpot=20),
          _mk("b", 50, 200, energy=1.0, tpot=25)]
    for r in rs: r.meets_slo = True
    w = ObjectiveWeights(ttft=0, tpot=0, throughput=1, power=0)
    compute_scores(rs, w)
    # b has higher tp → score 1.0, a → score 0.0
    assert rs[0].score == 0.0
    assert rs[1].score == 1.0


def test_compute_scores_skips_failed():
    rs = [_mk("a", 30, 100, energy=2.0), _mk("b", 30, 200, energy=1.0, state="failed")]
    rs[0].meets_slo = True; rs[1].meets_slo = False
    compute_scores(rs, ObjectiveWeights())
    assert rs[0].score is not None
    assert rs[1].score is None


# ------- Top-N ---------------------------------------------------------------

def test_top_n_diversity():
    rs = [_mk("a", 30, 100), _mk("b", 40, 200), _mk("c", 50, 50)]
    for r in rs: r.meets_slo = True
    compute_scores(rs, ObjectiveWeights(ttft=0.5, throughput=0.5, tpot=0, power=0))
    # Two candidates have same hw signature → only one returned
    cand_by_label = {
        "a": SimpleNamespace(hw_distribution={"A6000": 1}),
        "b": SimpleNamespace(hw_distribution={"A6000": 1}),  # same as a
        "c": SimpleNamespace(hw_distribution={"H100": 1}),
    }
    picked = top_n(rs, n=3, candidates_by_label=cand_by_label, diversity=True)
    labels = [rs[i].label for i in picked]
    # Should pick at most one of {a, b} (whichever scores higher) + c
    assert "c" in labels
    assert not ("a" in labels and "b" in labels)
    assert len(labels) == 2


def test_top_n_no_diversity():
    rs = [_mk("a", 30, 100), _mk("b", 40, 200)]
    for r in rs: r.meets_slo = True
    compute_scores(rs, ObjectiveWeights(throughput=1, ttft=0, tpot=0, power=0))
    cand_by_label = {
        "a": SimpleNamespace(hw_distribution={"A6000": 1}),
        "b": SimpleNamespace(hw_distribution={"A6000": 1}),
    }
    picked = top_n(rs, n=2, candidates_by_label=cand_by_label, diversity=False)
    assert len(picked) == 2


# ------- Full pipeline ------------------------------------------------------

def test_rank_candidates_pipeline():
    rs = [_mk("a", 30, 100, energy=2.0, tpot=20, itl=22),
          _mk("b", 50, 200, energy=1.0, tpot=25, itl=27),
          _mk("c", 200, 50, energy=5.0, tpot=30, itl=35)]  # SLO violator
    cand_by_label = {
        "a": SimpleNamespace(hw_distribution={"A6000": 1}),
        "b": SimpleNamespace(hw_distribution={"H100":  1}),
        "c": SimpleNamespace(hw_distribution={"RNGD":  1}),
    }
    ranked = rank_candidates(
        rs, Constraints(ttft_p99_ms=100),
        ObjectiveWeights(), top_n_count=3,
        candidates_by_label=cand_by_label,
    )
    # c violates → not Pareto, not Top-N
    assert "c" not in [rs[i].label for i in ranked.pareto_indices]
    assert "c" not in [rs[i].label for i in ranked.top_n_indices]
