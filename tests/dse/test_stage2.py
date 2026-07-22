"""Unit tests for webapp.dse.core.stage2_predictor.

Uses real profile data from llm_profile/perf_models/A6000/meta-llama/Llama-3.1-8B/tp1/
so results reflect actual measured latencies.
"""
import pytest
from pathlib import Path

from webapp.dse.core.stage2_predictor import (
    ProfileNotFoundError,
    _load_attn_decode_db,
    _load_attn_prefill_db,
    _load_layers_db,
    _lookup_attn_decode,
    _lookup_attn_prefill,
    _lookup_layer,
    predict_ttft_tpot,
    sanity_check_vs_roofline,
)

# ---------------------------------------------------------------------------
# Fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "backends" / "legacy" / "llm_profile" / "perf_models"
A6000_8B_TP1 = PROFILE_ROOT / "A6000" / "meta-llama" / "Llama-3.1-8B" / "tp1"


@pytest.fixture(scope="module")
def layers_db():
    return _load_layers_db(A6000_8B_TP1 / "layers.csv")


@pytest.fixture(scope="module")
def prefill_db():
    return _load_attn_prefill_db(A6000_8B_TP1 / "predictions" / "attn_prefill_predictions.csv")


@pytest.fixture(scope="module")
def decode_db():
    return _load_attn_decode_db(A6000_8B_TP1 / "predictions" / "attn_decode_predictions.csv")


# ---------------------------------------------------------------------------
# CSV loader tests

def test_load_layers_db_keys(layers_db):
    # Should contain basic Llama layer names at input=1
    assert ("embedding", 1, 0, 1) in layers_db
    assert ("gate_proj", 1, 0, 1) in layers_db
    assert ("lm_head", 1, 0, 1) in layers_db


def test_load_layers_db_positive_latency(layers_db):
    for latency in layers_db.values():
        assert latency > 0


def test_load_attn_prefill_db_keys(prefill_db):
    # (kv_cache_size=0, prefill_chunk_size=32) should be the smallest entry
    assert any(kv == 0 for (kv, _) in prefill_db)


def test_load_attn_decode_db_keys(decode_db):
    assert any(bs == 1 for (bs, _) in decode_db)


def test_load_layers_db_missing_raises():
    with pytest.raises(ProfileNotFoundError):
        _load_layers_db(PROFILE_ROOT / "NonExistentHW" / "model" / "tp1" / "layers.csv")


def test_load_attn_prefill_db_missing_raises():
    with pytest.raises(ProfileNotFoundError):
        _load_attn_prefill_db(PROFILE_ROOT / "NonExistentHW" / "model" / "tp1" /
                               "predictions" / "attn_prefill_predictions.csv")


# ---------------------------------------------------------------------------
# Lookup tests

def test_lookup_layer_exact(layers_db):
    # Exact key must return the stored value
    lat = _lookup_layer(layers_db, "embedding", 1, 0, 1)
    assert lat == layers_db[("embedding", 1, 0, 1)]


def test_lookup_layer_nearest_neighbour(layers_db):
    # Query a token count that may not be in the table; result must be positive
    lat = _lookup_layer(layers_db, "gate_proj", 999, 0, 1)
    assert lat > 0


def test_lookup_layer_missing_layer_raises(layers_db):
    with pytest.raises(ProfileNotFoundError):
        _lookup_layer(layers_db, "nonexistent_layer", 1, 0, 1)


def test_lookup_attn_prefill_exact(prefill_db):
    # First available key must return its own value
    kv, chunk = next(iter(prefill_db))
    assert _lookup_attn_prefill(prefill_db, kv, chunk) == prefill_db[(kv, chunk)]


def test_lookup_attn_prefill_nearest(prefill_db):
    lat = _lookup_attn_prefill(prefill_db, kv_cache_size=0, chunk_size=500)
    assert lat > 0


def test_lookup_attn_decode_exact(decode_db):
    bs, kv = next(iter(decode_db))
    assert _lookup_attn_decode(decode_db, bs, kv) == decode_db[(bs, kv)]


def test_lookup_attn_decode_nearest(decode_db):
    lat = _lookup_attn_decode(decode_db, batch_size=1, kv_cache_size=1500)
    assert lat > 0


# ---------------------------------------------------------------------------
# predict_ttft_tpot — real profile

def test_predict_returns_positive_ms():
    result = predict_ttft_tpot(
        hardware="A6000",
        model_name="meta-llama/Llama-3.1-8B",
        tp=1,
        num_layers=32,
        avg_prompt_len=512,
        max_seq_len=2048,
        decode_batch_size=1,
        profile_root=PROFILE_ROOT,
    )
    assert result["ttft_pred_ms"] > 0
    assert result["tpot_pred_ms"] > 0


def test_predict_ttft_larger_than_tpot_single_request():
    # TTFT processes avg_prompt_len tokens; TPOT processes 1 token.
    # For a 512-token prompt, TTFT must be much larger than TPOT.
    result = predict_ttft_tpot(
        hardware="A6000",
        model_name="meta-llama/Llama-3.1-8B",
        tp=1,
        num_layers=32,
        avg_prompt_len=512,
        max_seq_len=2048,
        decode_batch_size=1,
        profile_root=PROFILE_ROOT,
    )
    assert result["ttft_pred_ms"] > result["tpot_pred_ms"]


def test_predict_breakdown_sums_to_total():
    result = predict_ttft_tpot(
        hardware="A6000",
        model_name="meta-llama/Llama-3.1-8B",
        tp=1,
        num_layers=32,
        avg_prompt_len=128,
        max_seq_len=2048,
        decode_batch_size=1,
        profile_root=PROFILE_ROOT,
    )
    bd = result["breakdown"]
    ttft_ns_from_breakdown = (
        bd["ttft_emb_ns"] + bd["ttft_block_dense_ns"] +
        bd["ttft_attn_ns"] + bd["ttft_fln_lmh_ns"]
    )
    assert abs(ttft_ns_from_breakdown / 1_000_000 - result["ttft_pred_ms"]) < 1e-6


def test_predict_longer_prompt_increases_ttft():
    r_short = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                                 avg_prompt_len=64, max_seq_len=2048,
                                 profile_root=PROFILE_ROOT)
    r_long  = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                                 avg_prompt_len=512, max_seq_len=2048,
                                 profile_root=PROFILE_ROOT)
    assert r_long["ttft_pred_ms"] > r_short["ttft_pred_ms"]


def test_predict_larger_batch_returns_positive():
    # TPOT is not necessarily monotone with batch size: larger batches amortise
    # memory BW across more requests so per-token latency can decrease.
    # Just verify both return positive values without error.
    r_b1 = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                               avg_prompt_len=128, max_seq_len=2048,
                               decode_batch_size=1, profile_root=PROFILE_ROOT)
    r_b8 = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                               avg_prompt_len=128, max_seq_len=2048,
                               decode_batch_size=8, profile_root=PROFILE_ROOT)
    assert r_b1["tpot_pred_ms"] > 0
    assert r_b8["tpot_pred_ms"] > 0


def test_predict_tp4_vs_tp1():
    # tp=4 should be available for A6000/Llama-3.1-8B
    r_tp1 = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                                avg_prompt_len=128, max_seq_len=2048,
                                profile_root=PROFILE_ROOT)
    r_tp4 = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 4, 32,
                                avg_prompt_len=128, max_seq_len=2048,
                                profile_root=PROFILE_ROOT)
    # Both must be positive; tp4 latency may be lower due to tensor parallelism
    assert r_tp1["ttft_pred_ms"] > 0
    assert r_tp4["ttft_pred_ms"] > 0


def test_predict_missing_profile_raises():
    with pytest.raises(ProfileNotFoundError):
        predict_ttft_tpot(
            hardware="NonExistentHW",
            model_name="meta-llama/Llama-3.1-8B",
            tp=1,
            num_layers=32,
            avg_prompt_len=128,
            max_seq_len=2048,
            profile_root=PROFILE_ROOT,
        )


def test_predict_profile_source_contains_path():
    result = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                                avg_prompt_len=128, max_seq_len=2048,
                                profile_root=PROFILE_ROOT)
    assert "A6000" in result["profile_source"]
    assert "tp1" in result["profile_source"]


# ---------------------------------------------------------------------------
# sanity_check_vs_roofline

def test_sanity_no_warnings_when_pred_above_lb():
    warnings = sanity_check_vs_roofline(
        ttft_pred_ms=50.0, tpot_pred_ms=20.0,
        roofline_ttft_lb_ms=10.0, roofline_tpot_lb_ms=5.0,
    )
    assert warnings == []


def test_sanity_warns_when_tpot_below_lb():
    warnings = sanity_check_vs_roofline(
        ttft_pred_ms=50.0, tpot_pred_ms=1.0,
        roofline_ttft_lb_ms=10.0, roofline_tpot_lb_ms=20.0,
    )
    assert any("TPOT" in w for w in warnings)


def test_sanity_warns_when_ttft_below_lb():
    warnings = sanity_check_vs_roofline(
        ttft_pred_ms=5.0, tpot_pred_ms=20.0,
        roofline_ttft_lb_ms=100.0, roofline_tpot_lb_ms=5.0,
    )
    assert any("TTFT" in w for w in warnings)


def test_sanity_skips_none_lb():
    warnings = sanity_check_vs_roofline(
        ttft_pred_ms=1.0, tpot_pred_ms=1.0,
        roofline_ttft_lb_ms=None, roofline_tpot_lb_ms=None,
    )
    assert warnings == []


def test_sanity_real_prediction_passes_roofline():
    # A6000/8B: weight=16 GB, bw=768 GB/s → roofline TPOT lb ≈ 20.8 ms
    # Real TPOT from CSV should be ≥ 20.8 ms (it includes all layers, not just memory)
    result = predict_ttft_tpot("A6000", "meta-llama/Llama-3.1-8B", 1, 32,
                                avg_prompt_len=128, max_seq_len=2048,
                                profile_root=PROFILE_ROOT)
    roofline_tpot_lb_ms = (16.0 / 768.0) * 1000  # ≈ 20.8 ms
    warnings = sanity_check_vs_roofline(
        ttft_pred_ms=result["ttft_pred_ms"],
        tpot_pred_ms=result["tpot_pred_ms"],
        roofline_ttft_lb_ms=None,
        roofline_tpot_lb_ms=roofline_tpot_lb_ms,
    )
    assert warnings == [], f"Unexpected warnings: {warnings}"
