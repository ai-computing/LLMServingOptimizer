"""Unit tests for webapp.dse.core.stage1_filters.

Each filter has at least one passing and one failing case.
Expected values are derived from 03_catalog.yaml + known model parameters.
"""
import pytest

from webapp.dse.core.generator import load_metadata
from webapp.dse.core.stage1_filters import (
    filter_communication,
    filter_divisibility,
    filter_memory,
    filter_power,
    filter_roofline_decode,
    filter_roofline_prefill,
    filter_tokwh_roofline,
)


@pytest.fixture(scope="module")
def metadata():
    return load_metadata()


@pytest.fixture(scope="module")
def hw(metadata):
    return metadata["hardware"]


@pytest.fixture(scope="module")
def model_8b(metadata):
    # Llama-3.1-8B: params=8.03B, layers=32, hidden=4096, attn_heads=32, kv_heads=8
    return metadata["models"]["meta-llama/Llama-3.1-8B"]


@pytest.fixture(scope="module")
def model_70b(metadata):
    # Llama-3.1-70B: params=70.6B, layers=80, hidden=8192, attn_heads=64, kv_heads=8
    return metadata["models"]["meta-llama/Llama-3.1-70B"]


# ── filter_memory ─────────────────────────────────────────────────────────────

def test_memory_pass_8b_single_a6000(hw, model_8b):
    # weight=16 GB, KV(seq=2048)≈0.13 GB, act≈small → total << 0.85×40=34 GB
    assert filter_memory({"A6000": 1}, {"tp": 1, "pp": 1}, model_8b, hw, fp=16, max_seq_len=2048) is None


def test_memory_fail_70b_single_h100_no_tp(hw, model_70b):
    # weight=141 GB, tp=1, pp=1 → shard=141 GB > 0.85×80=68 GB
    result = filter_memory({"H100": 1}, {"tp": 1, "pp": 1}, model_70b, hw, fp=16)
    assert result is not None and "memory" in result


def test_memory_pass_70b_tp4_h100(hw, model_70b):
    # shard=141/4=35.25 GB, KV≈1.3 GB, act≈small → total ≈ 36.6 GB < 0.85×80=68 GB
    assert filter_memory({"H100": 4}, {"tp": 4, "pp": 1}, model_70b, hw, fp=16, max_seq_len=2048) is None


def test_memory_fail_kv_dominates_long_context(hw, model_70b):
    # 70B tp=4: shard=35.25 GB.  KV(128k): 2×80×8×128×2×131072/(4×1e9)≈10.7 GB
    # act(128k): 131072×8192×4×2/(4×1e9)≈2.1 GB.  total≈48 GB < 68 GB → still passes
    # Use tp=1 where weight alone causes failure regardless
    result = filter_memory({"H100": 1}, {"tp": 1, "pp": 1}, model_70b, hw, fp=16, max_seq_len=131072)
    assert result is not None


def test_memory_zero_tp_pp_rejected(hw, model_8b):
    result = filter_memory({"A6000": 1}, {"tp": 0, "pp": 1}, model_8b, hw, fp=16)
    assert result is not None and "tp×pp" in result


# ── filter_divisibility ────────────────────────────────────────────────────────

def test_divisibility_pass_70b_tp4_pp4(model_70b):
    # attn_heads=64 (64%4=0), kv_heads=8 (8%4=0), layers=80 (80%4=0)
    assert filter_divisibility({"tp": 4, "pp": 4}, model_70b) is None


def test_divisibility_pass_tp1_pp1(model_70b):
    # No parallelism → all checks trivially pass
    assert filter_divisibility({"tp": 1, "pp": 1}, model_70b) is None


def test_divisibility_fail_kv_heads_gqa_tp3(model_70b):
    # kv_heads=8, tp=3 → 8%3=2 ≠ 0 (GQA violation)
    result = filter_divisibility({"tp": 3, "pp": 1}, model_70b)
    assert result is not None and "divisibility" in result


def test_divisibility_fail_attn_heads_tp3(model_8b):
    # attn_heads=32, tp=3 → 32%3 ≠ 0
    result = filter_divisibility({"tp": 3, "pp": 1}, model_8b)
    assert result is not None and "divisibility" in result


def test_divisibility_fail_layers_pp3(model_70b):
    # layers=80, pp=3 → 80%3 ≠ 0
    result = filter_divisibility({"tp": 1, "pp": 3}, model_70b)
    assert result is not None and "divisibility" in result


def test_divisibility_pass_70b_tp8_pp8(model_70b):
    # attn_heads=64 (64%8=0), kv_heads=8 (8%8=0), layers=80 (80%8=0) → all pass
    assert filter_divisibility({"tp": 8, "pp": 8}, model_70b) is None


# ── filter_roofline_decode ─────────────────────────────────────────────────────

def test_roofline_decode_pass_8b_a6000(hw, model_8b):
    # shard=16 GB, A6000 bw=768 GB/s → lb=16/768×1000=20.8 ms < 100 ms
    assert filter_roofline_decode({"A6000": 1}, {"tp": 1, "pp": 1},
                                   model_8b, hw, fp=16, tpot_slo_ms=100.0) is None


def test_roofline_decode_fail_70b_a6000_tp2(hw, model_70b):
    # shard=141/2=70.5 GB, A6000 bw=768 GB/s → lb=91.8 ms > 50 ms
    result = filter_roofline_decode({"A6000": 2}, {"tp": 2, "pp": 1},
                                     model_70b, hw, fp=16, tpot_slo_ms=50.0)
    assert result is not None and "roofline_decode" in result


def test_roofline_decode_no_slo_skipped(hw, model_70b):
    assert filter_roofline_decode({"A6000": 1}, {"tp": 1, "pp": 1},
                                   model_70b, hw, fp=16, tpot_slo_ms=None) is None


def test_roofline_decode_pass_h100_tp8(hw, model_70b):
    # shard=141/8=17.6 GB, H100 bw=3350 GB/s → lb=5.3 ms < 100 ms
    assert filter_roofline_decode({"H100": 8}, {"tp": 8, "pp": 1},
                                   model_70b, hw, fp=16, tpot_slo_ms=100.0) is None


# ── filter_roofline_prefill ────────────────────────────────────────────────────

def test_roofline_prefill_pass_8b_h100(hw, model_8b):
    # flops=2×8.03e9×512=8.22e12, H100 peak=989 TFLOPS → lb=8.3 ms < 1000 ms
    assert filter_roofline_prefill({"H100": 1}, {"tp": 1, "pp": 1},
                                    model_8b, hw, fp=16,
                                    ttft_slo_ms=1000.0, avg_prompt_len=512) is None


def test_roofline_prefill_fail_70b_a6000_long_prompt(hw, model_70b):
    # flops=2×70.6e9×32768=4.63e15, A6000 peak=309.7 TFLOPS, tp=1
    # lb=4.63e15/(1×309.7e12)×1000≈14,950 ms >> 100 ms
    result = filter_roofline_prefill({"A6000": 1}, {"tp": 1, "pp": 1},
                                      model_70b, hw, fp=16,
                                      ttft_slo_ms=100.0, avg_prompt_len=32768)
    assert result is not None and "roofline_prefill" in result


def test_roofline_prefill_pass_with_tp8(hw, model_70b):
    # flops=2×70.6e9×512=7.23e13, H100 peak=989 TFLOPS, tp=8
    # lb=7.23e13/(8×989e12)×1000≈9.1 ms < 500 ms
    assert filter_roofline_prefill({"H100": 8}, {"tp": 8, "pp": 1},
                                    model_70b, hw, fp=16,
                                    ttft_slo_ms=500.0, avg_prompt_len=512) is None


def test_roofline_prefill_no_slo_skipped(hw, model_8b):
    assert filter_roofline_prefill({"A6000": 1}, {"tp": 1, "pp": 1},
                                    model_8b, hw, fp=16,
                                    ttft_slo_ms=None, avg_prompt_len=512) is None


# ── filter_communication ───────────────────────────────────────────────────────

def test_communication_tp1_always_passes(hw, model_8b):
    # tp=1: no all-reduce needed
    assert filter_communication({"A6000": 1}, {"tp": 1, "pp": 1},
                                 model_8b, hw, fp=16, tpot_slo_ms=1.0) is None


def test_communication_pass_h100_nvlink(hw, model_70b):
    # hidden=8192, layers=80, tp=4, H100 link=900 GB/s
    # one_ar = 2×(3/4)×16384/(900e9) ≈ 2.73e-8 s
    # total = 80×2×2.73e-8×1000 ≈ 4.4e-3 ms << 100 ms
    assert filter_communication({"H100": 4}, {"tp": 4, "pp": 1},
                                 model_70b, hw, fp=16, tpot_slo_ms=100.0) is None


def test_communication_fail_a6000_pcie_tight_slo(hw, model_70b):
    # hidden=8192, layers=80, tp=8, A6000 link=32 GB/s
    # one_ar = 2×(7/8)×16384/(32e9) ≈ 8.96e-7 s
    # total = 80×2×8.96e-7×1000 ≈ 0.143 ms > 0.05 ms
    result = filter_communication({"A6000": 8}, {"tp": 8, "pp": 1},
                                   model_70b, hw, fp=16, tpot_slo_ms=0.05)
    assert result is not None and "communication" in result


def test_communication_no_slo_skipped(hw, model_70b):
    assert filter_communication({"A6000": 8}, {"tp": 8, "pp": 1},
                                 model_70b, hw, fp=16, tpot_slo_ms=None) is None


def test_communication_no_interconnect_info_skipped(model_70b):
    # Hardware entry without interconnect_bw_gbs → filter must skip gracefully
    hw_no_link = {"FakeHW": {"mem_size_gb": 80, "mem_bw_gbs": 900,
                              "peak_flops_fp16_tflops": 500, "tdp_w": 400}}
    assert filter_communication({"FakeHW": 4}, {"tp": 4, "pp": 1},
                                 model_70b, hw_no_link, fp=16, tpot_slo_ms=1.0) is None


# ── filter_power ───────────────────────────────────────────────────────────────

def test_power_pass(hw):
    # 4×A6000 × 300W = 1200W < 2000W
    assert filter_power({"A6000": 4}, hw, power_max_w=2000.0) is None


def test_power_fail(hw):
    # 8×H100 × 700W = 5600W > 4000W
    result = filter_power({"H100": 8}, hw, power_max_w=4000.0)
    assert result is not None and "power" in result


def test_power_no_budget_skipped(hw):
    assert filter_power({"H100": 8}, hw, power_max_w=None) is None


def test_power_mixed_hw(hw):
    # 2×H100(700W) + 4×A6000(300W) = 2600W > 2000W
    result = filter_power({"H100": 2, "A6000": 4}, hw, power_max_w=2000.0)
    assert result is not None and "power" in result


# ── filter_tokwh_roofline ──────────────────────────────────────────────────────

def test_tokwh_pass(hw, model_8b):
    # 8B fp16, A6000×1, tp=1: shard=16 GB, bw=768 GB/s, TDP=300W
    # ub = (768/16)×3600/300 = 48×12 = 576 tok/Wh > 100 → passes
    assert filter_tokwh_roofline({"A6000": 1}, {"tp": 1, "pp": 1},
                                  model_8b, hw, fp=16, tokwh_min=100.0) is None


def test_tokwh_fail(hw, model_70b):
    # 70B fp16, H100×1, tp=1: shard=141 GB, bw=3350 GB/s, TDP=700W
    # ub = (3350/141)×3600/700 ≈ 122 tok/Wh < 500 → fails
    result = filter_tokwh_roofline({"H100": 1}, {"tp": 1, "pp": 1},
                                    model_70b, hw, fp=16, tokwh_min=500.0)
    assert result is not None and "tokwh" in result


def test_tokwh_no_requirement_skipped(hw, model_8b):
    assert filter_tokwh_roofline({"A6000": 1}, {"tp": 1, "pp": 1},
                                  model_8b, hw, fp=16, tokwh_min=None) is None
