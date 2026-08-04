"""Precision comes from the checkpoint, not from spec.model.fp.

Stage-1 sized every checkpoint as fp16. For the quantized 14B checkpoints that
overstated the weights 2-4x and rejected tp1 on a 24 GB card — configurations
we had physically measured serving at tp1 in the A5000 precision study.
"""
from __future__ import annotations

import pytest

from planner.milp_solver import _memory_feasible
from planner.utils import (
    estimate_kv_bytes_per_token,
    estimate_weight_bytes,
    load_model_config,
    model_precision,
)

GB = 1024 ** 3

# a 14B-class config, the size where 24 GB cards make the difference visible
QWEN14B = {"hidden_size": 5120, "num_hidden_layers": 48,
           "intermediate_size": 13824, "num_attention_heads": 40,
           "num_key_value_heads": 8, "vocab_size": 152064,
           "torch_dtype": "bfloat16"}


def _with_quant(**quant):
    return {**QWEN14B, "quantization_config": quant}


# ---- derivation from the config ---------------------------------------------

def test_unquantized_precision_follows_torch_dtype():
    p = model_precision(QWEN14B)
    assert (p.weight_bits, p.kv_bits, p.quant_method) == (16, 16, None)
    assert p.label == "fp16/bf16"
    assert model_precision({**QWEN14B, "torch_dtype": "float32"}).weight_bits == 32


def test_awq_int4_reports_bits_and_group():
    p = model_precision(_with_quant(quant_method="awq", bits=4, group_size=128))
    assert (p.weight_bits, p.quant_method) == (4, "awq")
    assert p.label == "int4 (AWQ, group 128)"
    # AWQ is weight-only: the KV cache stays at the model dtype
    assert p.kv_bits == 16


@pytest.mark.parametrize("num_type,expected", [("float", "fp8"), ("int", "int8")])
def test_compressed_tensors_distinguishes_fp8_from_int8(num_type, expected):
    p = model_precision(_with_quant(
        quant_method="compressed-tensors",
        config_groups={"group_0": {
            "weights": {"num_bits": 8, "type": num_type, "strategy": "channel"},
            "input_activations": {"num_bits": 8, "strategy": "token"}}}))
    assert p.weight_bits == 8 and p.kv_bits == 16
    assert p.label == f"{expected} (W8A8)"
    assert p.quant_method == "compressed-tensors"


def test_weight_only_scheme_is_labelled_as_such():
    p = model_precision(_with_quant(
        quant_method="compressed-tensors",
        config_groups={"group_0": {"weights": {"num_bits": 8, "type": "int"}}}))
    assert p.label == "int8 (weight-only)"


def test_explicit_kv_cache_scheme_shrinks_the_cache():
    p = model_precision(_with_quant(
        quant_method="compressed-tensors",
        kv_cache_scheme={"num_bits": 8, "type": "float"},
        config_groups={"group_0": {"weights": {"num_bits": 8, "type": "float"}}}))
    assert p.kv_bits == 8


# ---- the bug this fixes -----------------------------------------------------

@pytest.mark.parametrize("bits,fits_tp1", [(16, False), (8, True), (4, True)])
def test_quantized_14b_fits_one_24gb_card(bits, fits_tp1):
    """fp16 needs two cards (27.5 GiB of weights); 8- and 4-bit do not."""
    w = estimate_weight_bytes(QWEN14B, bits)
    kv = estimate_kv_bytes_per_token(QWEN14B, 16)      # KV stays 16-bit
    assert _memory_feasible(w, kv, 1, 24) is fits_tp1
    assert _memory_feasible(w, kv, 2, 24) is True      # tp2 always fits
    if bits == 16:
        assert w / GB == pytest.approx(27.5, abs=0.2)  # matches the measurement


# ---- the staged configs the service actually offers --------------------------

@pytest.mark.parametrize("model,bits,label", [
    ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 4, "int4 (AWQ, group 128)"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8", 8, "fp8 (W8A8)"),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 8, "int8 (W8A8)"),
    ("meta-llama/Llama-3.1-8B", 16, "fp16/bf16"),
])
def test_catalog_checkpoints_have_a_readable_config(model, bits, label):
    """Every precision entry the model list offers must be loadable, or Stage-1
    dies with FileNotFoundError the moment someone picks it."""
    p = model_precision(load_model_config(model))
    assert (p.weight_bits, p.label) == (bits, label)
