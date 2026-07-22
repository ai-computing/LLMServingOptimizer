"""Stage 1 analytical filters — prune candidates before simulation.

Each filter is a pure function: no I/O, no simulator calls.
Applied in generator.py AFTER label-dedup and BEFORE sampling so the
sampling budget is spent only on physically feasible candidates.

Filters (applied in order):
  1. memory          — weight_shard + KV_cache + activation per NPU fits in HBM
  2. divisibility    — TP/PP divisibility for attention heads and layers
  3. roofline_decode — TPOT roofline lower bound vs SLO
  4. roofline_prefill — TTFT roofline lower bound vs SLO
  5. communication   — TP ring all-reduce latency lower bound vs TPOT SLO
  6. power           — aggregate TDP vs power_max_w budget
  7. tokwh           — tokens/Wh roofline upper-bound vs minimum efficiency target
"""
from __future__ import annotations

from typing import Any

_MEM_SAFETY = 0.85    # weight shard must fit within 85 % of HBM


# ---------------------------------------------------------------------------
# Internal helpers

def _weight_gb(model_meta: dict, fp: int) -> float:
    key = f"weight_size_fp{fp}_gb"
    if key in model_meta:
        return float(model_meta[key])
    return float(model_meta.get("params_b", 0)) * (fp / 8)


def _min_mem_gb(hw_counts: dict[str, int], hw_meta: dict) -> float:
    """Smallest per-NPU HBM across all hardware types in the candidate."""
    return min(
        hw_meta[hw]["mem_size_gb"]
        for hw, cnt in hw_counts.items()
        if cnt > 0 and hw in hw_meta
    )


def _min_mem_bw_gbs(hw_counts: dict[str, int], hw_meta: dict) -> float:
    return min(
        hw_meta[hw]["mem_bw_gbs"]
        for hw, cnt in hw_counts.items()
        if cnt > 0 and hw in hw_meta
    )


def _total_tdp_w(hw_counts: dict[str, int], hw_meta: dict) -> float:
    return sum(
        cnt * hw_meta[hw]["tdp_w"]
        for hw, cnt in hw_counts.items()
        if cnt > 0 and hw in hw_meta
    )


def _min_peak_flops(hw_counts: dict[str, int], hw_meta: dict) -> float:
    """Minimum peak FP16 FLOPS (in FLOP/s) across all hardware types."""
    return min(
        hw_meta[hw]["peak_flops_fp16_tflops"] * 1e12
        for hw, cnt in hw_counts.items()
        if cnt > 0 and hw in hw_meta
    )


def _min_interconnect_bw_gbs(hw_counts: dict[str, int], hw_meta: dict) -> float:
    """Minimum intra-node interconnect bandwidth (GB/s).
    Returns 0.0 if interconnect_bw_gbs is absent from any entry.
    """
    bws = [
        hw_meta[hw].get("interconnect_bw_gbs", 0.0)
        for hw, cnt in hw_counts.items()
        if cnt > 0 and hw in hw_meta
    ]
    return min(bws) if bws else 0.0


# ---------------------------------------------------------------------------
# Individual filter functions
# Each returns a rejection reason string, or None if the candidate passes.

def filter_memory(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    max_seq_len: int = 2048,
) -> str | None:
    """Per-NPU memory check: weight_shard + KV_cache + activation ≤ _MEM_SAFETY × HBM.

    KV cache is estimated for a single request (batch=1) — even one request must
    fit alongside the model weights.  Activation covers one transformer layer peak
    (SwiGLU intermediate ≈ 4× hidden; gradient-checkpointing / recompute assumed).
    """
    tp = parallelism.get("tp", 1)
    pp = parallelism.get("pp", 1)
    if tp * pp == 0:
        return "memory: tp×pp == 0"

    bytes_per_elem = fp / 8

    # weight shard per device
    shard_gb = _weight_gb(model_meta, fp) / (tp * pp)

    # KV cache for one request: 2 × L × H_kv × d_head × bytes × seq_len / tp
    num_layers = model_meta.get("num_hidden_layers", 0)
    kv_heads   = model_meta.get("num_key_value_heads", 0)
    hidden     = model_meta.get("hidden_size", 0)
    attn_heads = model_meta.get("num_attention_heads", 1) or 1
    head_dim   = hidden // attn_heads if hidden and attn_heads else 128
    kv_per_token_bytes = 2 * num_layers * kv_heads * head_dim * bytes_per_elem
    kv_gb = (kv_per_token_bytes * max_seq_len) / (tp * 1e9)

    # peak activation for one layer (prefill, single request)
    act_gb = (max_seq_len * hidden * 4 * bytes_per_elem) / (tp * 1e9)

    total_gb = shard_gb + kv_gb + act_gb
    limit_gb = _min_mem_gb(hw_counts, hw_meta) * _MEM_SAFETY

    if total_gb > limit_gb:
        return (
            f"memory: weight={shard_gb:.1f} + KV={kv_gb:.2f} + act={act_gb:.2f} "
            f"= {total_gb:.1f} GB "
            f"> {_MEM_SAFETY}×HBM={limit_gb:.1f} GB "
            f"(tp={tp}, pp={pp}, seq={max_seq_len})"
        )
    return None


def filter_divisibility(
    parallelism: dict[str, int],
    model_meta: dict,
) -> str | None:
    """TP/PP divisibility constraints for transformer parallelism.

    Checks:
    - num_attention_heads % tp == 0  (column-parallel attention)
    - num_key_value_heads % tp == 0  (GQA: KV must split evenly across TP)
    - num_hidden_layers   % pp == 0  (pipeline stages must have equal layer counts)
    """
    tp = parallelism.get("tp", 1)
    pp = parallelism.get("pp", 1)

    attn_heads = model_meta.get("num_attention_heads")
    if attn_heads and tp > 1 and attn_heads % tp != 0:
        return (
            f"divisibility: num_attention_heads={attn_heads} "
            f"not divisible by tp={tp}"
        )

    kv_heads = model_meta.get("num_key_value_heads")
    if kv_heads and tp > 1 and kv_heads % tp != 0:
        return (
            f"divisibility: num_key_value_heads={kv_heads} "
            f"not divisible by tp={tp} (GQA constraint)"
        )

    num_layers = model_meta.get("num_hidden_layers")
    if num_layers and pp > 1 and num_layers % pp != 0:
        return (
            f"divisibility: num_hidden_layers={num_layers} "
            f"not divisible by pp={pp}"
        )

    return None


def filter_roofline_decode(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    tpot_slo_ms: float | None,
) -> str | None:
    """Decode TPOT roofline lower bound: weight_shard / mem_bw must fit SLO.

    During decode each generated token requires reading the full weight shard
    from HBM once (memory-bandwidth bound at batch-size 1).  This gives a
    hard lower bound on per-token latency regardless of batch size:

        TPOT_lb_ms = (weight_shard_GB / mem_bw_GB/s) × 1000
    """
    if tpot_slo_ms is None:
        return None
    tp = parallelism.get("tp", 1)
    pp = parallelism.get("pp", 1)
    if tp * pp == 0:
        return None
    shard_gb = _weight_gb(model_meta, fp) / (tp * pp)
    bw_gbs = _min_mem_bw_gbs(hw_counts, hw_meta)
    if bw_gbs <= 0:
        return None
    tpot_lb_ms = (shard_gb / bw_gbs) * 1000
    if tpot_lb_ms > tpot_slo_ms:
        return (
            f"roofline_decode: TPOT_lb={tpot_lb_ms:.1f} ms "
            f"> SLO={tpot_slo_ms:.1f} ms "
            f"(shard={shard_gb:.1f} GB, bw={bw_gbs:.0f} GB/s)"
        )
    return None


def filter_roofline_prefill(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    ttft_slo_ms: float | None,
    avg_prompt_len: int,
) -> str | None:
    """Prefill TTFT roofline lower bound: 2×N_params×T_prompt / (tp×peak_FLOPS).

    This is the compute lower bound for a single-request prefill.  If even this
    minimum time exceeds the TTFT SLO, no parallelism or batching trick can help.
    """
    if ttft_slo_ms is None:
        return None
    tp = parallelism.get("tp", 1)
    if tp <= 0:
        return None

    params = model_meta.get("params_b", 0) * 1e9
    # FMA: each parameter contributes 2 FLOPs per input token
    prefill_flops = 2.0 * params * avg_prompt_len
    peak_flops = _min_peak_flops(hw_counts, hw_meta)
    if peak_flops <= 0:
        return None

    ttft_lb_ms = (prefill_flops / (tp * peak_flops)) * 1000
    if ttft_lb_ms > ttft_slo_ms:
        return (
            f"roofline_prefill: TTFT_lb={ttft_lb_ms:.1f} ms "
            f"> SLO={ttft_slo_ms:.1f} ms "
            f"(params={params/1e9:.1f}B, prompt={avg_prompt_len} tok, tp={tp})"
        )
    return None


def filter_communication(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    tpot_slo_ms: float | None,
) -> str | None:
    """TP ring all-reduce latency lower bound vs TPOT SLO.

    For each decode step, TP requires 2 all-reduce ops per layer (after
    attention and after FFN).  Ring all-reduce latency per op:

        t = 2(tp-1)/tp × msg_bytes / link_bw

    where msg_bytes = hidden_size × bytes_per_elem  (1 decode token).
    Skipped when tp ≤ 1 or interconnect_bw_gbs is absent from the catalog.
    """
    if tpot_slo_ms is None:
        return None
    tp = parallelism.get("tp", 1)
    if tp <= 1:
        return None

    link_bw_gbs = _min_interconnect_bw_gbs(hw_counts, hw_meta)
    if link_bw_gbs <= 0:
        return None  # no interconnect info → skip conservatively

    hidden     = model_meta.get("hidden_size", 0)
    num_layers = model_meta.get("num_hidden_layers", 0)
    bytes_per_elem = fp / 8

    msg_bytes = hidden * bytes_per_elem
    link_bw   = link_bw_gbs * 1e9  # → bytes/s

    # ring all-reduce: 2(tp-1)/tp × msg / bw
    one_ar_s = 2.0 * (tp - 1) / tp * msg_bytes / link_bw
    # decode step: num_layers × 2 all-reduces (after attention + after FFN)
    total_comm_ms = num_layers * 2 * one_ar_s * 1000

    if total_comm_ms > tpot_slo_ms:
        return (
            f"communication: all-reduce/token={total_comm_ms:.3f} ms "
            f"> TPOT SLO={tpot_slo_ms:.1f} ms "
            f"(tp={tp}, hidden={hidden}, link={link_bw_gbs:.0f} GB/s, layers={num_layers})"
        )
    return None


def filter_power(
    hw_counts: dict[str, int],
    hw_meta: dict,
    power_max_w: float | None,
) -> str | None:
    """Aggregate TDP must not exceed the power budget."""
    if power_max_w is None:
        return None
    total_w = _total_tdp_w(hw_counts, hw_meta)
    if total_w > power_max_w:
        return (
            f"power: TDP={total_w:.0f} W > budget={power_max_w:.0f} W"
        )
    return None


def filter_tokwh_roofline(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    tokwh_min: float | None,
) -> str | None:
    """Tokens/Wh roofline upper-bound check.

    Best-case throughput (memory-BW limited, batch=1 decode) divided by TDP:

        tok/s_ub  = mem_bw_GB/s / weight_shard_GB
        tok/Wh_ub = tok/s_ub × 3600 / TDP_W

    If this ceiling is below tokwh_min, the candidate can never reach the
    required energy efficiency.
    """
    if tokwh_min is None:
        return None
    tp = parallelism.get("tp", 1)
    pp = parallelism.get("pp", 1)
    if tp * pp == 0:
        return None
    shard_gb = _weight_gb(model_meta, fp) / (tp * pp)
    if shard_gb <= 0:
        return None
    bw_gbs = _min_mem_bw_gbs(hw_counts, hw_meta)
    tdp_w = _total_tdp_w(hw_counts, hw_meta)
    if tdp_w <= 0:
        return None
    tokwh_ub = (bw_gbs / shard_gb) * 3600.0 / tdp_w
    if tokwh_ub < tokwh_min:
        return (
            f"tokwh: roofline_ub={tokwh_ub:.0f} tok/Wh "
            f"< required={tokwh_min:.0f} tok/Wh "
            f"(bw={bw_gbs:.0f} GB/s, shard={shard_gb:.1f} GB, TDP={tdp_w:.0f} W)"
        )
    return None


# ---------------------------------------------------------------------------
# Combined entry points

def apply_stage1_filters(
    candidates: list[Any],   # list[CandidateConfig]
    spec: Any,               # JobSpec
    metadata: dict,
) -> tuple[list[Any], dict[str, int]]:
    """Apply all Stage 1 analytical filters to a candidate list.

    Filters are applied in order; the first failing filter records the rejection.
    Returns (survivors, per-filter rejection counts).
    """
    hw_meta    = metadata.get("hardware", {})
    model_meta = metadata.get("models", {}).get(spec.model.name, {})
    fp         = spec.model.fp
    tpot_slo   = spec.constraints.tpot_p99_ms
    ttft_slo   = spec.constraints.ttft_p99_ms
    power_max  = spec.constraints.power_max_w
    tokwh_min  = spec.constraints.tokwh_min
    avg_prompt = spec.workload.avg_prompt_len
    max_seq    = spec.workload.max_seq_len

    rejection_counts: dict[str, int] = {}
    survivors: list[Any] = []

    for c in candidates:
        hw_counts = c.hw_distribution
        par       = c.parallelism

        reason = (
            filter_memory(hw_counts, par, model_meta, hw_meta, fp, max_seq)
            or filter_divisibility(par, model_meta)
            or filter_roofline_decode(hw_counts, par, model_meta, hw_meta, fp, tpot_slo)
            or filter_roofline_prefill(hw_counts, par, model_meta, hw_meta, fp, ttft_slo, avg_prompt)
            or filter_communication(hw_counts, par, model_meta, hw_meta, fp, tpot_slo)
            or filter_power(hw_counts, hw_meta, power_max)
            or filter_tokwh_roofline(hw_counts, par, model_meta, hw_meta, fp, tokwh_min)
        )

        if reason:
            key = reason.split(":")[0]
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
        else:
            survivors.append(c)

    return survivors, rejection_counts


def check_candidate(
    hw_counts: dict[str, int],
    parallelism: dict[str, int],
    model_meta: dict,
    hw_meta: dict,
    fp: int,
    tpot_slo: float | None,
    power_max: float | None,
    tokwh_min: float | None = None,
    ttft_slo: float | None = None,
    avg_prompt_len: int = 512,
    max_seq_len: int = 2048,
) -> bool:
    """Single-candidate pass/fail check (used by dry_run_detail)."""
    return not (
        filter_memory(hw_counts, parallelism, model_meta, hw_meta, fp, max_seq_len)
        or filter_divisibility(parallelism, model_meta)
        or filter_roofline_decode(hw_counts, parallelism, model_meta, hw_meta, fp, tpot_slo)
        or filter_roofline_prefill(hw_counts, parallelism, model_meta, hw_meta, fp, ttft_slo, avg_prompt_len)
        or filter_communication(hw_counts, parallelism, model_meta, hw_meta, fp, tpot_slo)
        or filter_power(hw_counts, hw_meta, power_max)
        or filter_tokwh_roofline(hw_counts, parallelism, model_meta, hw_meta, fp, tokwh_min)
    )
