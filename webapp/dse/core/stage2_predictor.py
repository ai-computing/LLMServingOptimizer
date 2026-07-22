"""Stage 2 latency predictor — estimate TTFT and TPOT from llm_profile CSVs.

Reads pre-measured per-layer latency tables (layers.csv) and pre-computed
attention prediction tables to estimate TTFT/TPOT without running the simulator.

Layer breakdown per inference step:

  TTFT (prefill of avg_prompt_len tokens, single request):
      embedding(N)
    + num_layers × [dense_block(N) + attn_prefill(kv=0, chunk=N)]
    + final_layernorm(1) + lm_head(1)

  TPOT (decode of decode_batch_size sequences, 1 new token each):
      embedding(B)
    + num_layers × [dense_block(B) + attn_decode(batch=B, kv=S/2)]
    + final_layernorm(B) + lm_head(1)

where N=avg_prompt_len, B=decode_batch_size, S=max_seq_len.
All latencies are in nanoseconds internally; public API returns milliseconds.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


class ProfileNotFoundError(FileNotFoundError):
    pass


# Per-block dense layers (appear once per transformer layer = num_layers times).
# "attn" is deliberately excluded — it comes from the predictions/ CSV.
_BLOCK_LAYERS = (
    "input_layernorm", "q_proj", "k_proj", "v_proj", "rope",
    "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj",
)

# lm_head always processes 1 token (predict next token from last position only).
_LM_HEAD_INPUT = 1
# final_layernorm in TTFT: only the last token feeds into lm_head.
_FINAL_LN_PREFILL_INPUT = 1


def _default_profile_root() -> Path:
    """Legacy-format profile tree (backends/legacy/llm_profile/perf_models/).

    Stage-2 prediction reads the legacy CSV layout only; when targeting the
    upstream backend, pass profile_root explicitly or skip stage-2.
    """
    return (Path(__file__).resolve().parents[3]
            / "backends" / "legacy" / "llm_profile" / "perf_models")


def profile_root_from_env() -> Path:
    """Honour LLMSERVINGSIM_PROFILE_ROOT env-var; fall back to repo-relative default."""
    env = os.environ.get("LLMSERVINGSIM_PROFILE_ROOT")
    return Path(env) if env else _default_profile_root()


def _tp_dir(profile_root: Path, hardware: str, model_name: str, tp: int) -> Path:
    return profile_root / hardware / model_name / f"tp{tp}"


# ---------------------------------------------------------------------------
# CSV loaders — cached per path so the same file is read only once per process.

@lru_cache(maxsize=None)
def _load_layers_db(layers_csv: Path) -> dict:
    """Build {(layer_name, input, kv_cache, tp_size): latency_ns} from layers.csv."""
    if not layers_csv.exists():
        raise ProfileNotFoundError(f"layers.csv not found: {layers_csv}")
    df = pd.read_csv(layers_csv)
    required = {"layer_name", "input", "kv_cache", "tp_size", "latency(ns)"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {layers_csv}: {missing}")
    db: dict = {}
    for _, row in df.iterrows():
        key = (str(row["layer_name"]), int(row["input"]), int(row["kv_cache"]), int(row["tp_size"]))
        if key not in db:
            db[key] = int(row["latency(ns)"])
    return db


@lru_cache(maxsize=None)
def _load_attn_prefill_db(csv_path: Path) -> dict:
    """Build {(kv_cache_size, prefill_chunk_size): latency_ns} from attn_prefill_predictions.csv."""
    if not csv_path.exists():
        raise ProfileNotFoundError(f"attn_prefill_predictions.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    db: dict = {}
    for _, row in df.iterrows():
        key = (int(row["kv_cache_size"]), int(row["prefill_chunk_size"]))
        if key not in db:
            db[key] = int(row["prediction"])
    return db


@lru_cache(maxsize=None)
def _load_attn_decode_db(csv_path: Path) -> dict:
    """Build {(batch_size, kv_cache_size): latency_ns} from attn_decode_predictions.csv."""
    if not csv_path.exists():
        raise ProfileNotFoundError(f"attn_decode_predictions.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    db: dict = {}
    for _, row in df.iterrows():
        key = (int(row["batch_size"]), int(row["kv_cache_size"]))
        if key not in db:
            db[key] = int(row["prediction"])
    return db


# ---------------------------------------------------------------------------
# Nearest-neighbour lookup helpers

def _lookup_layer(db: dict, layer_name: str, input_len: int, kv_cache: int, tp: int) -> int:
    """Exact lookup; falls back to nearest input_len if key is absent."""
    key = (layer_name, input_len, kv_cache, tp)
    if key in db:
        return db[key]
    best_lat, best_diff = None, None
    for (ln, inp, kv, t), lat in db.items():
        if ln != layer_name or kv != kv_cache or t != tp:
            continue
        diff = abs(inp - input_len)
        if best_diff is None or diff < best_diff:
            best_diff, best_lat = diff, lat
    if best_lat is None:
        raise ProfileNotFoundError(
            f"No profile row for layer={layer_name!r}, tp={tp}, kv={kv_cache}"
        )
    return best_lat


def _lookup_attn_prefill(db: dict, kv_cache_size: int, chunk_size: int) -> int:
    """Nearest-neighbour: prefer exact kv_cache_size, then nearest chunk_size."""
    if (kv_cache_size, chunk_size) in db:
        return db[(kv_cache_size, chunk_size)]
    best_lat, best_diff, best_kv_match = None, None, False
    for (kv, chunk), lat in db.items():
        kv_match = kv == kv_cache_size
        diff = abs(chunk - chunk_size)
        better = (
            best_lat is None
            or (kv_match and not best_kv_match)
            or (kv_match == best_kv_match and diff < best_diff)
        )
        if better:
            best_diff, best_lat, best_kv_match = diff, lat, kv_match
    if best_lat is None:
        raise ProfileNotFoundError("attn_prefill_predictions.csv is empty")
    return best_lat


def _lookup_attn_decode(db: dict, batch_size: int, kv_cache_size: int) -> int:
    """Nearest-neighbour: prefer exact batch_size, then nearest kv_cache_size."""
    if (batch_size, kv_cache_size) in db:
        return db[(batch_size, kv_cache_size)]
    best_lat, best_diff, best_batch_match = None, None, False
    for (bs, kv), lat in db.items():
        batch_match = bs == batch_size
        diff = abs(kv - kv_cache_size)
        better = (
            best_lat is None
            or (batch_match and not best_batch_match)
            or (batch_match == best_batch_match and diff < best_diff)
        )
        if better:
            best_diff, best_lat, best_batch_match = diff, lat, batch_match
    if best_lat is None:
        raise ProfileNotFoundError("attn_decode_predictions.csv is empty")
    return best_lat


# ---------------------------------------------------------------------------
# Core prediction function

def predict_ttft_tpot(
    hardware: str,
    model_name: str,
    tp: int,
    num_layers: int,
    avg_prompt_len: int,
    max_seq_len: int,
    decode_batch_size: int = 1,
    profile_root: Path | None = None,
) -> dict[str, Any]:
    """Estimate TTFT and TPOT (ms) from llm_profile CSV tables.

    Args:
        hardware:          HW directory name matching perf_models/ (e.g. "A6000")
        model_name:        HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B")
        tp:                tensor-parallel degree
        num_layers:        number of transformer blocks
        avg_prompt_len:    average prompt length in tokens (for TTFT)
        max_seq_len:       max sequence length; KV cache at decode ≈ max_seq_len // 2
        decode_batch_size: concurrent decode sequences (for TPOT, default 1)
        profile_root:      root of perf_models/ directory; uses repo default if None

    Returns:
        dict with keys: ttft_pred_ms, tpot_pred_ms, breakdown (dict), profile_source (str)

    Raises:
        ProfileNotFoundError: if any required CSV is missing
    """
    if profile_root is None:
        profile_root = profile_root_from_env()

    tpdir = _tp_dir(profile_root, hardware, model_name, tp)
    if not tpdir.exists():
        available = sorted(
            int(p.name[2:]) for p in tpdir.parent.glob("tp*") if p.is_dir()
        ) if tpdir.parent.exists() else []
        raise ProfileNotFoundError(
            f"Profile directory not found: {tpdir}\n"
            f"  Available TP values for this hw/model: {available}"
        )

    layers_db  = _load_layers_db(tpdir / "layers.csv")
    prefill_db = _load_attn_prefill_db(tpdir / "predictions" / "attn_prefill_predictions.csv")
    decode_db  = _load_attn_decode_db(tpdir / "predictions" / "attn_decode_predictions.csv")

    avg_kv_len = max_seq_len // 2

    # ── TTFT: prefill of avg_prompt_len tokens ──────────────────────────────
    emb_ns      = _lookup_layer(layers_db, "embedding",       avg_prompt_len,          0, tp)
    fln_ns      = _lookup_layer(layers_db, "final_layernorm", _FINAL_LN_PREFILL_INPUT, 0, tp)
    lmh_ns      = _lookup_layer(layers_db, "lm_head",         _LM_HEAD_INPUT,          0, tp)

    block_pre_ns = sum(
        _lookup_layer(layers_db, ln, avg_prompt_len, 0, tp) for ln in _BLOCK_LAYERS
    )
    attn_pre_ns  = _lookup_attn_prefill(prefill_db, kv_cache_size=0, chunk_size=avg_prompt_len)

    ttft_ns = emb_ns + num_layers * (block_pre_ns + attn_pre_ns) + fln_ns + lmh_ns

    # ── TPOT: decode of decode_batch_size sequences, 1 token each ──────────
    emb_dec_ns  = _lookup_layer(layers_db, "embedding",       decode_batch_size, 0, tp)
    fln_dec_ns  = _lookup_layer(layers_db, "final_layernorm", decode_batch_size, 0, tp)
    lmh_dec_ns  = _lookup_layer(layers_db, "lm_head",         _LM_HEAD_INPUT,   0, tp)

    block_dec_ns = sum(
        _lookup_layer(layers_db, ln, decode_batch_size, 0, tp) for ln in _BLOCK_LAYERS
    )
    attn_dec_ns  = _lookup_attn_decode(decode_db, batch_size=decode_batch_size, kv_cache_size=avg_kv_len)

    tpot_ns = emb_dec_ns + num_layers * (block_dec_ns + attn_dec_ns) + fln_dec_ns + lmh_dec_ns

    return {
        "ttft_pred_ms": ttft_ns / 1_000_000,
        "tpot_pred_ms": tpot_ns / 1_000_000,
        "breakdown": {
            "ttft_emb_ns":         emb_ns,
            "ttft_block_dense_ns": num_layers * block_pre_ns,
            "ttft_attn_ns":        num_layers * attn_pre_ns,
            "ttft_fln_lmh_ns":     fln_ns + lmh_ns,
            "tpot_emb_ns":         emb_dec_ns,
            "tpot_block_dense_ns": num_layers * block_dec_ns,
            "tpot_attn_ns":        num_layers * attn_dec_ns,
            "tpot_fln_lmh_ns":     fln_dec_ns + lmh_dec_ns,
        },
        "profile_source": str(tpdir),
    }


# ---------------------------------------------------------------------------
# Sanity check

def sanity_check_vs_roofline(
    ttft_pred_ms: float,
    tpot_pred_ms: float,
    roofline_tpot_lb_ms: float | None,
    roofline_ttft_lb_ms: float | None,
    tolerance: float = 0.9,
) -> list[str]:
    """Return warning strings if Stage 2 predictions violate Stage 1 roofline lower bounds.

    Stage 2 prediction must be ≥ Stage 1 lower bound × tolerance (default 90%).
    A violation indicates a unit error or a catalog / CSV mismatch.
    """
    warnings: list[str] = []
    if roofline_tpot_lb_ms is not None and tpot_pred_ms < roofline_tpot_lb_ms * tolerance:
        warnings.append(
            f"TPOT pred {tpot_pred_ms:.2f} ms < roofline lb {roofline_tpot_lb_ms:.2f} ms "
            f"(×{tolerance:.0%}) — check HBM BW in catalog or latency(ns) units in layers.csv"
        )
    if roofline_ttft_lb_ms is not None and ttft_pred_ms < roofline_ttft_lb_ms * tolerance:
        warnings.append(
            f"TTFT pred {ttft_pred_ms:.2f} ms < roofline lb {roofline_ttft_lb_ms:.2f} ms "
            f"(×{tolerance:.0%}) — check peak FLOPS in catalog or latency(ns) units in layers.csv"
        )
    return warnings


# ---------------------------------------------------------------------------
# Batch application over a list of CandidateConfig objects

def apply_stage2_predictions(
    candidates: list[Any],   # list[CandidateConfig]
    spec: Any,               # JobSpec
    metadata: dict,
    profile_root: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Run Stage 2 predictions over Stage 1 survivors.

    For mixed-hardware candidates, uses the hardware type with the most NPUs
    as the primary for profile lookup.

    Returns:
        (predictions, warnings)
        predictions: list of dicts sorted by ttft_pred_ms, each containing
                     {candidate_id, label, ttft_pred_ms, tpot_pred_ms, breakdown, profile_source}
        warnings: list of sanity-check warning strings
    """
    if profile_root is None:
        profile_root = profile_root_from_env()

    model_meta = metadata.get("models", {}).get(spec.model.name, {})
    num_layers  = model_meta.get("num_hidden_layers", 0)
    avg_prompt  = spec.workload.avg_prompt_len
    max_seq     = spec.workload.max_seq_len

    all_warnings: list[str] = []
    predictions: list[dict] = []

    for c in candidates:
        # Pick primary hardware: the type with most NPUs
        hw = max(c.hw_distribution, key=lambda h: c.hw_distribution[h])
        tp = c.parallelism.get("tp", 1)

        try:
            result = predict_ttft_tpot(
                hardware=hw,
                model_name=spec.model.name,
                tp=tp,
                num_layers=num_layers,
                avg_prompt_len=avg_prompt,
                max_seq_len=max_seq,
                decode_batch_size=1,
                profile_root=profile_root,
            )
        except (ProfileNotFoundError, KeyError) as exc:
            predictions.append({
                "candidate_id": c.candidate_id,
                "label": c.label,
                "ttft_pred_ms": float("inf"),
                "tpot_pred_ms": float("inf"),
                "breakdown": {},
                "profile_source": "N/A",
                "error": str(exc),
            })
            continue

        predictions.append({
            "candidate_id": c.candidate_id,
            "label": c.label,
            **result,
        })

    predictions.sort(key=lambda p: p["ttft_pred_ms"])
    return predictions, all_warnings
