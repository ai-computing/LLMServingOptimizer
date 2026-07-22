"""Logging, hashing/caching, profile-catalog scan, and model-size estimation.

These helpers are intentionally self-contained (they do not import the simulator
or the webapp) so the planner can be installed and unit-tested on its own.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths (relative to the repo root, i.e. the parent of the ``planner`` package)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
# legacy-format profile/model-config trees live in the legacy backend
# submodule (Stage-1 device tables + spec validation read this layout)
LEGACY_ROOT = REPO_ROOT / "backends" / "legacy"
PERF_MODELS_DIR = LEGACY_ROOT / "llm_profile" / "perf_models"
MODEL_CONFIG_DIR = LEGACY_ROOT / "model_config"

_TP_RE = re.compile(r"^tp(\d+)$")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "planner", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False  # avoid duplicate lines via parent/root handlers
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Hashing / disk cache
# ---------------------------------------------------------------------------
def hash_obj(obj) -> str:
    """Stable short hash of a JSON-serializable object (for run caching)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def stage_path(out_dir: Path, subpath: str) -> tuple[Path, str]:
    """Resolve a simulator-facing file to (abs_write_path, repo_relative_path).

    ``build_cluster_config`` (and the sim's --output handling) prepend ``../`` to
    the path they are given, assuming it is *repo-root-relative*. Absolute paths
    therefore break. If ``out_dir`` is inside the repo we use it directly;
    otherwise we mirror the file under ``output/planner_stage/<hash>/`` inside the
    repo so the ``../`` prefix always resolves.
    """
    out_dir = Path(out_dir).resolve()
    target = out_dir / subpath
    try:
        rel = target.relative_to(REPO_ROOT)
        return target, str(rel)
    except ValueError:
        staged = REPO_ROOT / "output" / "planner_stage" / hash_obj(str(out_dir)) / subpath
        return staged, str(staged.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Profile catalog
# ---------------------------------------------------------------------------
def _tp_dir_is_complete(tp_dir: Path) -> bool:
    """Whether a ``tp<N>/`` directory has the files the simulator needs.

    Mirrors ``webapp/hardware_catalog.py``: needs ``layers.csv`` plus attention
    prediction data (pkl or csv). Incomplete profiles (e.g. some models that only
    ship ``layers.csv``) would otherwise crash at simulation time.
    """
    if not (tp_dir / "layers.csv").is_file():
        return False
    pred = tp_dir / "predictions"
    pkl_ok = (pred / "attn_prefill_prediction_dict.pkl").is_file() and (
        pred / "attn_decode_prediction_dict.pkl"
    ).is_file()
    csv_ok = (pred / "attn_prefill_predictions.csv").is_file() and (
        pred / "attn_decode_predictions.csv"
    ).is_file()
    return pkl_ok or csv_ok


def scan_profile_catalog(
    perf_root: Optional[Path] = None,
) -> dict[tuple[str, str], frozenset[int]]:
    """Return ``{(hardware, model_name): frozenset(tp_degrees)}``.

    Layout scanned: ``<perf_root>/<hardware>/<vendor>/<model>/tp<N>/``; the model
    key is ``"<vendor>/<model>"`` so it matches HuggingFace ``model_name`` fields.
    Only complete profiles are included. Empty dict if the root does not exist.
    """
    root = Path(perf_root) if perf_root else PERF_MODELS_DIR
    catalog: dict[tuple[str, str], set[int]] = {}
    if not root.is_dir():
        return {}
    for hw_dir in sorted(root.iterdir()):
        if not hw_dir.is_dir():
            continue
        hardware = hw_dir.name
        for vendor_dir in sorted(hw_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            for model_dir in sorted(vendor_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = f"{vendor_dir.name}/{model_dir.name}"
                tps: set[int] = set()
                for tp_dir in model_dir.iterdir():
                    m = _TP_RE.match(tp_dir.name)
                    if m and tp_dir.is_dir() and _tp_dir_is_complete(tp_dir):
                        tps.add(int(m.group(1)))
                if tps:
                    catalog[(hardware, model_name)] = tps
    return {k: frozenset(v) for k, v in catalog.items()}


# ---------------------------------------------------------------------------
# Model size / memory estimation (linear proxy for Stage-1 feasibility)
# ---------------------------------------------------------------------------
def load_model_config(model_name: str) -> dict:
    """Load ``model_config/<model_name>.json`` (HuggingFace-style fields)."""
    path = MODEL_CONFIG_DIR / f"{model_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model config not found: {path} (model_name='{model_name}')"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_weight_bytes(cfg: dict, fp_bits: int) -> float:
    """Estimate total model weight size in bytes from an HF-style config.

    Covers embeddings, per-layer attention (GQA-aware) + MLP (gated), and lm_head.
    This is a coarse proxy for the Stage-1 memory constraint; Stage-2 simulation
    uses the simulator's own ``memory_model``.
    """
    h = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    inter = cfg.get("intermediate_size", 4 * h)
    n_heads = cfg.get("num_attention_heads", 1)
    n_kv = cfg.get("num_key_value_heads", n_heads)
    vocab = cfg.get("vocab_size", 0)

    head_dim = h // n_heads if n_heads else h
    kv_dim = head_dim * n_kv

    # attention: q(h*h) + k(h*kv_dim) + v(h*kv_dim) + o(h*h)
    attn = 2 * h * h + 2 * h * kv_dim
    # gated MLP: gate(h*inter) + up(h*inter) + down(inter*h)
    mlp = 3 * h * inter
    per_layer = attn + mlp
    params = n_layers * per_layer + 2 * vocab * h  # embed + lm_head (tied ~ 2*)
    return params * (fp_bits / 8.0)


def estimate_kv_bytes_per_token(cfg: dict, fp_bits: int) -> float:
    """KV-cache bytes per token: 2 (K,V) * layers * kv_dim * bytes."""
    h = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg.get("num_attention_heads", 1)
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = h // n_heads if n_heads else h
    kv_dim = head_dim * n_kv
    return 2 * n_layers * kv_dim * (fp_bits / 8.0)
