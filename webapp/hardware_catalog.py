"""Hardware/model/TP catalog built from llm_profile/perf_models/.

Directory layout:
    llm_profile/perf_models/<hardware>/<vendor>/<model>/tp<N>/

The model key is "<vendor>/<model>" so it matches HuggingFace IDs used in
cluster JSON `model_name` fields.
"""
from __future__ import annotations

import re

from .config import LLM_PROFILE_DIR

_TP_RE = re.compile(r"^tp(\d+)$")


def _tp_dir_is_complete(tp_dir) -> bool:
    """Check whether the tp<N>/ directory has the files trace_generator.py needs.

    Required:
      - layers.csv (for _load_perf_db_dict)
      - Attention data via either:
          * predictions/attn_prefill_prediction_dict.pkl + attn_decode_prediction_dict.pkl
          * predictions/attn_prefill_predictions.csv     + attn_decode_predictions.csv

    Skipping the check would let incomplete profiles (e.g. H100/Llama-3.1-8B,
    which only ships layers.csv) appear valid until simulation time, where they
    crash with FileNotFoundError in _load_attn_perf_db_dict.
    """
    if not (tp_dir / "layers.csv").is_file():
        return False
    pred = tp_dir / "predictions"
    pkl_ok = (
        (pred / "attn_prefill_prediction_dict.pkl").is_file()
        and (pred / "attn_decode_prediction_dict.pkl").is_file()
    )
    csv_ok = (
        (pred / "attn_prefill_predictions.csv").is_file()
        and (pred / "attn_decode_predictions.csv").is_file()
    )
    return pkl_ok or csv_ok


def build_catalog(backend: str = "legacy") -> dict[tuple[str, str], frozenset[int]]:
    """Return {(hardware, model): frozenset[tp]} for the given backend.

    legacy   — scan LLM_PROFILE_DIR (old CSV layout) with completeness checks.
    upstream — delegate to sim_backends (profiler/perf new layout).

    Returns an empty dict if the profile directory does not exist.
    Only includes TP entries whose tp<N>/ directory has all required perf
    files — incomplete profiles are silently skipped.
    """
    if backend == "upstream":
        from sim_backends import get_backend
        hw_map = get_backend("upstream").list_hardware()
        return {(hw, model): frozenset(tps)
                for hw, models in hw_map.items()
                for model, tps in models.items()}

    catalog: dict[tuple[str, str], set[int]] = {}
    if not LLM_PROFILE_DIR.is_dir():
        return {}

    for hw_dir in sorted(LLM_PROFILE_DIR.iterdir()):
        if not hw_dir.is_dir():
            continue
        hardware = hw_dir.name
        for vendor_dir in sorted(hw_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            vendor = vendor_dir.name
            for model_dir in sorted(vendor_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = f"{vendor}/{model_dir.name}"
                tps: set[int] = set()
                for tp_dir in model_dir.iterdir():
                    if not tp_dir.is_dir():
                        continue
                    m = _TP_RE.match(tp_dir.name)
                    if m and _tp_dir_is_complete(tp_dir):
                        tps.add(int(m.group(1)))
                if tps:
                    catalog[(hardware, model)] = tps

    return {k: frozenset(v) for k, v in catalog.items()}


def list_hardware_models(catalog: dict[tuple[str, str], frozenset[int]]) -> list[tuple[str, str]]:
    """Return sorted list of all (hardware, model) pairs in the catalog."""
    return sorted(catalog.keys())


def get_tp_options(
    catalog: dict[tuple[str, str], frozenset[int]],
    hardware: str,
    model: str,
) -> list[int]:
    """Return sorted list of TP values for (hw, model), or [] if not found."""
    return sorted(catalog.get((hardware, model), frozenset()))


def list_hardware(catalog: dict[tuple[str, str], frozenset[int]]) -> list[str]:
    """Return sorted list of all hardware names in the catalog."""
    return sorted({hw for (hw, _) in catalog.keys()})


def list_models_for_hardware(
    catalog: dict[tuple[str, str], frozenset[int]],
    hardware: str,
) -> list[str]:
    """Return sorted list of model names available for the given hardware."""
    return sorted({model for (hw, model) in catalog.keys() if hw == hardware})
