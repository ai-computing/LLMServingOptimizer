"""Interconnect resolution for DSE candidates.

Turns a candidate (a ConfigSpec with a known TP degree) plus an optional
*fabric* definition into the link parameters consumed by
`inference_serving.config_builder._create_network_config`:

    {link_bw, link_latency, tp_group_shape}

Two modes:

1. **Fabric preset** (docs/dse/fabrics.yaml): the physical box is described as a
   tier list, innermost (fastest) first.  A candidate's TP group is decomposed
   onto the leading tiers — e.g. tiers [2,2,2] map TP4 → tp_group_shape=[2,2]
   with per-dim bandwidths [nvlink, pcie].  This mirrors the hierarchical
   topology support added in config_builder.py (tp_group_shape).

2. **Catalog scalar fallback** (no fabric): use the per-hardware
   `interconnect_bw_gbs` from 03_catalog.yaml as a flat scalar.  For
   heterogeneous candidates the *minimum* across the candidate's hardware is
   used — the slowest link bounds collective (all-reduce) cost.

Returning an empty dict means "no interconnect info" — the caller should fall
back to LINK_BW_DEFAULT (preserves the legacy flat behaviour).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_FABRICS_YAML = Path(__file__).resolve().parents[3] / "docs" / "dse" / "fabrics.yaml"


@lru_cache(maxsize=1)
def load_fabrics() -> dict[str, dict[str, Any]]:
    """Load docs/dse/fabrics.yaml → {fabric_name: {description, tiers}}.

    Returns an empty dict if the file is missing or malformed.
    """
    if not _FABRICS_YAML.is_file():
        return {}
    try:
        with open(_FABRICS_YAML) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}
    return data.get("fabrics", {}) or {}


def _candidate_tp(config_spec: Any) -> int:
    """TP degree of the candidate (npus_per_group). Display value `tp` on ConfigSpec."""
    return max(1, int(getattr(config_spec, "tp", 1) or 1))


def _candidate_hardware(config_spec: Any) -> list[str]:
    return [inst.hardware for inst in getattr(config_spec, "instances", [])]


def _decompose_tp(tp: int, tiers: list[dict]) -> tuple[list[int], list[float], list[float]] | None:
    """Place a TP group of size `tp` onto fabric tiers (innermost-first).

    Returns (tp_group_shape, bw_per_dim, latency_per_dim) or None when `tp`
    cannot be fully factored across the available tiers (caller falls back to
    a conservative scalar).  Uses gcd so a tier larger than the remaining TP
    contributes only the divisible part (e.g. tier size 8, remaining 4 → 4).
    """
    shape: list[int] = []
    bws: list[float] = []
    lats: list[float] = []
    remaining = tp
    for tier in tiers:
        if remaining == 1:
            break
        use = math.gcd(int(tier["size"]), remaining)
        if use <= 1:
            continue
        shape.append(use)
        bws.append(float(tier["bw_gbs"]))
        lats.append(float(tier.get("latency_ns", 0)))
        remaining //= use
    if remaining != 1:
        return None
    return shape, bws, lats


def _catalog_scalar_bw(config_spec: Any, hw_meta: dict[str, Any]) -> float | None:
    """Minimum interconnect_bw_gbs across the candidate's hardware, or None."""
    vals = [
        float(hw_meta[hw]["interconnect_bw_gbs"])
        for hw in _candidate_hardware(config_spec)
        if hw in hw_meta and hw_meta[hw].get("interconnect_bw_gbs") is not None
    ]
    return min(vals) if vals else None


def resolve_interconnect(
    config_spec: Any,
    fabric_def: dict[str, Any] | None,
    hw_meta: dict[str, Any],
) -> dict[str, Any]:
    """Resolve {link_bw, link_latency, tp_group_shape} for one candidate.

    See module docstring for the two modes. Returns {} when no information is
    available (caller uses LINK_BW_DEFAULT).
    """
    tp = _candidate_tp(config_spec)

    if fabric_def:
        tiers = fabric_def.get("tiers") or []
        # Load-dependent collective-overhead block (opt-in, fabric-level). Passed
        # through verbatim to the cluster JSON; the trace generator gates it on the
        # TP group crossing socket_size, so attaching it to every candidate is safe
        # (TP<=socket_size candidates get no overhead). None when the fabric omits it.
        co = fabric_def.get("collective_overhead")
        if tiers:
            if tp > 1:
                decomp = _decompose_tp(tp, tiers)
                if decomp is not None:
                    shape, bws, lats = decomp
                    return {
                        "link_bw": bws,
                        "link_latency": lats,
                        "tp_group_shape": shape,
                        "collective_overhead": co,
                    }
                # TP doesn't factor onto the fabric → slowest tier as scalar.
                slow = min(tiers, key=lambda t: float(t["bw_gbs"]))
                return {
                    "link_bw": float(slow["bw_gbs"]),
                    "link_latency": float(slow.get("latency_ns", 0)),
                    "tp_group_shape": None,
                    "collective_overhead": co,
                }
            # TP=1: no intra-group collective; cross-group (PP/DP) traffic
            # crosses the whole box → use the slowest (outermost) tier.
            slow = min(tiers, key=lambda t: float(t["bw_gbs"]))
            return {
                "link_bw": float(slow["bw_gbs"]),
                "link_latency": float(slow.get("latency_ns", 0)),
                "tp_group_shape": None,
                "collective_overhead": co,
            }

    # Catalog scalar fallback.
    bw = _catalog_scalar_bw(config_spec, hw_meta)
    if bw is None:
        return {}
    return {"link_bw": bw, "link_latency": 0, "tp_group_shape": None}
