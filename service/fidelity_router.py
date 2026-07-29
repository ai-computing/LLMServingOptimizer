"""Fidelity Router: choose the Stage-2 evaluation backend per candidate (§5.4).

Cost-accuracy ladder policy:

====================================  =========  ==========
situation                             backend    confidence
====================================  =========  ==========
spec.force_backend set                (forced)   high
NPU present & bucket oracle exists    measured   high (forced — layer-profile
                                                 sim is methodologically unfit
                                                 for bucketed NPUs, §2.1)
oracles exist for every device        measured   high
no oracle & GPU-only                  upstream   medium
no oracle & NPU (RNGD) present        legacy     low (flagged)
NPU, no oracle, no legacy profile     error      —
====================================  =========  ==========
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: hardware treated as statically-compiled/bucketed NPUs (§2.1)
NPU_HARDWARE = {"RNGD", "TPU-v6e-1"}


class RoutingError(Exception):
    """No backend can evaluate this candidate with any stated confidence."""


@dataclass(frozen=True)
class RouteDecision:
    backend: str          # measured | upstream | legacy
    confidence: str       # high | medium | low
    reason: str


def _default_oracle_catalog() -> dict[str, dict[str, list[int]]]:
    from sim_backends import get_backend
    return get_backend("measured").list_hardware()


def _default_legacy_catalog() -> set[tuple[str, str]]:
    from planner.utils import scan_profile_catalog
    return set(scan_profile_catalog())


def _oracle_covers(catalog, hw: str, model: str, tps: list[int]) -> bool:
    have = catalog.get(hw, {}).get(model, [])
    return all(tp in have for tp in tps)


def route(hardware_tps: dict[str, list[int]], model: str,
          force_backend: Optional[str] = None,
          oracle_catalog: Optional[dict] = None,
          legacy_catalog: Optional[set] = None) -> RouteDecision:
    """Pick the evaluation backend for a candidate.

    ``hardware_tps``: {hardware: [tp degrees the candidate uses]}. Catalogs are
    injectable for tests; defaults scan the measured oracle tree and the legacy
    profile tree.
    """
    if not hardware_tps:
        raise ValueError("candidate uses no hardware")
    if force_backend is not None:
        return RouteDecision(backend=force_backend, confidence="high",
                             reason="forced by spec.force_backend")

    oracles = oracle_catalog if oracle_catalog is not None else _default_oracle_catalog()
    npus = sorted(set(hardware_tps) & NPU_HARDWARE)
    covered = {hw: _oracle_covers(oracles, hw, model, tps)
               for hw, tps in hardware_tps.items()}

    if npus:
        if all(covered[hw] for hw in npus) and all(covered.values()):
            return RouteDecision(
                backend="measured", confidence="high",
                reason=f"NPU present ({','.join(npus)}) with bucket oracles; "
                       "layer-profile simulation is methodologically unfit")
        legacy = legacy_catalog if legacy_catalog is not None else _default_legacy_catalog()
        legacy_ok = all((hw, model) in legacy for hw in npus)
        if not legacy_ok:
            raise RoutingError(
                f"NPU {npus} has neither a measured oracle nor a legacy "
                f"profile for '{model}' — cannot evaluate this candidate")
        return RouteDecision(
            backend="legacy", confidence="low",
            reason=f"NPU {','.join(npus)} lacks oracles; falling back to the "
                   "legacy layer-profile path (structurally suspect, §2.1)")

    if all(covered.values()):
        return RouteDecision(backend="measured", confidence="high",
                             reason="oracles cover every device; best accuracy "
                                    "at seconds-per-candidate cost")
    return RouteDecision(backend="upstream", confidence="medium",
                         reason="GPU-only candidate without full oracle "
                                "coverage; upstream simulator")
