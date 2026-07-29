"""M9 tests: fidelity-router policy table — one case per rule + error case."""
from __future__ import annotations

import pytest

from service.fidelity_router import RouteDecision, RoutingError, route

MODEL = "meta-llama/Llama-3.1-8B"

ORACLES_ALL = {"A40": {MODEL: [1, 2]}, "RNGD": {MODEL: [1]}}
ORACLES_GPU_ONLY = {"A40": {MODEL: [1, 2]}}
ORACLES_NONE: dict = {}
LEGACY_WITH_RNGD = {("RNGD", MODEL), ("A40", MODEL)}
LEGACY_EMPTY: set = set()


def test_forced_backend_overrides_everything():
    d = route({"A40": [1]}, MODEL, force_backend="legacy",
              oracle_catalog=ORACLES_NONE, legacy_catalog=LEGACY_EMPTY)
    assert d == RouteDecision("legacy", "high", "forced by spec.force_backend")


def test_npu_with_bucket_oracle_forces_measured():
    d = route({"RNGD": [1], "A40": [1]}, MODEL, oracle_catalog=ORACLES_ALL)
    assert d.backend == "measured" and d.confidence == "high"
    assert "NPU" in d.reason


def test_full_oracle_coverage_prefers_measured():
    d = route({"A40": [1, 2]}, MODEL, oracle_catalog=ORACLES_GPU_ONLY)
    assert d.backend == "measured" and d.confidence == "high"


def test_gpu_only_without_oracles_goes_upstream():
    d = route({"A40": [1], "H100": [2]}, MODEL, oracle_catalog=ORACLES_NONE)
    assert d.backend == "upstream" and d.confidence == "medium"


def test_partial_oracle_coverage_still_upstream():
    # A40 covered but only for tp1; candidate needs tp4 -> not covered
    d = route({"A40": [4]}, MODEL, oracle_catalog=ORACLES_GPU_ONLY)
    assert d.backend == "upstream"


def test_npu_without_oracle_falls_back_to_legacy_low_confidence():
    d = route({"RNGD": [1]}, MODEL, oracle_catalog=ORACLES_NONE,
              legacy_catalog=LEGACY_WITH_RNGD)
    assert d.backend == "legacy" and d.confidence == "low"


def test_npu_without_oracle_or_legacy_profile_is_an_error():
    with pytest.raises(RoutingError, match="neither"):
        route({"RNGD": [1]}, MODEL, oracle_catalog=ORACLES_NONE,
              legacy_catalog=LEGACY_EMPTY)


def test_mixed_npu_gpu_where_only_npu_covered_falls_back():
    # RNGD oracle exists but the A40 side is uncovered -> not all covered ->
    # legacy fallback (with profile) rather than a half-measured evaluation
    oracles = {"RNGD": {MODEL: [1]}}
    d = route({"RNGD": [1], "A40": [1]}, MODEL, oracle_catalog=oracles,
              legacy_catalog=LEGACY_WITH_RNGD)
    assert d.backend == "legacy" and d.confidence == "low"


def test_empty_candidate_rejected():
    with pytest.raises(ValueError):
        route({}, MODEL, oracle_catalog=ORACLES_NONE)
