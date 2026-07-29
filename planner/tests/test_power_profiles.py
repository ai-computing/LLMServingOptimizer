"""M1 tests: power profile schema, measured interpolation, fallback chain,
and the no-profile solver regression guarantee."""
from __future__ import annotations

import logging

import pytest
import yaml
from pydantic import ValidationError

from planner import power_profiles as pp
from planner.milp_solver import solve
from planner.power_profiles import (
    LEGACY_ACTIVE_POWER_W,
    device_active_w,
    effective_power_w,
    load_power_profile,
)


def _write(root, hw, data):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{hw}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _profile(active_w=300, idle_w=25, mem_gb=48, measured=None):
    d = {"device": {"name": "X", "active_w": active_w, "idle_w": idle_w, "mem_gb": mem_gb}}
    if measured is not None:
        d["measured"] = measured
    return d


# ---- schema validation -----------------------------------------------------

def test_negative_power_rejected(tmp_path):
    _write(tmp_path, "X", _profile(active_w=-5))
    with pytest.raises(ValidationError):
        load_power_profile("X", root=tmp_path)


def test_missing_required_field_rejected(tmp_path):
    bad = {"device": {"name": "X", "active_w": 300, "idle_w": 25}}  # no mem_gb
    _write(tmp_path, "X", bad)
    with pytest.raises(ValidationError):
        load_power_profile("X", root=tmp_path)


def test_measured_load_out_of_range_rejected(tmp_path):
    _write(tmp_path, "X", _profile(measured=[
        {"model": "m", "tp": 1, "load": 1.5, "avg_w": 100}]))
    with pytest.raises(ValidationError):
        load_power_profile("X", root=tmp_path)


# ---- measured interpolation --------------------------------------------------

def test_measured_linear_interpolation(tmp_path):
    _write(tmp_path, "X", _profile(measured=[
        {"model": "m", "tp": 2, "load": 0.5, "avg_w": 100},
        {"model": "m", "tp": 2, "load": 1.0, "avg_w": 200},
    ]))
    w, src = effective_power_w("X", model="m", tp=2, load=0.75, root=tmp_path)
    assert src == "measured"
    assert w == pytest.approx(150.0)


def test_measured_clamped_outside_range(tmp_path):
    _write(tmp_path, "X", _profile(measured=[
        {"model": "m", "tp": 2, "load": 0.5, "avg_w": 100},
        {"model": "m", "tp": 2, "load": 1.0, "avg_w": 200},
    ]))
    assert effective_power_w("X", "m", 2, 0.1, root=tmp_path) == (100.0, "measured")
    assert effective_power_w("X", "m", 2, 1.0, root=tmp_path) == (200.0, "measured")


def test_measured_requires_exact_model_tp_match(tmp_path):
    _write(tmp_path, "X", _profile(active_w=300, measured=[
        {"model": "m", "tp": 4, "load": 0.8, "avg_w": 1180}]))
    # different tp -> falls through to active_w * tp
    w, src = effective_power_w("X", model="m", tp=2, load=0.8, root=tmp_path)
    assert (w, src) == (600.0, "active_w")


# ---- fallback chain ----------------------------------------------------------

def test_fallback_no_measured_uses_active_w(tmp_path):
    _write(tmp_path, "X", _profile(active_w=250))
    assert effective_power_w("X", "m", 2, 0.5, root=tmp_path) == (500.0, "active_w")
    assert device_active_w("X", root=tmp_path) == (250.0, "profile")


def test_fallback_no_file_known_hw_uses_legacy_constant(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(pp.log, "propagate", True)  # planner loggers default to propagate=False
    with caplog.at_level(logging.WARNING, logger="planner.power"):
        w, src = device_active_w("A5000", root=tmp_path)  # empty dir
    assert (w, src) == (float(LEGACY_ACTIVE_POWER_W["A5000"]), "legacy_const")
    assert any("legacy constant" in r.message for r in caplog.records)


def test_fallback_no_file_unknown_hw_uses_default(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(pp.log, "propagate", True)
    with caplog.at_level(logging.WARNING, logger="planner.power"):
        w, src = device_active_w("NoSuchHW", root=tmp_path)
    assert (w, src) == (300.0, "default")
    assert any("default" in r.message for r in caplog.records)


def test_malformed_file_fails_loudly_not_fallback(tmp_path):
    _write(tmp_path, "X", _profile(active_w=-1))
    with pytest.raises(ValidationError):
        device_active_w("X", root=tmp_path)


# ---- shipped profiles + solver regression ------------------------------------

def test_shipped_profiles_match_legacy_constants():
    """Guards the M1 invariant: solver output identical with or without files."""
    for hw, const in LEGACY_ACTIVE_POWER_W.items():
        prof = load_power_profile(hw)  # default root = profiles/power/
        assert prof.device.active_w == const, hw


def test_solver_inputs_unchanged_without_profiles(spec, tmp_path, monkeypatch):
    """Regression (M1 DoD): with *no* profile files at all, the solver's power
    coefficients (and hence its model) are identical to the shipped-profiles
    case. We compare the enumerated templates rather than solve() output
    because CP-SAT's 8-worker search may return different ties among equally
    optimal solutions across runs even with identical inputs."""
    from planner.graph_model import build_graph, device_inventory
    from planner.milp_solver import _enumerate_templates

    inv = device_inventory(build_graph(spec))
    with_profiles = _enumerate_templates(spec, inv)
    monkeypatch.setattr(pp, "DEFAULT_POWER_ROOT", tmp_path / "empty")
    without_profiles = _enumerate_templates(spec, inv)
    assert with_profiles == without_profiles  # frozen dataclasses: full field compare
    assert with_profiles, "spec fixture must yield at least one template"


def test_solve_green_without_profiles(spec, tmp_path, monkeypatch):
    """solve() itself still succeeds with no profile files present."""
    monkeypatch.setattr(pp, "DEFAULT_POWER_ROOT", tmp_path / "empty")
    allocations = solve(spec)
    assert allocations
