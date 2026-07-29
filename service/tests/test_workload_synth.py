"""M3 tests: workload synthesizer — rate accuracy, demand conversion,
determinism, and dataset schema compliance for both backends."""
from __future__ import annotations

import hashlib
import json

import pytest

from service.presets import PRESETS, LengthDist, get_preset
from service.workload_synth import (
    ScaleSpec,
    demand_toks_per_s,
    synthesize,
)


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_poisson_arrival_rate_within_5pct(tmp_path):
    spec = ScaleSpec(req_per_s=10, duration_s=200, preset="chat", seed=42)
    res = synthesize(spec, tmp_path / "w.jsonl")
    assert res.stats["actual_req_per_s"] == pytest.approx(10, rel=0.05)


def test_uniform_arrival_rate_exact(tmp_path):
    spec = ScaleSpec(req_per_s=10, duration_s=50, preset="chat", arrival="uniform", seed=1)
    res = synthesize(spec, tmp_path / "w.jsonl")
    assert res.num_requests == 500
    rows = _read_rows(res.path)
    gaps = {rows[i + 1]["arrival_time_ns"] - rows[i]["arrival_time_ns"]
            for i in range(len(rows) - 1)}
    assert gaps == {100_000_000}  # exactly 0.1 s


def test_demand_toks_per_s_matches_file_within_1pct(tmp_path):
    spec = ScaleSpec(req_per_s=5, duration_s=100, preset="chat", seed=7)
    res = synthesize(spec, tmp_path / "w.jsonl")
    rows = _read_rows(res.path)
    file_rate = sum(r["input_toks"] + r["output_toks"] for r in rows) / spec.duration_s
    assert res.demand_toks_per_s == pytest.approx(file_rate, rel=0.01)


def test_seeded_determinism_same_file_hash(tmp_path):
    spec = ScaleSpec(req_per_s=5, duration_s=20, preset="agentic", seed=123)
    h = []
    for name in ("a.jsonl", "b.jsonl"):
        res = synthesize(spec, tmp_path / name)
        h.append(hashlib.sha256(open(res.path, "rb").read()).hexdigest())
    assert h[0] == h[1]
    res2 = synthesize(ScaleSpec(req_per_s=5, duration_s=20, preset="agentic", seed=124),
                      tmp_path / "c.jsonl")
    assert hashlib.sha256(open(res2.path, "rb").read()).hexdigest() != h[0]


def test_dataset_schema_required_by_both_backends(tmp_path):
    """dataset/README.md field contract: input_toks/output_toks/arrival_time_ns/
    input_tok_ids, arrivals nondecreasing, id-list length == input_toks."""
    spec = ScaleSpec(req_per_s=8, duration_s=10, preset="summarize", seed=3)
    res = synthesize(spec, tmp_path / "w.jsonl")
    rows = _read_rows(res.path)
    assert rows
    prev = -1
    for r in rows:
        assert isinstance(r["input_toks"], int) and r["input_toks"] >= 1
        assert isinstance(r["output_toks"], int) and r["output_toks"] >= 1
        assert isinstance(r["arrival_time_ns"], int) and r["arrival_time_ns"] >= 0
        assert len(r["input_tok_ids"]) == r["input_toks"]
        assert len(r["output_tok_ids"]) == r["output_toks"]
        assert r["arrival_time_ns"] >= prev
        prev = r["arrival_time_ns"]


def test_pulse_mode_is_bursty(tmp_path):
    spec = ScaleSpec(req_per_s=10, duration_s=60, preset="chat",
                     arrival="pulse", pulse_bursts=3, seed=5)
    res = synthesize(spec, tmp_path / "w.jsonl")
    rows = _read_rows(res.path)
    ts = [r["arrival_time_ns"] / 1e9 for r in rows]
    # bursts occupy pulse_burst_frac=0.2 of the span; a large idle gap must exist
    max_gap = max(b - a for a, b in zip(ts, ts[1:]))
    assert max_gap > 5.0  # idle stretch between bursts (uniform would be ~0.1s)


def test_demand_conversion_analytic():
    p = get_preset("chat")
    expected = 5 * (p.input_dist.mean + p.output_dist.mean)
    assert demand_toks_per_s(5, preset="chat") == pytest.approx(expected)


def test_explicit_dist_overrides_preset(tmp_path):
    fixed = LengthDist(mean=100, p90=101)  # nearly-constant lengths
    spec = ScaleSpec(req_per_s=5, duration_s=10, preset="chat",
                     input_dist=fixed, output_dist=fixed, seed=1)
    res = synthesize(spec, tmp_path / "w.jsonl")
    assert 95 <= res.mean_input_toks <= 105


def test_length_stats_track_preset_means(tmp_path):
    for name, p in PRESETS.items():
        res = synthesize(ScaleSpec(req_per_s=20, duration_s=100, preset=name, seed=11),
                         tmp_path / f"{name}.jsonl")
        assert res.mean_input_toks == pytest.approx(p.input_dist.mean, rel=0.15), name
        # output means sit closer to the 2048 clamp; allow wider tolerance
        assert res.mean_output_toks == pytest.approx(p.output_dist.mean, rel=0.20), name


def test_invalid_specs_rejected(tmp_path):
    with pytest.raises(ValueError):
        synthesize(ScaleSpec(req_per_s=0, duration_s=10, preset="chat"), tmp_path / "x")
    with pytest.raises(ValueError):
        synthesize(ScaleSpec(req_per_s=1, duration_s=10), tmp_path / "x")  # no dists
    with pytest.raises(KeyError):
        synthesize(ScaleSpec(req_per_s=1, duration_s=10, preset="nope"), tmp_path / "x")
