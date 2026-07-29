"""M7 tests: three-way comparison report — diff math, table generation, and a
live measured-vs-fixture completion check (fixtures only; no hardware)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "compare_backends", REPO_ROOT / "scripts" / "compare_backends.py")
cb = importlib.util.module_from_spec(_spec)
sys.modules["compare_backends"] = cb
_spec.loader.exec_module(cb)

NS_MS = 1_000_000


def _rec(ttft_ms, tpot_ms, out=100, arrival_ns=0):
    lat = int(ttft_ms * NS_MS + tpot_ms * NS_MS * (out - 1))
    return {"ttft_ns": int(ttft_ms * NS_MS), "tpot_ns": int(tpot_ms * NS_MS),
            "latency_ns": lat, "output": out, "arrival_ns": arrival_ns,
            "end_time_ns": arrival_ns + lat}


def test_aggregate_and_diff_math():
    vllm = cb.aggregate([_rec(100, 20), _rec(200, 40, arrival_ns=10 * NS_MS)])
    sim = cb.aggregate([_rec(120, 24), _rec(240, 48, arrival_ns=10 * NS_MS)])
    results = {"vllm": vllm, "upstream": sim}
    table = cb.diff_table(results)
    assert "+20.0%" in table          # ttft mean 150 -> 180
    errs = cb.error_summary(results)
    # every latency metric inflated by exactly 20%; throughput differs too
    assert errs["upstream"] == pytest.approx(20.0, abs=2.0)


def test_error_summary_orders_sources():
    vllm = cb.aggregate([_rec(100, 20)])
    good = cb.aggregate([_rec(105, 21)])   # ~5% off
    bad = cb.aggregate([_rec(200, 40)])    # ~100% off
    errs = cb.error_summary({"vllm": vllm, "measured": good, "upstream": bad})
    assert errs["measured"] < errs["upstream"]


def test_load_vllm_jsonl(tmp_path):
    p = tmp_path / "v.jsonl"
    p.write_text(json.dumps({
        "req_idx": 0, "input_toks": 85, "output_toks": 416,
        "actual_output_toks": 400, "arrival_time_ns": 1000,
        "ttft_ns": 87129772, "tpot_ns": 88854320,
        "total_latency_ns": 36961689135}) + "\n")
    recs = cb.load_vllm_jsonl(p)
    assert recs[0]["output"] == 400            # actual_output preferred
    assert recs[0]["end_time_ns"] == 1000 + 36961689135


def test_end_to_end_report_with_live_measured_run(tmp_path, monkeypatch):
    """CLI completes: vllm fixture + live measured backend run -> md report."""
    # oracle + config for the measured run
    oracle_root = tmp_path / "oracles"
    d = oracle_root / "A40" / "m"
    d.mkdir(parents=True)
    (d / "tp1.yaml").write_text(yaml.safe_dump({
        "kind": "capacity_curve", "hw": "A40", "model": "m", "tp": 1,
        "idle_w": 25, "points": [
            {"concurrency": 1, "thr_toks_s": 100, "ttft_ms": 100, "tpot_ms": 20,
             "itl_p99_ms": 25, "avg_w": 200},
            {"concurrency": 8, "thr_toks_s": 500, "ttft_ms": 150, "tpot_ms": 25,
             "itl_p99_ms": 35, "avg_w": 250}]}))
    monkeypatch.setenv("LLMSS_MEASURED_ORACLES", str(oracle_root))

    cfg = tmp_path / "cluster.json"
    cfg.write_text(json.dumps({"nodes": [{"instances": [
        {"model_name": "m", "hardware": "A40", "num_npus": 1, "tp_size": 1}]}]}))

    dataset = tmp_path / "wl.jsonl"
    with open(dataset, "w") as f:
        for i in range(10):
            f.write(json.dumps({"input_toks": 64, "output_toks": 32,
                                "arrival_time_ns": i * 100_000_000,
                                "input_tok_ids": [1] * 64}) + "\n")

    vllm_fixture = tmp_path / "vllm.jsonl"
    with open(vllm_fixture, "w") as f:
        for i in range(10):
            f.write(json.dumps({
                "req_idx": i, "input_toks": 64, "output_toks": 32,
                "arrival_time_ns": i * 100_000_000,
                "ttft_ns": 110 * NS_MS, "tpot_ns": 21 * NS_MS,
                "total_latency_ns": int(110 * NS_MS + 31 * 21 * NS_MS)}) + "\n")

    out_md = tmp_path / "report.md"
    rc = cb.main(["--vllm-jsonl", str(vllm_fixture),
                  "--measured-config", str(cfg),
                  "--dataset", str(dataset), "--num-reqs", "10",
                  "--out", str(out_md)])
    assert rc == 0
    text = out_md.read_text()
    assert "measured" in text and "Mean |diff%|" in text
