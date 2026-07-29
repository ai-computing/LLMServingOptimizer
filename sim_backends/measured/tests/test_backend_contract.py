"""M5 tests: MeasuredBackend honors the SimBackend contract and plugs into the
planner's Stage-2 evaluator unmodified (mock oracle fixtures, no hardware)."""
from __future__ import annotations

import json

import pytest
import yaml

from sim_backends import (
    ClusterSpec,
    InstanceSpec,
    NodeSpec,
    ScenarioSpec,
    SimBackend,
    get_backend,
    list_backends,
)

MODEL = "meta-llama/Llama-3.1-8B"


@pytest.fixture()
def oracle_env(tmp_path, monkeypatch):
    """Oracle root with one synthetic A5000 capacity curve + tiny dataset."""
    root = tmp_path / "oracles"
    d = root / "A5000" / MODEL
    d.mkdir(parents=True)
    (d / "tp1.yaml").write_text(yaml.safe_dump({
        "kind": "capacity_curve", "hw": "A5000", "model": MODEL, "tp": 1,
        "idle_w": 20,
        "meta": {"stack": "synthetic", "measured_at": "2026-07-29"},
        "points": [
            {"concurrency": 1, "thr_toks_s": 50, "ttft_ms": 80, "tpot_ms": 20,
             "itl_p99_ms": 25, "avg_w": 180},
            {"concurrency": 16, "thr_toks_s": 400, "ttft_ms": 200, "tpot_ms": 40,
             "itl_p99_ms": 60, "avg_w": 230},
        ]}))
    monkeypatch.setenv("LLMSS_MEASURED_ORACLES", str(root))

    dataset = tmp_path / "wl.jsonl"
    with open(dataset, "w") as f:
        for i in range(20):
            f.write(json.dumps({
                "input_toks": 64, "output_toks": 32,
                "arrival_time_ns": i * 50_000_000,
                "input_tok_ids": [1] * 64}) + "\n")
    return {"root": root, "dataset": dataset, "tmp": tmp_path}


def test_registry_exposes_measured():
    assert "measured" in list_backends()
    b = get_backend("measured")
    assert isinstance(b, SimBackend)
    assert b.available()


def test_run_and_parse_match_backend_schema(oracle_env):
    b = get_backend("measured")
    cluster = ClusterSpec(nodes=[NodeSpec(instances=[
        InstanceSpec(model_name=MODEL, hardware="A5000", num_npus=2, tp_size=1)])])
    cfg = b.build_cluster_config(cluster)
    cfg_path = oracle_env["tmp"] / "cluster.json"
    cfg_path.write_text(json.dumps(cfg))

    out_csv = oracle_env["tmp"] / "out.csv"
    proc = b.run(str(cfg_path), str(out_csv),
                 ScenarioSpec(dataset=str(oracle_env["dataset"]), num_reqs=20))
    assert proc.returncode == 0, proc.stderr

    rows = b.parse_csv(str(out_csv))
    assert len(rows) == 20
    # normalized keys identical to the other adapters' parse_csv output
    expected_keys = {"instance_id", "request_id", "model", "input", "output",
                     "arrival_ns", "end_time_ns", "latency_ns",
                     "queuing_delay_ns", "ttft_ns", "tpot_ns", "itl_ns"}
    assert set(rows[0]) == expected_keys
    assert all(r["output"] == 32 for r in rows)  # pure output semantics

    summary = b.parse_stdout(proc.stdout)
    for key in ("ttft_mean_ms", "tpot_mean_ms", "total_token_throughput",
                "total_energy_kj"):
        assert key in summary, f"missing {key} in stdout summary"


def test_failure_maps_to_returncode(oracle_env):
    b = get_backend("measured")
    cfg_path = oracle_env["tmp"] / "bad.json"
    cfg_path.write_text(json.dumps({"nodes": [{"instances": [{
        "model_name": MODEL, "hardware": "NoSuchHW", "num_npus": 1, "tp_size": 1}]}]}))
    proc = b.run(str(cfg_path), str(oracle_env["tmp"] / "o.csv"),
                 ScenarioSpec(dataset=str(oracle_env["dataset"])))
    assert proc.returncode != 0
    assert "NoSuchHW" in proc.stderr


def test_sim_evaluator_works_with_measured_backend(oracle_env):
    """planner Stage-2 evaluate(backend='measured') end-to-end via subprocess."""
    from planner.sim_evaluator import evaluate
    from planner.types import Metrics

    b = get_backend("measured")
    cluster = ClusterSpec(nodes=[NodeSpec(instances=[
        InstanceSpec(model_name=MODEL, hardware="A5000", num_npus=1, tp_size=1)])])
    cfg_path = oracle_env["tmp"] / "cluster_eval.json"
    cfg_path.write_text(json.dumps(b.build_cluster_config(cluster)))

    cli_args = ["--cluster-config", str(cfg_path),
                "--dataset", str(oracle_env["dataset"]),
                "--num-reqs", "20",
                "--request-routing-policy", "RR"]
    res = evaluate(cli_args, run_id="contract", out_dir=oracle_env["tmp"] / "out",
                   timeout_sec=120, backend="measured")
    assert isinstance(res, Metrics), getattr(res, "reason", None)
    assert res.num_requests == 20
    assert res.ttft_ms > 0 and res.throughput_toks_s > 0
    assert res.energy_j and res.energy_j > 0
    assert res.power_w and res.power_w > 0  # from sim energy, not estimate


def test_config_renderer_supports_measured(tmp_path):
    """planner config_renderer renders upstream-style configs for measured."""
    from planner.config_renderer import render
    from planner.spec_schema import PlannerSpec
    from planner.types import Allocation, Instance

    spec = PlannerSpec.model_validate({
        "model": {"name": MODEL, "fp": 16},
        "workload": {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl",
                     "num_req": 10},
        "topology": {"nodes": [{"id": "node0", "devices": [
            {"name": "A5000", "count": 2, "mem_gb": 24}]}]},
    })
    alloc = Allocation(instances=[Instance(
        node_id="node0", hardware="A5000", model_name=MODEL,
        tp=1, npu_num=2, npu_mem_gb=24)])
    rel_cfg, cli_args = render(alloc, spec, tmp_path, "m5test", backend="measured")
    # rel_cfg is REPO_ROOT-relative; load through REPO_ROOT
    from planner.utils import REPO_ROOT
    cfg = json.loads((REPO_ROOT / rel_cfg).read_text())
    inst = cfg["nodes"][0]["instances"][0]
    assert inst["tp_size"] == 1 and inst["num_npus"] == 1
    assert "--num-reqs" in cli_args
