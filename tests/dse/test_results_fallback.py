"""api_get_results fallback: completed candidates show even when the ranked
*.json files were never persisted (e.g. a cancelled / interrupted job).

Regression for the bug where Cancel deleted the job dir and Open Results showed
nothing; now Cancel preserves the job and finished candidates are ranked on read.
"""
import asyncio
import json

import pytest

from webapp.dse.server import routes


def _write_job(job_dir, configs, *, state="cancelled"):
    job_dir.mkdir(parents=True, exist_ok=True)
    # Minimal valid JobSpec (spec.json) — needed by the fallback's rank step.
    spec = {
        "resource_pool": {"items": [{"hw": "A40", "min": 1, "max": 8}]},
        "model": {"name": "meta-llama/Llama-3.1-70B", "fp": 16},
        "workload": {"num_req": 10, "dataset": "dataset/example_trace.jsonl"},
    }
    (job_dir / "spec.json").write_text(json.dumps(spec))
    (job_dir / "candidates.json").write_text(json.dumps([
        {"candidate_id": f"c{i}", "label": lbl,
         "hw_distribution": {"A40": 8}, "parallelism": {"tp": 8, "pp": 1, "dp": 1},
         "pd_layout": "—"}
        for i, lbl in enumerate(configs)
    ]))
    (job_dir / "status.json").write_text(json.dumps({
        "sweep_id": job_dir.name, "state": state, "configs": configs,
    }))
    (job_dir / "configs").mkdir(exist_ok=True)


def _done_metrics(ttft, tpot, tput):
    return {"p99_ttft_ms": ttft, "p99_tpot_ms": tpot, "total_token_tp": tput,
            "total_energy_wh": 1.0, "tok_per_wh": tput}


@pytest.fixture
def patched_root(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "DSE_ROOT", tmp_path)
    return tmp_path


def test_cancelled_job_shows_done_candidates(patched_root):
    job = patched_root / "job1"
    _write_job(job, {
        "A40x8_tp8_a": {"state": "done", "elapsed_s": 3.0, "metrics": _done_metrics(150, 90, 600)},
        "A40x8_tp8_b": {"state": "done", "elapsed_s": 4.0, "metrics": _done_metrics(160, 95, 550)},
        "A40x8_tp8_c": {"state": "cancelled", "elapsed_s": 0.0},
        "A40x8_tp8_d": {"state": "queued"},  # never ran — must not appear as a result
    })

    res = asyncio.run(routes.api_get_results("job1"))
    body = json.loads(res.body)

    # done + cancelled are results; queued is not.
    states = sorted(r["state"] for r in body["all_candidates"])
    assert states == ["cancelled", "done", "done"]
    # top_n / pareto contain only the done candidates.
    assert len(body["top_n"]) >= 1
    assert all(r["state"] == "done" for r in body["top_n"])
    assert all(r["state"] == "done" for r in body["pareto"])


def test_persisted_ranked_files_take_precedence(patched_root):
    job = patched_root / "job2"
    _write_job(job, {"A40x8_tp8_a": {"state": "done", "metrics": _done_metrics(150, 90, 600)}})
    # A real ranked file present → fallback must NOT override it.
    (job / "all_candidates.json").write_text(json.dumps(
        [{"candidate_id": "c0", "label": "PERSISTED", "state": "done",
          "elapsed_s": 1.0, "metrics": {}}]
    ))
    body = json.loads(asyncio.run(routes.api_get_results("job2")).body)
    assert [r["label"] for r in body["all_candidates"]] == ["PERSISTED"]


def test_no_done_candidates_returns_empty(patched_root):
    job = patched_root / "job3"
    _write_job(job, {
        "A40x8_tp8_a": {"state": "failed", "error": "boom"},
        "A40x8_tp8_b": {"state": "cancelled"},
    })
    body = json.loads(asyncio.run(routes.api_get_results("job3")).body)
    assert body["all_candidates"] == []
    assert body["top_n"] == []
