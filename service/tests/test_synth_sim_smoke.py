"""M3 sim smoke: a synthesized jsonl actually runs on the upstream backend."""
from __future__ import annotations

import pytest

from service.workload_synth import ScaleSpec, synthesize

pytestmark = pytest.mark.sim


def _upstream_available():
    try:
        from sim_backends import get_backend
        return get_backend("upstream").available()
    except Exception:
        return False


@pytest.mark.skipif(not _upstream_available(), reason="upstream backend not built")
def test_synth_jsonl_runs_on_upstream(tmp_path):
    from sim_backends import ClusterSpec, InstanceSpec, NodeSpec, ScenarioSpec, get_backend

    b = get_backend("upstream")
    res = synthesize(ScaleSpec(req_per_s=5, duration_s=2, preset="chat", seed=0),
                     tmp_path / "synth.jsonl")

    cluster = ClusterSpec(nodes=[NodeSpec(instances=[InstanceSpec(
        model_name="meta-llama/Llama-3.1-8B", hardware="A5000",
        num_npus=1, tp_size=1,
        npu_mem={"mem_size": 24, "mem_bw": 768, "mem_latency": 0})])])
    cfg = b.build_cluster_config(cluster)
    cfg_path = tmp_path / "cluster.json"
    import json
    cfg_path.write_text(json.dumps(cfg))

    out_csv = tmp_path / "out.csv"
    scenario = ScenarioSpec(dataset=res.path, num_reqs=10, dtype="bf16")
    proc = b.run(str(cfg_path), str(out_csv), scenario, run_id="synth_smoke",
                 timeout=900)
    assert proc.returncode == 0, proc.stderr[-800:]
    rows = b.parse_csv(str(out_csv))
    assert len(rows) == 10
    assert all(r["output"] > 0 for r in rows)
