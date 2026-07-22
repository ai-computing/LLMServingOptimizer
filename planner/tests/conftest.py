"""Shared fixtures for planner tests."""
from __future__ import annotations

import pytest

from planner.spec_schema import PlannerSpec

_SPEC_DICT = {
    "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
    "workload": {"dataset": "dataset/sharegpt_req100_rate10_llama.jsonl", "num_req": 10},
    "topology": {
        "nodes": [
            {"id": "node0", "devices": [
                {"name": "H100", "count": 2, "mem_gb": 80},
                {"name": "A6000", "count": 4, "mem_gb": 48},
            ]},
        ],
        "links": [{"src": "node0", "dst": "node0", "bandwidth": "200Gbps", "latency": "0.0005ms"}],
    },
    "requirements": {
        "ttft_ms": {"constraint": "<=", "value": 500},
        "objectives": [{"metric": "throughput", "direction": "max", "weight": 1.0}],
    },
    "search_space": {"tp_choices": [1, 2], "batch_tokens_choices": [2048]},
    "solver": {"top_k": 4, "time_limit_sec": 10},
}


@pytest.fixture
def spec() -> PlannerSpec:
    return PlannerSpec.model_validate(_SPEC_DICT)
