from planner import milp_solver
from planner.spec_schema import PlannerSpec


def _mk_spec(**overrides):
    base = {
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": {"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 48}]}]},
        "search_space": {"tp_choices": [1, 2], "batch_tokens_choices": [2048]},
        "solver": {"top_k": 3, "time_limit_sec": 10},
    }
    base.update(overrides)
    return PlannerSpec.model_validate(base)


def test_solve_produces_candidates(spec):
    allocs = milp_solver.solve(spec)
    assert 1 <= len(allocs) <= spec.solver.top_k
    for a in allocs:
        assert a.instances
        # device usage never exceeds inventory
        assert a.total_devices() <= 6  # 2 H100 + 4 A6000


def test_candidates_are_distinct():
    allocs = milp_solver.solve(_mk_spec())
    sigs = {a.signature() for a in allocs}
    assert len(sigs) == len(allocs)  # no-good cuts => all distinct


def test_device_capacity_respected():
    spec = _mk_spec(
        topology={"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 48}]}]},
    )
    for a in milp_solver.solve(spec):
        used = sum(i.npu_num for i in a.instances if i.hardware == "A6000")
        assert used <= 4


def test_tp_is_valid_divisor():
    for a in milp_solver.solve(_mk_spec()):
        for inst in a.instances:
            assert inst.tp <= inst.npu_num
            assert inst.npu_num % inst.tp == 0


def test_infeasible_when_memory_too_small():
    # 1 GB devices cannot hold an 8B model
    spec = _mk_spec(
        topology={"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 1}]}]},
    )
    assert milp_solver.solve(spec) == []


# --- Max-Flow link-bandwidth constraint (P/D KV transfer) ------------------
def _pd_spec(link_bw, count=1):
    """Two single-device nodes; forced cross-node P/D unless count allows co-location."""
    return PlannerSpec.model_validate({
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": {
            "nodes": [
                {"id": "n0", "devices": [{"name": "A6000", "count": count, "mem_gb": 48}]},
                {"id": "n1", "devices": [{"name": "A6000", "count": count, "mem_gb": 48}]},
            ],
            "links": [{"src": "n0", "dst": "n1", "bandwidth": link_bw, "latency": "0.0005ms"}],
        },
        "search_space": {
            "pd_disaggregation": True, "tp_choices": [1],
            "xpyd_prefill_range": [1, 1], "xpyd_decode_range": [1, 1],
            "batch_tokens_choices": [2048],
        },
        "solver": {"top_k": 2, "time_limit_sec": 10},
    })


def test_flow_feasible_with_wide_link():
    # 200Gbps = 25 GB/s >> ~1 GB/s KV demand -> cross-node P/D is feasible
    allocs = milp_solver.solve(_pd_spec("200Gbps"))
    assert allocs
    roles = {i.pd_type for a in allocs for i in a.instances}
    assert roles == {"prefill", "decode"}


def test_flow_infeasible_with_narrow_link():
    # 1Gbps = 0.125 GB/s < ~1 GB/s KV demand, single device per node forces
    # cross-node placement -> no feasible allocation
    assert milp_solver.solve(_pd_spec("1Gbps")) == []


def test_narrow_link_ok_when_colocation_possible():
    # 2 devices per node let the solver put prefill+decode on the same node,
    # avoiding the (narrow) link entirely -> feasible again
    allocs = milp_solver.solve(_pd_spec("1Gbps", count=2))
    assert allocs


# --- epsilon-constraint Pareto sweep --------------------------------------
def _hetero_spec(top_k=5, steps=5):
    return PlannerSpec.model_validate({
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": {"nodes": [{"id": "n0", "devices": [
            {"name": "H100", "count": 2, "mem_gb": 80},
            {"name": "A6000", "count": 4, "mem_gb": 48},
            {"name": "A5000", "count": 2, "mem_gb": 24},
        ]}]},
        "search_space": {"tp_choices": [1, 2, 4], "batch_tokens_choices": [2048]},
        "solver": {"top_k": top_k, "time_limit_sec": 30, "pareto_epsilon_steps": steps},
    })


def test_epsilon_sweep_spans_front():
    allocs = milp_solver.solve(_hetero_spec(top_k=5, steps=5))
    # distinct candidates spread across the throughput/power trade-off
    sigs = {a.signature() for a in allocs}
    assert len(sigs) == len(allocs)
    assert len(allocs) >= 3
    scores = [a.proxy_score for a in allocs]
    assert max(scores) > min(scores)          # not all the same point
    assert max(scores) >= 3 * max(min(scores), 1e-9)  # meaningful spread


def test_epsilon_respects_top_k_cap():
    allocs = milp_solver.solve(_hetero_spec(top_k=3, steps=6))
    assert len(allocs) <= 3


def test_epsilon_single_step_returns_max_throughput():
    one = milp_solver.solve(_hetero_spec(top_k=5, steps=1))
    assert len(one) == 1
    # the single point must be the global throughput-proxy maximizer
    many = milp_solver.solve(_hetero_spec(top_k=8, steps=8))
    assert one[0].proxy_score == max(a.proxy_score for a in many)
