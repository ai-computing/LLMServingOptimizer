"""Stage 1: structural allocation via CP-SAT (Google OR-Tools).

We enumerate feasible instance *templates* -- (hardware, tp, role) triples whose
memory fits -- and let the solver choose how many of each to deploy, subject to
per-hardware device availability, (when P/D disaggregation is on) a prefill/decode
balance window, and a Max-Flow link-bandwidth constraint for cross-node KV transfer.

Two coarse proxy objectives drive candidate generation: a throughput proxy
(maximize) and a power proxy (minimize). Instead of collapsing them with a fixed
weight, we run an **epsilon-constraint sweep**: maximize the throughput proxy
subject to ``power <= eps_k`` for a grid of ``eps_k`` spanning the feasible power
range. Each level yields a distinct point along the throughput/power trade-off,
so the Top-K candidates are spread across the (proxy) Pareto front -- including
non-convex regions a weighted sum would miss. Stage 2 (simulation) then measures
real metrics and re-ranks.

The proxies are deliberately transparent (documented constants) rather than
precise: the plan delegates accuracy to the Stage-2 simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .graph_model import device_inventory
from .power_profiles import device_active_w, host_overhead_w
from .spec_schema import PlannerSpec
from .types import Allocation, Device, InfeasibleReport, Instance
from .utils import (
    estimate_kv_bytes_per_token,
    estimate_weight_bytes,
    get_logger,
    load_model_config,
    model_precision,
)

log = get_logger("planner.milp")

# Coarse per-hardware relative compute throughput (unitless; A6000 = 1.0 baseline)
# and active power (W). Only used for the Stage-1 proxy ordering; refined by sim.
_HW_REL_THROUGHPUT = {
    "H100": 6.0, "A100": 3.0, "A6000": 1.0, "A40": 1.1,
    "A40x": 1.1, "A5000": 0.8, "RTX3090": 0.9, "RNGD": 1.5, "TPU-v6e-1": 2.0,
}
# Active power now resolves through planner.power_profiles (profiles/power/*.yaml
# with a fallback to the legacy constants that used to live here as
# _HW_ACTIVE_POWER — see power_profiles.LEGACY_ACTIVE_POWER_W).
# KV-cache headroom assumed available per instance, expressed in tokens, for the
# memory feasibility proxy (weights + this many tokens of KV must fit).
_KV_RESERVE_TOKENS = 8192

# --- Max-Flow (P/D KV transfer) constants ---------------------------------
# Coarse cross-node KV *export* rate demanded by one prefill instance, in GB/s
# per unit of (rel_throughput * tp). Prefill produces prompt KV that must be
# shipped to a decode instance; this proxies that rate. Deliberately simple --
# Stage 2 simulation is the source of truth. Only P/D-disaggregated, multi-node
# placements activate the flow constraint (TP traffic is intra-node here).
_KV_EXPORT_PER_UNIT_GBPS = 1.0
# integerization factor for CP-SAT (works in units of 1/_FLOW_SCALE GB/s)
_FLOW_SCALE = 1000

# --- power-min mode constants (design doc §4.1) ----------------------------
# Calibration between the unitless Stage-1 throughput proxy and real token
# rate: tokens/s produced by 1.0 proxy unit (A6000 tp1 serving an 8B model,
# coarse). Overridable per spec via requirements.demand.proxy_toks_per_unit;
# Stage 2 remains the final judge of whether demand is truly met.
_PROXY_TOKS_PER_UNIT = 1000.0
# headroom sweep granularity: level k requires thr >= demand * (1 + k*delta)
_HEADROOM_DELTA = 0.10


@dataclass(frozen=True)
class _Template:
    hardware: str
    node_id: str
    tp: int
    role: str | None  # "prefill" | "decode" | None
    mem_gb: float     # per-device memory
    rel_throughput: float
    power_w: float

    def devices_per_instance(self) -> int:
        return self.tp


def _memory_feasible(
    weight_bytes: float, kv_per_tok: float, tp: int, per_device_mem_gb: float
) -> bool:
    """Per-NPU feasibility: a TP shard of the weights + KV reserve must fit on ONE
    device. With tensor parallelism both weights and KV heads are split across the
    tp ranks, so per-device need ~= (weights + KV_reserve) / tp. This matches the
    simulator's MemoryModel, which rejects a config when a single NPU's shard
    exceeds its memory (verified: 70B needs tp>=4 on 48GB devices)."""
    per_device_mem_bytes = per_device_mem_gb * (1024 ** 3)
    per_device_need = (weight_bytes + kv_per_tok * _KV_RESERVE_TOKENS) / tp
    return per_device_need <= per_device_mem_bytes


def _enumerate_templates(spec: PlannerSpec, inv: list[Device]) -> list[_Template]:
    cfg = load_model_config(spec.model.name)
    # Precision is a property of the checkpoint: int4 weights cannot be served
    # as fp16, so a quantized config overrides spec.model.fp here (which stays
    # the runtime dtype for the simulator CLI). Assuming fp16 for every
    # checkpoint overstated 14B quantized weights by 2-4x and rejected tp1 on
    # 24 GB cards that we had actually measured serving them.
    prec = model_precision(cfg)
    weight_bits = prec.weight_bits if prec.quant_method else spec.model.fp
    kv_bits = prec.kv_bits if prec.quant_method else spec.model.fp
    weight_bytes = estimate_weight_bytes(cfg, weight_bits)
    kv_per_tok = estimate_kv_bytes_per_token(cfg, kv_bits)

    roles: list[str | None] = ["prefill", "decode"] if spec.search_space.pd_disaggregation else [None]
    hw_tp = spec.search_space.hw_tp_choices
    templates: list[_Template] = []
    for dev in inv:
        for tp in spec.search_space.tp_choices:
            if hw_tp is not None and dev.hardware in hw_tp and tp not in hw_tp[dev.hardware]:
                continue  # this hardware cannot run (or be evaluated at) this TP
            if tp > dev.count:
                continue
            if not _memory_feasible(weight_bytes, kv_per_tok, tp, dev.mem_gb):
                continue
            for role in roles:
                templates.append(
                    _Template(
                        hardware=dev.hardware,
                        node_id=dev.node_id,
                        tp=tp,
                        role=role,
                        mem_gb=dev.mem_gb,
                        rel_throughput=_HW_REL_THROUGHPUT.get(dev.hardware, 1.0),
                        power_w=device_active_w(dev.hardware)[0],
                    )
                )
    return templates


def _add_flow_constraints(model, n, templates, graph, inv) -> bool:
    """Add the Max-Flow link-bandwidth constraint for P/D KV transfer.

    Single-commodity flow: KV produced by prefill instances must be routable to
    nodes hosting decode instances without any directed link carrying more than
    its bandwidth. Node-local prefill+decode needs no link. Returns True if a
    constraint was added (i.e. there was cross-node-capable structure).

    Formulation (all quantities integer-scaled by _FLOW_SCALE, unit = GB/s):
      prod[v]      = sum over prefill templates on v of n[i] * export_rate_i
      has_decode[v]= 1 iff any decode instance is placed on v
      f[u,v]       >= 0, <= capacity(u,v)               (edge capacity)
      prod[v] + inflow(v) == consume[v] + outflow(v)    (flow conservation)
      consume[v]   <= M * has_decode[v]                 (sink only where decode is)
    Summing conservation over all nodes forces every produced unit to reach a
    decode-hosting node; a too-small link then makes the model infeasible.
    """
    nodes = list(graph.nodes)
    if len(nodes) < 2 or graph.number_of_edges() == 0:
        return False  # single node / no links => KV transfer is intra-node

    export = {}
    for i, t in enumerate(templates):
        if t.role == "prefill":
            export[i] = max(1, int(round(
                t.rel_throughput * t.tp * _KV_EXPORT_PER_UNIT_GBPS * _FLOW_SCALE)))
    if not export:
        return False  # no prefill instances => nothing to ship

    prod = {v: [] for v in nodes}
    decode_terms = {v: [] for v in nodes}
    for i, t in enumerate(templates):
        if t.role == "prefill" and t.node_id in prod:
            prod[t.node_id].append(export[i] * n[i])
        elif t.role == "decode" and t.node_id in decode_terms:
            decode_terms[t.node_id].append(n[i])

    # Upper bound on total KV production, used to size flow/consume vars.
    # Derive it from device counts (each instance uses >=1 device, so instance
    # count <= total devices). Do NOT introspect a var's proto domain here:
    # IntVar.Proto() returns a dangling reference whose contents are garbage and
    # reading it corrupts the model (segfault at Validate/Solve).
    total_devices = sum(d.count for d in inv)
    ub = max(1, max(export.values()) * max(1, total_devices))

    # edge flow variables, bounded by link capacity
    f = {}
    for u, v, data in graph.edges(data=True):
        cap = max(0, int(round(float(data.get("bw_gbps", 0.0)) * _FLOW_SCALE)))
        f[(u, v)] = model.NewIntVar(0, cap, f"f_{u}_{v}")

    for v in nodes:
        # linear "decode present" gate: consume can be > 0 only if this node has
        # at least one decode instance (dsum >= 1 => consume <= ub, else 0).
        dsum = sum(decode_terms[v]) if decode_terms[v] else 0
        consume = model.NewIntVar(0, ub, f"consume_{v}")
        model.Add(consume <= ub * dsum)

        inflow = [f[(u, v)] for u in nodes if (u, v) in f]
        outflow = [f[(v, w)] for w in nodes if (v, w) in f]
        prod_v = sum(prod[v]) if prod[v] else 0
        model.Add(prod_v + sum(inflow) == consume + sum(outflow))
    return True


def _allocation_from_counts(counts: dict[int, int], templates: list[_Template],
                            spec: PlannerSpec, score: float) -> Allocation:
    instances: list[Instance] = []
    for idx, n in counts.items():
        if n <= 0:
            continue
        t = templates[idx]
        # one Instance object carries all replicas of this template (npu_num = n*tp)
        instances.append(
            Instance(
                node_id=t.node_id,
                hardware=t.hardware,
                model_name=spec.model.name,
                tp=t.tp,
                npu_num=n * t.tp,
                npu_mem_gb=t.mem_gb,
                pd_type=t.role,
            )
        )
    return Allocation(instances=instances, proxy_score=score,
                      meta={"stage": "milp", "counts": dict(counts)})


_THR_SCALE = 1000  # integerization for the throughput proxy


def _build_base_model(spec: PlannerSpec, templates: list[_Template], inv, graph):
    """Create the CP-SAT model with all structural constraints (no objective).

    Returns (model, n, thr_expr, pwr_expr) where n[i] is the instance count for
    template i, thr_expr is the integer throughput proxy, and pwr_expr the integer
    power proxy. Both proxies are linear in n.
    """
    model = cp_model.CpModel()
    n = {}
    for i, t in enumerate(templates):
        hw_total = sum(d.count for d in inv if d.hardware == t.hardware and d.node_id == t.node_id)
        n[i] = model.NewIntVar(0, hw_total // t.tp, f"n_{i}")

    # device-capacity per (node, hardware)
    for dev in inv:
        using = [n[i] * templates[i].tp for i, t in enumerate(templates)
                 if t.hardware == dev.hardware and t.node_id == dev.node_id]
        if using:
            model.Add(sum(using) <= dev.count)

    # at least one instance overall
    model.Add(sum(n.values()) >= 1)

    # P/D balance window (xPyD) + Max-Flow link constraint
    ss = spec.search_space
    if ss.pd_disaggregation:
        prefill_terms = [n[i] for i, t in enumerate(templates) if t.role == "prefill"]
        decode_terms = [n[i] for i, t in enumerate(templates) if t.role == "decode"]
        if prefill_terms:
            model.Add(sum(prefill_terms) >= ss.xpyd_prefill_range[0])
            model.Add(sum(prefill_terms) <= ss.xpyd_prefill_range[1])
        if decode_terms:
            model.Add(sum(decode_terms) >= ss.xpyd_decode_range[0])
            model.Add(sum(decode_terms) <= ss.xpyd_decode_range[1])
        _add_flow_constraints(model, n, templates, graph, inv)

    thr_expr = sum(int(round(t.rel_throughput * t.tp * _THR_SCALE)) * n[i]
                   for i, t in enumerate(templates))
    pwr_expr = sum(int(round(t.power_w * t.tp)) * n[i]
                   for i, t in enumerate(templates))
    return model, n, thr_expr, pwr_expr


def _solve_once(spec, templates, inv, graph, objective, sense, extra=None):
    """Build the base model, apply one objective (+ optional extra constraint),
    solve, and return (counts, thr_value, pwr_value) or None if infeasible."""
    model, n, thr_expr, pwr_expr = _build_base_model(spec, templates, inv, graph)
    if extra is not None:
        extra(model, pwr_expr)
    obj = thr_expr if objective == "thr" else pwr_expr
    model.Maximize(obj) if sense == "max" else model.Minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1, spec.solver.time_limit_sec)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    counts = {i: int(solver.Value(n[i])) for i in n if solver.Value(n[i]) > 0}
    if not counts:
        return None
    thr_val = sum(int(round(templates[i].rel_throughput * templates[i].tp * _THR_SCALE)) * c
                  for i, c in counts.items())
    pwr_val = sum(int(round(templates[i].power_w * templates[i].tp)) * c
                  for i, c in counts.items())
    return counts, thr_val, pwr_val


def _is_power_min(spec: PlannerSpec) -> bool:
    return any(o.metric == "power_w" for o in spec.requirements.objectives)


def _demand_proxy_units(spec: PlannerSpec) -> float:
    """Demand in Stage-1 proxy units (raises when unresolved or absent)."""
    d = spec.requirements.demand
    if d is None:
        raise ValueError("power-min mode requires requirements.demand")
    if d.toks_per_s is None:
        raise ValueError(
            "demand.req_per_s must be resolved to toks_per_s (via the workload "
            "synthesizer) before Stage-1 solving"
        )
    unit = d.proxy_toks_per_unit if d.proxy_toks_per_unit else _PROXY_TOKS_PER_UNIT
    return d.toks_per_s / unit


def _node_host_base_w(spec: PlannerSpec, node_id: str) -> float:
    """Host base power for one node: spec override, else max of the node's
    hardware power-profile host_overhead.base_w values, else 0."""
    node = next((nd for nd in spec.topology.nodes if nd.id == node_id), None)
    if node is None:
        return 0.0
    if node.host_base_w is not None:
        return float(node.host_base_w)
    vals = [host_overhead_w(d.name).base_w for d in node.devices]
    return max(vals) if vals else 0.0


def _solve_power_min_level(spec, templates, inv, graph, thr_floor_scaled: int,
                           host_w: dict[str, float]):
    """min(device power + host base power) s.t. thr_proxy >= floor.

    Returns (counts, thr_val, pwr_val, host_val) or None if infeasible.
    ``used[h]`` is a per-node activation boolean: any instance on node h forces
    used[h]=1 (sum of counts <= device_cap * used); minimization drives it to 0
    on empty hosts.
    """
    model, n, thr_expr, pwr_expr = _build_base_model(spec, templates, inv, graph)
    model.Add(thr_expr >= thr_floor_scaled)

    host_terms = []
    used_by_node: dict[str, object] = {}
    for node_id in sorted({t.node_id for t in templates}):
        base_w = int(round(host_w.get(node_id, 0.0)))
        n_on_node = [n[i] for i, t in enumerate(templates) if t.node_id == node_id]
        if not n_on_node or base_w <= 0:
            continue
        cap = sum(d.count for d in inv if d.node_id == node_id)
        used = model.NewBoolVar(f"used_{node_id}")
        model.Add(sum(n_on_node) <= max(1, cap) * used)
        used_by_node[node_id] = used
        host_terms.append(base_w * used)

    model.Minimize(pwr_expr + sum(host_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1, spec.solver.time_limit_sec)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    counts = {i: int(solver.Value(n[i])) for i in n if solver.Value(n[i]) > 0}
    if not counts:
        return None
    thr_val = sum(int(round(templates[i].rel_throughput * templates[i].tp * _THR_SCALE)) * c
                  for i, c in counts.items())
    pwr_val = sum(int(round(templates[i].power_w * templates[i].tp)) * c
                  for i, c in counts.items())
    active_nodes = {templates[i].node_id for i in counts}
    host_val = sum(int(round(host_w.get(v, 0.0))) for v in active_nodes)
    return counts, thr_val, pwr_val, host_val


def _diagnose_infeasible(spec, templates, inv, graph,
                         demand_units: float) -> InfeasibleReport:
    """Identify which constraint family blocks the power-min model (§6:
    infeasible is a first-class result)."""
    unit = (spec.requirements.demand.proxy_toks_per_unit
            if spec.requirements.demand and spec.requirements.demand.proxy_toks_per_unit
            else _PROXY_TOKS_PER_UNIT)
    if not templates:
        return InfeasibleReport(
            bottleneck="memory",
            detail="no instance template fits: every (hardware, tp) combination "
                   "exceeds per-device memory for this model",
            suggestions=["allow higher tp in search_space.tp_choices",
                         "use devices with more memory",
                         "reduce model size / precision"],
        )
    # Is the structural model solvable at all (demand floor removed)?
    res = _solve_once(spec, templates, inv, graph, "thr", "max")
    if res is not None:
        max_units = res[1] / _THR_SCALE
        if max_units < demand_units:
            factor = demand_units / max_units if max_units > 0 else float("inf")
            return InfeasibleReport(
                bottleneck="demand",
                detail=(f"demand {demand_units * unit:.0f} toks/s exceeds the max "
                        f"achievable {max_units * unit:.0f} toks/s (proxy) on the "
                        f"available inventory"),
                max_achievable_toks_s=max_units * unit,
                suggestions=[
                    f"reduce demand to <= {max_units * unit:.0f} toks/s",
                    f"add ~{max(0, math.ceil((factor - 1) * 100))}% more device capacity",
                ],
            )
        return InfeasibleReport(
            bottleneck="structure",
            detail="demand fits within max achievable throughput yet the joint "
                   "model is UNSAT (interaction between balance/link constraints)",
            max_achievable_toks_s=max_units * unit,
            suggestions=["widen xpyd ranges", "relax link constraints"],
        )
    if spec.search_space.pd_disaggregation and graph.number_of_edges() > 0:
        return InfeasibleReport(
            bottleneck="links",
            detail="structural model UNSAT with P/D flow constraints active; "
                   "cross-node KV link bandwidth is the likely blocker",
            suggestions=["increase inter-node link bandwidth",
                         "disable pd_disaggregation",
                         "co-locate prefill and decode on one node"],
        )
    return InfeasibleReport(
        bottleneck="availability",
        detail="structural model UNSAT: device availability cannot host any "
               "allocation for this search space",
        suggestions=["add devices", "widen tp_choices"],
    )


def _solve_power_min(spec, templates, inv, graph
                     ) -> tuple[list[Allocation], InfeasibleReport | None]:
    """Power-min mode: min total power s.t. throughput proxy >= demand, plus a
    headroom sweep (demand * (1 + k*delta), k = 0..steps-1) for Top-K
    alternatives with spare capacity. Candidates keep sweep order (increasing
    headroom); the final ranking is Stage-2's job."""
    demand_units = _demand_proxy_units(spec)
    top_k = max(1, spec.solver.top_k)
    steps = max(1, spec.solver.pareto_epsilon_steps)
    host_w = {nd.id: _node_host_base_w(spec, nd.id) for nd in spec.topology.nodes}

    seen: set[str] = set()
    allocations: list[Allocation] = []
    for k in range(steps):
        headroom = k * _HEADROOM_DELTA
        floor_scaled = int(math.ceil(demand_units * (1 + headroom) * _THR_SCALE))
        res = _solve_power_min_level(spec, templates, inv, graph, floor_scaled, host_w)
        if res is None:
            if k == 0:
                report = _diagnose_infeasible(spec, templates, inv, graph, demand_units)
                log.warning("Stage-1 power-min infeasible: %s — %s",
                            report.bottleneck, report.detail)
                return [], report
            break  # headroom levels beyond capacity: stop the sweep
        counts, thr_val, pwr_val, host_val = res
        alloc = _allocation_from_counts(counts, templates, spec, thr_val / _THR_SCALE)
        alloc.meta.update({
            "mode": "power_min",
            "headroom_frac": round(headroom, 4),
            "power_proxy_w": pwr_val + host_val,
            "device_power_w": pwr_val,
            "host_power_w": host_val,
            "thr_proxy_units": thr_val / _THR_SCALE,
            "demand_units": demand_units,
        })
        sig = alloc.signature()
        if sig not in seen:
            seen.add(sig)
            allocations.append(alloc)
        if len(allocations) >= top_k:
            break

    # Diversity candidates: the global sweep is SLO-blind, so a config that is
    # the ONLY one meeting a tight latency SLO (e.g. the tp2 variant of a slow
    # GPU) may never be power-minimal at any headroom level. Add the min-power
    # solution restricted to each single (hardware, tp) combo so Stage-2 — the
    # final judge — always gets to see them. These are appended beyond top_k
    # (there are at most a handful of combos).
    combos = sorted({(t.hardware, t.tp) for t in templates})
    if len(combos) > 1:
        floor0 = int(math.ceil(demand_units * _THR_SCALE))
        for hw, tp in combos:
            sub = [t for t in templates if t.hardware == hw and t.tp == tp]
            res = _solve_power_min_level(spec, sub, inv, graph, floor0, host_w)
            if res is None:
                continue  # this combo alone cannot meet the demand
            counts, thr_val, pwr_val, host_val = res
            alloc = _allocation_from_counts(counts, sub, spec, thr_val / _THR_SCALE)
            alloc.meta.update({
                "mode": "power_min",
                "diversity_combo": f"{hw}-tp{tp}",
                "power_proxy_w": pwr_val + host_val,
                "device_power_w": pwr_val,
                "host_power_w": host_val,
                "thr_proxy_units": thr_val / _THR_SCALE,
                "demand_units": demand_units,
            })
            sig = alloc.signature()
            if sig not in seen:
                seen.add(sig)
                allocations.append(alloc)

    log.info("Stage-1 power-min produced %d candidate(s): %d headroom level(s) "
             "+ per-combo diversity (demand %.2f proxy units)",
             len(allocations), steps, demand_units)
    return allocations, None


def solve(spec: PlannerSpec, graph=None) -> list[Allocation]:
    """Return up to ``spec.solver.top_k`` candidate allocations (see
    :func:`solve_with_report`; this wrapper drops the infeasibility report)."""
    return solve_with_report(spec, graph=graph)[0]


def solve_with_report(spec: PlannerSpec, graph=None
                      ) -> tuple[list[Allocation], InfeasibleReport | None]:
    """Stage-1 entry point.

    * default (max-throughput) mode: epsilon-constraint sweep over the proxy
      throughput/power front (unchanged pre-M2 behaviour);
    * power-min mode (objective ``power_w``): min power s.t. demand, with a
      headroom sweep — and a structured :class:`InfeasibleReport` on UNSAT.

    ``graph`` is accepted for API symmetry; if None it is built from the spec.
    """
    if graph is None:
        from .graph_model import build_graph
        graph = build_graph(spec)
    inv = device_inventory(graph)

    templates = _enumerate_templates(spec, inv)
    if not templates:
        log.warning("no feasible instance templates (memory/tp constraints too tight)")
        if _is_power_min(spec):
            return [], _diagnose_infeasible(spec, templates, inv, graph,
                                            _demand_proxy_units(spec))
        return [], None

    if _is_power_min(spec):
        return _solve_power_min(spec, templates, inv, graph)

    top_k = max(1, spec.solver.top_k)
    steps = max(1, spec.solver.pareto_epsilon_steps)

    # Anchors: max-throughput point (upper power bound) and min-power point.
    hi = _solve_once(spec, templates, inv, graph, "thr", "max")
    if hi is None:
        log.warning("Stage-1 infeasible (no allocation satisfies structural constraints)")
        return [], None
    lo = _solve_once(spec, templates, inv, graph, "pwr", "min")
    p_hi = hi[2]                       # power at max throughput
    p_lo = lo[2] if lo else hi[2]      # min feasible power

    # Epsilon grid over [p_lo, p_hi]; each level: maximize throughput s.t. power <= eps.
    seen: set[str] = set()
    allocations: list[Allocation] = []

    def _record(res):
        if res is None:
            return
        counts, thr_val, _ = res
        alloc = _allocation_from_counts(counts, templates, spec, thr_val / _THR_SCALE)
        sig = alloc.signature()
        if sig not in seen:
            seen.add(sig)
            allocations.append(alloc)

    if p_hi <= p_lo:
        # degenerate range (single achievable power level): just take the anchor
        _record(hi)
    else:
        fracs = [1.0] if steps == 1 else [k / (steps - 1) for k in range(steps)]
        for frac in fracs:
            eps = int(math.floor(p_lo + (p_hi - p_lo) * frac))
            res = _solve_once(spec, templates, inv, graph, "thr", "max",
                              extra=lambda m, pwr, e=eps: m.Add(pwr <= e))
            _record(res)

    # Rank by throughput proxy (desc) and cap to top_k, preserving front spread.
    allocations.sort(key=lambda a: a.proxy_score, reverse=True)
    if len(allocations) > top_k:
        # even subsample across the sorted front to keep diversity
        idx = [round(i * (len(allocations) - 1) / (top_k - 1)) for i in range(top_k)] \
            if top_k > 1 else [0]
        allocations = [allocations[j] for j in sorted(set(idx))]

    log.info("Stage-1 produced %d candidate allocation(s) via epsilon-sweep "
             "(power range [%d, %d], %d steps)", len(allocations), p_lo, p_hi, steps)
    return allocations, None
