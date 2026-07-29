"""Event-driven cluster simulator over measured instance oracles (§5.1).

Composes cluster dynamics — routing, admission/queuing, continuous-batching
approximated as quasi-stationary processor sharing, P/D KV hand-off, power
integration — on top of per-instance capacity curves. Pure stdlib (heapq),
seconds-of-runtime per candidate.

Modeling assumptions (deliberate, documented):
* Between events, per-request decode rate is constant at 1/TPOT(c) where c is
  the instance's current concurrency; rates are recomputed whenever c changes
  (piecewise-constant processor sharing).
* A request's prefill duration is fixed at admission time to TTFT(c') where c'
  is the concurrency including itself; queue wait is added on top.
* Per-request ITL entries are reported as the request's average decode
  interval — cross-request p99 remains meaningful; intra-request jitter is
  below this model's resolution.
* All times are integer nanoseconds (matches the backends' CSV semantics).
"""
from __future__ import annotations

import heapq
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .oracles.base import InstanceOracle

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


# ---------------------------------------------------------------------------
# cluster model
# ---------------------------------------------------------------------------
@dataclass
class InstanceCfg:
    inst_id: int
    node_id: str
    oracle: InstanceOracle
    role: Optional[str] = None       # None | "prefill" | "decode"
    weight: float = 1.0              # routing weight (WEIGHTED policy)
    admit_cap: Optional[int] = None  # max in-flight requests; None -> envelope

    def cap(self) -> int:
        return self.admit_cap if self.admit_cap is not None \
            else self.oracle.envelope.max_concurrency


@dataclass
class HostCfg:
    node_id: str
    base_w: float = 0.0


@dataclass
class IdleDeviceCfg:
    hw: str
    count: int = 0
    idle_w: float = 0.0


@dataclass
class ClusterModel:
    instances: list[InstanceCfg]
    hosts: list[HostCfg] = field(default_factory=list)
    idle_devices: list[IdleDeviceCfg] = field(default_factory=list)
    routing: str = "RR"              # RR | WEIGHTED
    kv_link_gbps: float = 10.0       # P/D KV transfer bandwidth
    kv_bytes_per_token: float = 131072.0  # ~8B-model KV per token, fp16


@dataclass
class WorkloadReq:
    rid: int
    input_toks: int
    output_toks: int
    arrival_ns: int


def load_jsonl_workload(path: str | Path, num_reqs: int = 0) -> list[WorkloadReq]:
    reqs: list[WorkloadReq] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if num_reqs and len(reqs) >= num_reqs:
                break
            d = json.loads(line)
            reqs.append(WorkloadReq(
                rid=i, input_toks=int(d["input_toks"]),
                output_toks=int(d["output_toks"]),
                arrival_ns=int(d["arrival_time_ns"])))
    return sorted(reqs, key=lambda r: r.arrival_ns)


# ---------------------------------------------------------------------------
# simulation internals
# ---------------------------------------------------------------------------
@dataclass
class _Req:
    w: WorkloadReq
    inst_id: Optional[int] = None            # decode-serving instance
    admitted_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    end_ns: Optional[int] = None
    decode_target: int = 0
    decode_done: float = 0.0
    queued_at_ns: Optional[int] = None


class _InstState:
    def __init__(self, cfg: InstanceCfg):
        self.cfg = cfg
        self.prefill: dict[int, tuple[int, _Req]] = {}   # rid -> (deadline_ns, req)
        self.decode: dict[int, _Req] = {}
        self.queue: deque[_Req] = deque()
        self.dispatched: float = 0.0  # routing counter
        self.max_queue_len: int = 0

    @property
    def concurrency(self) -> int:
        return len(self.prefill) + len(self.decode)

    def tpot_ns(self) -> float:
        c = max(1, self.concurrency)
        return self.cfg.oracle.steady_state(c).tpot_ms * NS_PER_MS

    def power_now_w(self) -> float:
        c = self.concurrency
        if c == 0:
            return self.cfg.oracle.idle_w()
        return self.cfg.oracle.power_w(c / max(1, self.cfg.cap()))


@dataclass
class RequestRecord:
    instance_id: int
    request_id: int
    model: str
    input: int
    output: int
    arrival_ns: int
    end_time_ns: int
    latency_ns: int
    queuing_delay_ns: int
    ttft_ns: int
    tpot_ns: int
    itl_ns: list[int]


@dataclass
class SimSummary:
    duration_s: float
    energy_j: float
    avg_power_w: float
    total_output_toks: int
    max_queue_len: int
    num_requests: int


@dataclass
class SimOutput:
    records: list[RequestRecord]
    summary: SimSummary


class EventSimulator:
    def __init__(self, cluster: ClusterModel):
        if not cluster.instances:
            raise ValueError("cluster has no instances")
        self.cluster = cluster
        self.insts = {c.inst_id: _InstState(c) for c in cluster.instances}
        roles = {c.role for c in cluster.instances}
        self.pd_mode = "prefill" in roles and "decode" in roles
        self._route_targets = {
            "arrival": [s for s in self.insts.values()
                        if s.cfg.role in (None, "prefill")] or list(self.insts.values()),
            "decode": [s for s in self.insts.values() if s.cfg.role == "decode"],
        }

    # -- routing ---------------------------------------------------------------
    def _route(self, pool: list[_InstState]) -> _InstState:
        if not pool:
            raise ValueError("no routable instances for this stage")
        if self.cluster.routing == "WEIGHTED":
            # deficit-style deterministic weighted RR: pick the instance whose
            # dispatched/weight ratio is lowest (zero weight is never picked
            # unless every weight is zero)
            weighted = [s for s in pool if s.cfg.weight > 0] or pool
            chosen = min(weighted, key=lambda s: (s.dispatched / s.cfg.weight
                                                  if s.cfg.weight > 0 else float("inf"),
                                                  s.cfg.inst_id))
        else:
            # RR: least dispatched first
            chosen = min(pool, key=lambda s: (s.dispatched, s.cfg.inst_id))
        chosen.dispatched += 1
        return chosen

    # -- main loop ---------------------------------------------------------------
    def run(self, workload: list[WorkloadReq]) -> SimOutput:
        cl = self.cluster
        arrivals = [(w.arrival_ns, w.rid, _Req(w=w)) for w in workload]
        heapq.heapify(arrivals)
        kv_transfers: list[tuple[int, int, _Req]] = []   # (ready_ns, rid, req)

        records: list[RequestRecord] = []
        energy_j = 0.0
        now = 0
        pending = len(arrivals)

        def _advance(to_ns: int):
            """Integrate decode progress + energy over [now, to_ns]."""
            nonlocal energy_j, now
            dt = to_ns - now
            if dt <= 0:
                now = to_ns
                return
            for st in self.insts.values():
                if st.decode:
                    rate = 1.0 / st.tpot_ns()          # tokens per ns
                    for r in st.decode.values():
                        r.decode_done += rate * dt
                energy_j += st.power_now_w() * (dt / NS_PER_S)
            energy_j += sum(h.base_w for h in cl.hosts) * (dt / NS_PER_S)
            energy_j += sum(d.idle_w * d.count for d in cl.idle_devices) * (dt / NS_PER_S)
            now = to_ns

        def _admit(st: _InstState, r: _Req, t: int):
            if r.admitted_ns is None:
                r.admitted_ns = t
            if r.first_token_ns is not None:
                # P/D hand-off: prefill already ran on the prefill instance;
                # enter decode directly
                st.decode[r.w.rid] = r
                return
            c_incl = st.concurrency + 1
            sp = st.cfg.oracle.steady_state(min(c_incl, st.cfg.cap()))
            deadline = t + int(round(sp.ttft_ms * NS_PER_MS))
            st.prefill[r.w.rid] = (deadline, r)

        def _try_admit(st: _InstState, t: int):
            while st.queue and st.concurrency < st.cfg.cap():
                _admit(st, st.queue.popleft(), t)

        def _enqueue(st: _InstState, r: _Req, t: int):
            if st.concurrency < st.cfg.cap():
                _admit(st, r, t)
            else:
                r.queued_at_ns = r.queued_at_ns if r.queued_at_ns is not None else t
                st.queue.append(r)
                st.max_queue_len = max(st.max_queue_len, len(st.queue))

        def _finish(st: _InstState, r: _Req, t: int):
            r.end_ns = t
            w = r.w
            ttft = (r.first_token_ns or t) - w.arrival_ns
            latency = t - w.arrival_ns
            qd = (r.admitted_ns or w.arrival_ns) - w.arrival_ns
            n_itl = max(0, w.output_toks - 1)
            tpot = int(round((t - (r.first_token_ns or t)) / n_itl)) if n_itl else 0
            records.append(RequestRecord(
                instance_id=st.cfg.inst_id, request_id=w.rid,
                model=st.cfg.oracle.model, input=w.input_toks,
                output=w.output_toks, arrival_ns=w.arrival_ns, end_time_ns=t,
                latency_ns=latency, queuing_delay_ns=qd, ttft_ns=ttft,
                tpot_ns=tpot, itl_ns=[tpot] * n_itl))
            _try_admit(st, t)

        while pending > 0 or kv_transfers or any(
                s.prefill or s.decode or s.queue for s in self.insts.values()):
            # next event time
            cands: list[int] = []
            if arrivals:
                cands.append(arrivals[0][0])
            if kv_transfers:
                cands.append(min(k[0] for k in kv_transfers))
            for st in self.insts.values():
                if st.prefill:
                    cands.append(min(d for d, _ in st.prefill.values()))
                if st.decode:
                    tpot = st.tpot_ns()
                    for r in st.decode.values():
                        remaining = max(0.0, r.decode_target - r.decode_done)
                        cands.append(now + int(remaining * tpot) + 1)
            if not cands:
                break
            te = max(now, min(cands))
            _advance(te)

            # 1) arrivals at te
            while arrivals and arrivals[0][0] <= now:
                _, _, r = heapq.heappop(arrivals)
                pending -= 1
                _enqueue(self._route(self._route_targets["arrival"]), r, now)

            # 2) KV transfers completing at te (P/D)
            if kv_transfers:
                ready = [k for k in kv_transfers if k[0] <= now]
                kv_transfers = [k for k in kv_transfers if k[0] > now]
                for _, _, r in ready:
                    _enqueue(self._route(self._route_targets["decode"]), r, now)

            # 3) prefill completions
            for st in self.insts.values():
                done = [rid for rid, (d, _) in st.prefill.items() if d <= now]
                for rid in done:
                    _, r = st.prefill.pop(rid)
                    r.first_token_ns = now
                    if r.w.output_toks <= 1:
                        _finish(st, r, now)
                        continue
                    r.decode_target = r.w.output_toks - 1
                    r.decode_done = 0.0
                    if self.pd_mode and st.cfg.role == "prefill":
                        kv_bytes = r.w.input_toks * self.cluster.kv_bytes_per_token
                        delay = int(kv_bytes / (self.cluster.kv_link_gbps * 1e9)
                                    * NS_PER_S) if self.cluster.kv_link_gbps > 0 else 0
                        kv_transfers.append((now + delay, r.w.rid, r))
                        _try_admit(st, now)
                    else:
                        st.decode[rid] = r

            # 4) decode completions
            for st in self.insts.values():
                done = [rid for rid, r in st.decode.items()
                        if r.decode_done >= r.decode_target - 1e-9]
                for rid in done:
                    _finish(st, st.decode.pop(rid), now)

        duration_s = (max((r.end_time_ns for r in records), default=0) / NS_PER_S)
        total_out = sum(r.output for r in records)
        summary = SimSummary(
            duration_s=duration_s,
            energy_j=energy_j,
            avg_power_w=(energy_j / duration_s) if duration_s > 0 else 0.0,
            total_output_toks=total_out,
            max_queue_len=max((s.max_queue_len for s in self.insts.values()),
                              default=0),
            num_requests=len(records),
        )
        return SimOutput(records=sorted(records, key=lambda r: r.request_id),
                         summary=summary)


# ---------------------------------------------------------------------------
# CSV output — byte-compatible with the backends' result schema
# ---------------------------------------------------------------------------
def write_csv(out: SimOutput, path: str | Path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance id", "request id", "model", "input", "output",
                    "arrival", "end_time", "latency", "queuing_delay",
                    "TTFT", "TPOT", "ITL"])
        for r in out.records:
            w.writerow([r.instance_id, r.request_id, r.model, r.input, r.output,
                        r.arrival_ns, r.end_time_ns, r.latency_ns,
                        r.queuing_delay_ns, r.ttft_ns, r.tpot_ns,
                        json.dumps(r.itl_ns)])


def summary_stdout(out: SimOutput) -> str:
    """Summary block matching the patterns SimBackend.parse_stdout and
    sim_evaluator._parse_energy_from_stdout already scan for."""
    s = out.summary
    recs = out.records
    n = max(1, len(recs))
    mean_ttft = sum(r.ttft_ns for r in recs) / n / NS_PER_MS
    tpots = [r.tpot_ns for r in recs if r.tpot_ns > 0]
    mean_tpot = (sum(tpots) / len(tpots) / NS_PER_MS) if tpots else 0.0
    thr = s.total_output_toks / s.duration_s if s.duration_s > 0 else 0.0
    req_thr = s.num_requests / s.duration_s if s.duration_s > 0 else 0.0
    return "\n".join([
        f"Mean TTFT (ms): {mean_ttft:.3f}",
        f"Mean TPOT (ms): {mean_tpot:.3f}",
        f"Total token throughput (tok/s): {thr:.3f}",
        f"Request throughput (req/s): {req_thr:.3f}",
        f"Total energy consumption (kJ): {s.energy_j / 1000.0:.6f}",
        f"Max queue length: {s.max_queue_len}",
    ]) + "\n"
