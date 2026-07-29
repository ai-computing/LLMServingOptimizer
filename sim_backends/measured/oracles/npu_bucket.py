"""LENS-style bucket oracle for statically-compiled NPUs (design doc §5.2).

NPUs with static compilation + bucketing (RNGD, TPU, Inferentia) quantize
latency into a step function over KV-length buckets: measuring TTFT_b/TBT_b
once per bucket (2|B| E2E runs) captures compiler and bucketing effects
exactly (LENS, arXiv 2606.18042 — 2.15% observed error vs up to 493% for
layer-sum simulation).

Latency composition for one request (input l_in, output l_out):

    T = TTFT_bucket(l_in) + sum_j TBT_bucket(kv_j),   kv_j = l_in + j

split into segments wherever kv_j crosses a bucket boundary. Batch expansion
(LENS): prefill cost is per-request additive; batched decode steps run at the
TBT of the *maximum* KV position in the batch.

YAML:

.. code-block:: yaml

    kind: npu_bucket
    hw: RNGD
    model: meta-llama/Llama-3.1-8B
    tp: 1
    idle_w: 15
    max_concurrency: 16
    typical_input_toks: 512     # used only for the steady-state approximation
    typical_output_toks: 256
    meta: {stack: "furiosa-sdk-1.x", measured_at: "..."}
    buckets:
      - {size: 4096, ttft_ms: 100, tbt_ms: 10, avg_w: 180}
      - {size: 8192, ttft_ms: 120, tbt_ms: 14, avg_w: 200}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .base import Envelope, OracleMeta, SteadyPoint


class InfeasibleError(ValueError):
    """Request cannot run on this oracle (e.g. input exceeds the max bucket)."""


@dataclass(frozen=True)
class Bucket:
    size: int          # max KV length this compiled bucket covers
    ttft_ms: float     # prefill E2E latency measured at this bucket
    tbt_ms: float      # time-between-tokens measured at this bucket
    avg_w: float = 0.0


class NpuBucketOracle:
    def __init__(self, hw: str, model: str, tp: int, buckets: list[Bucket],
                 idle_w: float = 0.0, max_concurrency: int = 16,
                 typical_input_toks: int | None = None,
                 typical_output_toks: int | None = None,
                 meta: OracleMeta | None = None):
        if not buckets:
            raise ValueError("npu_bucket oracle needs at least one bucket")
        self.hw = hw
        self.model = model
        self.tp = tp
        self.buckets = sorted(buckets, key=lambda b: b.size)
        if len({b.size for b in self.buckets}) != len(self.buckets):
            raise ValueError("duplicate bucket sizes")
        self._idle_w = float(idle_w)
        self.meta = meta or OracleMeta()
        max_kv = self.buckets[-1].size
        self.typical_input_toks = typical_input_toks or max_kv // 4
        self.typical_output_toks = typical_output_toks or max_kv // 8
        self.envelope = Envelope(max_concurrency=max_concurrency,
                                 max_input_toks=max_kv, max_output_toks=max_kv)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NpuBucketOracle":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        for key in ("hw", "model", "tp", "buckets"):
            if key not in d:
                raise ValueError(f"npu_bucket YAML missing '{key}': {path}")
        return cls(
            hw=d["hw"], model=d["model"], tp=int(d["tp"]),
            buckets=[Bucket(**b) for b in d["buckets"]],
            idle_w=float(d.get("idle_w", 0.0)),
            max_concurrency=int(d.get("max_concurrency", 16)),
            typical_input_toks=d.get("typical_input_toks"),
            typical_output_toks=d.get("typical_output_toks"),
            meta=OracleMeta(**d.get("meta", {})) if d.get("meta") else OracleMeta(),
        )

    # -- bucket lookups -----------------------------------------------------
    def _bucket_for(self, kv_len: int) -> Bucket:
        for b in self.buckets:
            if kv_len <= b.size:
                return b
        raise InfeasibleError(
            f"KV length {kv_len} exceeds the largest compiled bucket "
            f"({self.buckets[-1].size}) on {self.hw}/{self.model}/tp{self.tp}")

    def ttft_ms(self, input_toks: int) -> float:
        return self._bucket_for(input_toks).ttft_ms

    def tbt_ms(self, kv_len: int) -> float:
        return self._bucket_for(kv_len).tbt_ms

    # -- LENS composition -----------------------------------------------------
    def request_latency_ms(self, input_toks: int, output_toks: int) -> float:
        """Single-request E2E latency: TTFT at the input bucket plus per-token
        TBT segments split at every bucket boundary the KV position crosses."""
        if output_toks < 1:
            raise ValueError("output_toks must be >= 1")
        total = self.ttft_ms(input_toks)
        # decode tokens j = 1..output_toks-1 run at kv position input_toks + j
        remaining = output_toks - 1
        j = 1
        while remaining > 0:
            b = self._bucket_for(input_toks + j)
            # tokens until this bucket is exhausted (kv position <= b.size)
            span = min(remaining, b.size - (input_toks + j) + 1)
            total += span * b.tbt_ms
            remaining -= span
            j += span
        return total

    def batch_decode_tbt_ms(self, kv_positions: list[int]) -> float:
        """Batched decode step cost = TBT of the max KV position in the batch
        (LENS batch expansion rule)."""
        if not kv_positions:
            raise ValueError("empty batch")
        return self.tbt_ms(max(kv_positions))

    def batch_prefill_ms(self, input_lens: list[int]) -> float:
        """Prefill cost is per-request additive on bucketed NPUs."""
        return sum(self.ttft_ms(l) for l in input_lens)

    # -- InstanceOracle protocol (event-sim integration) --------------------------
    def steady_state(self, concurrency: int) -> SteadyPoint:
        """Steady-state approximation at ``typical_*`` lengths: a batch of c
        requests decodes at the TBT of its max KV position; prefill latencies
        add up across the batch."""
        c = max(1, min(concurrency, self.envelope.max_concurrency))
        kv_typ = min(self.typical_input_toks + self.typical_output_toks,
                     self.buckets[-1].size)
        tbt = self.tbt_ms(kv_typ)
        ttft = self.ttft_ms(self.typical_input_toks) * c  # additive prefill
        thr = c * 1000.0 / tbt if tbt > 0 else 0.0
        b = self._bucket_for(kv_typ)
        return SteadyPoint(concurrency=c, thr_toks_s=thr, ttft_ms=ttft,
                           tpot_ms=tbt, itl_p99_ms=tbt, avg_w=b.avg_w)

    def power_w(self, load: float) -> float:
        load = max(0.0, min(1.0, load))
        if load == 0.0:
            return self._idle_w
        idx = min(len(self.buckets) - 1, int(load * len(self.buckets)))
        return self.buckets[idx].avg_w

    def idle_w(self) -> float:
        return self._idle_w
