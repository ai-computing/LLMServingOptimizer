"""Capacity-curve oracle: the common fallback (design doc §5.2, P0.5 first).

Backed by a YAML table measured with a per-instance concurrency sweep:

.. code-block:: yaml

    hw: A5000
    model: meta-llama/Llama-3.1-8B
    tp: 1
    idle_w: 20
    meta: {stack: "vllm-0.19", measured_at: "2026-07-29", source: measured}
    points:
      - {concurrency: 1,  thr_toks_s: 45,  ttft_ms: 90,  tpot_ms: 22, itl_p99_ms: 30, avg_w: 190}
      - {concurrency: 8,  thr_toks_s: 260, ttft_ms: 150, tpot_ms: 30, itl_p99_ms: 45, avg_w: 225}
      - {concurrency: 32, thr_toks_s: 520, ttft_ms: 420, tpot_ms: 60, itl_p99_ms: 110, avg_w: 235}

Queries between measured concurrencies interpolate linearly; queries outside
the measured range clamp to the nearest endpoint and emit
:class:`ExtrapolationWarning`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from .base import Envelope, ExtrapolationWarning, OracleMeta, SteadyPoint


class CapacityCurveOracle:
    def __init__(self, hw: str, model: str, tp: int, points: list[SteadyPoint],
                 idle_w: float = 0.0, meta: OracleMeta | None = None,
                 envelope: Envelope | None = None):
        if not points:
            raise ValueError("capacity curve needs at least one measured point")
        self.hw = hw
        self.model = model
        self.tp = tp
        self.points = sorted(points, key=lambda p: p.concurrency)
        if len({p.concurrency for p in self.points}) != len(self.points):
            raise ValueError("duplicate concurrency levels in capacity curve")
        self._idle_w = float(idle_w)
        self.meta = meta or OracleMeta()
        self.envelope = envelope or Envelope(
            max_concurrency=self.points[-1].concurrency)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "CapacityCurveOracle":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        for key in ("hw", "model", "tp", "points"):
            if key not in d:
                raise ValueError(f"capacity curve YAML missing '{key}': {path}")
        points = [SteadyPoint(**p) for p in d["points"]]
        meta = OracleMeta(**d.get("meta", {})) if d.get("meta") else OracleMeta()
        env = None
        if d.get("envelope"):
            env = Envelope(**d["envelope"])
        return cls(hw=d["hw"], model=d["model"], tp=int(d["tp"]), points=points,
                   idle_w=float(d.get("idle_w", 0.0)), meta=meta, envelope=env)

    # -- oracle protocol ------------------------------------------------------
    def steady_state(self, concurrency: int) -> SteadyPoint:
        pts = self.points
        c = concurrency
        if c <= pts[0].concurrency:
            if c < pts[0].concurrency:
                warnings.warn(
                    f"{self.hw}/{self.model}/tp{self.tp}: concurrency {c} below "
                    f"measured range [{pts[0].concurrency}, {pts[-1].concurrency}]",
                    ExtrapolationWarning, stacklevel=2)
            return pts[0]
        if c >= pts[-1].concurrency:
            if c > pts[-1].concurrency:
                warnings.warn(
                    f"{self.hw}/{self.model}/tp{self.tp}: concurrency {c} above "
                    f"measured range [{pts[0].concurrency}, {pts[-1].concurrency}]",
                    ExtrapolationWarning, stacklevel=2)
            return pts[-1]
        for a, b in zip(pts, pts[1:]):
            if a.concurrency <= c <= b.concurrency:
                f = (c - a.concurrency) / (b.concurrency - a.concurrency)

                def lerp(x, y):
                    return x + f * (y - x)

                return SteadyPoint(
                    concurrency=c,
                    thr_toks_s=lerp(a.thr_toks_s, b.thr_toks_s),
                    ttft_ms=lerp(a.ttft_ms, b.ttft_ms),
                    tpot_ms=lerp(a.tpot_ms, b.tpot_ms),
                    itl_p99_ms=lerp(a.itl_p99_ms, b.itl_p99_ms),
                    avg_w=lerp(a.avg_w, b.avg_w),
                )
        raise AssertionError("unreachable")

    def power_w(self, load: float) -> float:
        """Power at fractional load: interpolate avg_w over the measured curve,
        mapping load in [0,1] onto the measured concurrency range."""
        load = max(0.0, min(1.0, load))
        if load == 0.0:
            return self._idle_w
        c = self.points[0].concurrency + load * (
            self.points[-1].concurrency - self.points[0].concurrency)
        pts = self.points
        if c <= pts[0].concurrency:
            return pts[0].avg_w
        for a, b in zip(pts, pts[1:]):
            if a.concurrency <= c <= b.concurrency:
                f = (c - a.concurrency) / (b.concurrency - a.concurrency)
                return a.avg_w + f * (b.avg_w - a.avg_w)
        return pts[-1].avg_w

    def idle_w(self) -> float:
        return self._idle_w
