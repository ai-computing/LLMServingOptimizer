"""Monitoring retention (plan §3.4): per-deployment in-memory ring buffers for
the live hour, energy integration, and a terminal Wh summary for the
prediction-vs-actual calibration loop (D5)."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerSample:
    ts: float
    device_id: str
    power_w: float
    util: float = 0.0
    mem_used_gb: float = 0.0
    temp_c: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class DeploymentBuffers:
    """Ring buffers (1h at 5s = 720 entries) + trapezoidal energy integral."""

    def __init__(self, dep_id: str, maxlen: int = 720):
        self.dep_id = dep_id
        self.metrics: deque = deque(maxlen=maxlen)
        self.power: deque = deque(maxlen=maxlen)     # (ts, total_w)
        self.logs: deque = deque(maxlen=5000)
        self._energy_j = 0.0
        self._last_power: Optional[tuple[float, float]] = None
        self._lock = threading.Lock()

    def push_metrics(self, sample) -> None:
        with self._lock:
            self.metrics.append(sample)

    def push_power(self, ts: float, total_w: float) -> None:
        with self._lock:
            if self._last_power is not None:
                t0, w0 = self._last_power
                if ts > t0:
                    self._energy_j += (w0 + total_w) / 2.0 * (ts - t0)
            self._last_power = (ts, total_w)
            self.power.append((ts, total_w))

    def push_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)

    @property
    def energy_wh(self) -> float:
        return self._energy_j / 3600.0

    def summary(self) -> dict:
        with self._lock:
            span = (self.power[-1][0] - self.power[0][0]) if len(self.power) > 1 else 0.0
            avg_w = (self._energy_j / span) if span > 0 else 0.0
            return {"dep_id": self.dep_id, "energy_wh": round(self.energy_wh, 3),
                    "avg_power_w": round(avg_w, 1), "span_s": round(span, 1)}
