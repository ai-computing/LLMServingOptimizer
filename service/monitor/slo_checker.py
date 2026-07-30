"""Sliding-window SLO verdicts (plan §4.3): 30s window over MetricSamples,
p95 observed vs promised targets; 3 consecutive violated windows trigger the
DEGRADED transition callback exactly once, recovery flips back to READY."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .collector import MetricSample

_FIELD_FOR_TARGET = {"ttft_ms": "ttft_p95_ms", "tpot_ms": "tpot_p95_ms",
                     "itl_p99_ms": "tpot_p95_ms"}  # itl proxied by tpot p95 (P0)


@dataclass
class SLOStatus:
    dep_id: str
    window_s: int
    targets: dict
    observed: dict
    verdict: str                  # ok | warn | violated
    breaches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SLOChecker:
    dep_id: str
    targets: dict                             # promised SLO (subset of keys)
    window_s: int = 30
    consecutive_for_degraded: int = 3
    on_degraded: Optional[Callable[[str], None]] = None
    on_recovered: Optional[Callable[[str], None]] = None
    _samples: deque = field(default_factory=deque)
    _violations: int = 0
    _degraded: bool = False

    def push(self, m: MetricSample) -> SLOStatus:
        self._samples.append(m)
        while self._samples and m.ts - self._samples[0].ts > self.window_s:
            self._samples.popleft()
        observed, breaches = {}, []
        for target_key, limit in (self.targets or {}).items():
            fld = _FIELD_FOR_TARGET.get(target_key)
            if fld is None or limit is None:
                continue
            vals = [getattr(s, fld) for s in self._samples if getattr(s, fld) > 0]
            if not vals:
                continue
            worst = max(vals)
            observed[target_key] = round(worst, 3)
            if worst > limit:
                breaches.append(f"{target_key}: {worst:.0f} > {limit:.0f}")

        if breaches:
            self._violations += 1
        else:
            self._violations = 0
            if self._degraded:
                self._degraded = False
                if self.on_recovered:
                    self.on_recovered(self.dep_id)

        if breaches and self._violations >= self.consecutive_for_degraded:
            verdict = "violated"
            if not self._degraded:
                self._degraded = True
                if self.on_degraded:
                    self.on_degraded(self.dep_id)
        elif breaches:
            verdict = "warn"
        else:
            verdict = "ok"
        return SLOStatus(dep_id=self.dep_id, window_s=self.window_s,
                         targets=self.targets, observed=observed,
                         verdict=verdict, breaches=breaches)
