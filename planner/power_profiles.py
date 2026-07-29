"""Power profile schema + loader (PLAN_power_service.md M1, design doc §4.2).

``profiles/power/<HW>.yaml`` is the canonical per-hardware power source:

.. code-block:: yaml

    device: {name: A40, active_w: 300, idle_w: 25, mem_gb: 48}
    host_overhead: {base_w: 250, per_device_w: 15}
    measured:            # optional, wins over everything else
      - {model: "meta-llama/Llama-3.1-70B", tp: 4, load: 0.8, avg_w: 1180}
    meta: {source: "datasheet", measured_at: "2026-07-01", stack: "vllm-0.19"}

Resolution is a three-tier fallback (most-trusted first):

1. ``measured`` table entry for (model, tp), linearly interpolated over ``load``
   (clamped to the nearest endpoint outside the measured range);
2. ``device.active_w`` from the profile file;
3. no profile file at all -> legacy constant table (previously
   ``milp_solver._HW_ACTIVE_POWER``) with a warning, so behaviour without
   profile files is byte-identical to the pre-M1 planner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .utils import get_logger

log = get_logger("planner.power")

REPO_ROOT = Path(__file__).resolve().parents[1]
#: default profile directory; tests monkeypatch this module global
DEFAULT_POWER_ROOT = REPO_ROOT / "profiles" / "power"

#: legacy per-device active power (W) — moved verbatim from milp_solver so the
#: solver's no-profile behaviour (and its golden solutions) cannot drift.
LEGACY_ACTIVE_POWER_W = {
    "H100": 700, "A100": 400, "A6000": 300, "A40": 300,
    "A40x": 300, "A5000": 230, "RTX3090": 350, "RNGD": 150, "TPU-v6e-1": 200,
}
_DEFAULT_ACTIVE_W = 300.0  # matches the old ``.get(hw, 300)`` default


class DeviceBlock(BaseModel):
    name: str
    active_w: float = Field(gt=0)
    idle_w: float = Field(ge=0)
    mem_gb: float = Field(gt=0)


class HostOverhead(BaseModel):
    base_w: float = Field(ge=0, default=0.0)
    per_device_w: float = Field(ge=0, default=0.0)


class MeasuredPoint(BaseModel):
    model: str
    tp: int = Field(ge=1)
    load: float = Field(gt=0, le=1.0)
    avg_w: float = Field(gt=0)


class MetaBlock(BaseModel):
    source: str = "estimate"  # nvidia-smi | datasheet | estimate
    measured_at: Optional[str] = None
    stack: Optional[str] = None


class PowerProfile(BaseModel):
    device: DeviceBlock
    host_overhead: HostOverhead = Field(default_factory=HostOverhead)
    measured: list[MeasuredPoint] = Field(default_factory=list)
    meta: MetaBlock = Field(default_factory=MetaBlock)


def load_power_profile(hw: str, root: str | Path | None = None) -> PowerProfile:
    """Load and validate ``<root>/<hw>.yaml``.

    Raises ``FileNotFoundError`` when the file is absent and pydantic's
    ``ValidationError`` when it is malformed (malformed files fail loudly —
    only *absence* triggers the legacy-constant fallback).
    """
    root = Path(root) if root is not None else DEFAULT_POWER_ROOT
    path = root / f"{hw}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no power profile for hardware '{hw}' under {root}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PowerProfile.model_validate(data)


def _try_load(hw: str, root: str | Path | None) -> Optional[PowerProfile]:
    try:
        return load_power_profile(hw, root)
    except FileNotFoundError:
        return None


def device_active_w(hw: str, root: str | Path | None = None) -> tuple[float, str]:
    """Per-device active power in W with source tag: profile | legacy_const | default."""
    prof = _try_load(hw, root)
    if prof is not None:
        return float(prof.device.active_w), "profile"
    if hw in LEGACY_ACTIVE_POWER_W:
        log.warning("no power profile file for '%s'; using legacy constant %dW", hw,
                    LEGACY_ACTIVE_POWER_W[hw])
        return float(LEGACY_ACTIVE_POWER_W[hw]), "legacy_const"
    log.warning("no power profile or legacy constant for '%s'; using default %.0fW",
                hw, _DEFAULT_ACTIVE_W)
    return _DEFAULT_ACTIVE_W, "default"


def device_idle_w(hw: str, root: str | Path | None = None) -> float:
    """Per-device idle power in W (0.0 when no profile file exists)."""
    prof = _try_load(hw, root)
    return float(prof.device.idle_w) if prof is not None else 0.0


def host_overhead_w(hw: str, root: str | Path | None = None) -> HostOverhead:
    """Host-level overhead block (zeros when no profile file exists)."""
    prof = _try_load(hw, root)
    return prof.host_overhead if prof is not None else HostOverhead()


def effective_power_w(
    hw: str,
    model: Optional[str] = None,
    tp: Optional[int] = None,
    load: Optional[float] = None,
    root: str | Path | None = None,
) -> tuple[float, str]:
    """Per-*instance* power (W) for (hw, model, tp) at ``load``, with source tag.

    An instance spans ``tp`` devices, so the active_w fallbacks are multiplied
    by ``tp``; ``measured.avg_w`` rows are whole-instance measurements and are
    used as-is. Source tags: measured | active_w | legacy_const | default.
    """
    ntp = tp if tp else 1
    prof = _try_load(hw, root)
    if prof is not None:
        if model is not None and tp is not None:
            pts = sorted(
                (p for p in prof.measured if p.model == model and p.tp == tp),
                key=lambda p: p.load,
            )
            if pts:
                q = load if load is not None else 1.0
                if q <= pts[0].load:
                    return float(pts[0].avg_w), "measured"
                if q >= pts[-1].load:
                    return float(pts[-1].avg_w), "measured"
                for a, b in zip(pts, pts[1:]):
                    if a.load <= q <= b.load:
                        frac = (q - a.load) / (b.load - a.load)
                        return float(a.avg_w + frac * (b.avg_w - a.avg_w)), "measured"
        return float(prof.device.active_w) * ntp, "active_w"
    per_dev, source = device_active_w(hw, root)  # legacy_const | default (warns)
    return per_dev * ntp, source
