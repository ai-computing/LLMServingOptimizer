"""M4 (P0) E2E: power-min planning end-to-end on the real simulator.

sim-marked: needs the legacy backend built. Verifies the P0 completion
criterion — on the same heterogeneous topology, the power-min spec yields a
different, lower-power final recommendation than the max-throughput spec,
and no SLO violator is recommended.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from planner import report
from planner.search_orchestrator import run_spec
from planner.spec_schema import PlannerSpec

pytestmark = pytest.mark.sim

SPEC_PATH = Path(__file__).resolve().parents[2] / "examples/service/spec_powermin_hetero.yaml"


def _legacy_available() -> bool:
    try:
        from sim_backends import get_backend
        return get_backend("legacy").available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def powermin_spec() -> PlannerSpec:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return PlannerSpec.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def out_root(tmp_path_factory):
    return tmp_path_factory.mktemp("e2e_powermin")


@pytest.mark.skipif(not _legacy_available(), reason="legacy backend not built")
def test_powermin_vs_maxthroughput_e2e(powermin_spec, out_root):
    # --- power-min run ------------------------------------------------------
    pm_dir = out_root / "powermin"
    pm = run_spec(copy.deepcopy(powermin_spec), out_dir=pm_dir, jobs=4,
                  timeout_sec=1200)
    assert pm.best is not None, "power-min: no candidate passed the SLO"
    assert pm.best.passed and pm.best.metrics is not None
    assert pm.best.metrics.power_w is not None

    paths = report.write_reports(pm, pm_dir)
    best_cfg = Path(paths["best_config"])
    assert best_cfg.is_file()
    cfg = json.loads(best_cfg.read_text())
    assert cfg.get("nodes"), "best_cluster_config.json must be a valid cluster config"

    # --- max-throughput run on the same topology ----------------------------
    mt_spec = copy.deepcopy(powermin_spec)
    mt_spec.requirements.objectives = [
        type(powermin_spec.requirements.objectives[0]).model_validate(
            {"metric": "throughput", "direction": "max", "weight": 1.0})
    ]
    mt = run_spec(mt_spec, out_dir=out_root / "maxthr", jobs=4, timeout_sec=1200)
    assert mt.best is not None, "max-throughput: no candidate passed the SLO"
    assert mt.best.metrics is not None and mt.best.metrics.power_w is not None

    # --- P0 differentiation criteria ----------------------------------------
    assert pm.best.allocation.signature() != mt.best.allocation.signature(), \
        "power-min and max-throughput picked the same allocation"
    assert pm.best.metrics.power_w < mt.best.metrics.power_w, (
        f"power-min best ({pm.best.metrics.power_w:.0f} W) is not below "
        f"max-throughput best ({mt.best.metrics.power_w:.0f} W)"
    )

    # no SLO violator recommended anywhere
    for res in (pm, mt):
        assert res.best.passed
        assert not res.best.violations


@pytest.mark.skipif(not _legacy_available(), reason="legacy backend not built")
def test_report_shows_power_columns(powermin_spec, out_root):
    """report artifacts from the cached power-min run carry power/backend info."""
    pm_dir = out_root / "powermin"
    csv_path = pm_dir / "pareto.csv"
    if not csv_path.is_file():
        pytest.skip("previous e2e test did not produce reports")
    header = csv_path.read_text().splitlines()[0]
    for col in ("power_w", "power_source", "backend"):
        assert col in header
    md = (pm_dir / "report.md").read_text()
    assert "power=" in md
