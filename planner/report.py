"""Write planner outputs: best config, a per-candidate CSV, and a markdown report."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .search_orchestrator import PlannerResult
from .utils import REPO_ROOT, get_logger

log = get_logger("planner.report")

_CSV_FIELDS = [
    "run_id", "on_pareto", "passed", "score", "batch_tokens",
    "ttft_ms", "tpot_ms", "itl_p99_ms", "throughput_toks_s", "toks_per_wh",
    "config_path", "status",
]


def write_reports(result: PlannerResult, out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 1) pareto.csv
    csv_path = out_dir / "pareto.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for c in result.candidates:
            m = c.metrics
            if result.dry_run:
                status = "rendered (dry-run)"
            elif m is not None:
                status = "ok"
            else:
                status = c.infeasible_reason or "infeasible"
            w.writerow({
                "run_id": c.run_id,
                "on_pareto": c.run_id in result.pareto,
                "passed": c.passed,
                "score": f"{c.score:.4f}" if c.score != float("-inf") else "",
                "batch_tokens": c.batch_tokens,
                "ttft_ms": f"{m.ttft_ms:.3f}" if m else "",
                "tpot_ms": f"{m.tpot_ms:.3f}" if m else "",
                "itl_p99_ms": f"{m.itl_p99_ms:.3f}" if m else "",
                "throughput_toks_s": f"{m.throughput_toks_s:.2f}" if m else "",
                "toks_per_wh": f"{m.toks_per_wh:.2f}" if (m and m.toks_per_wh) else "",
                "config_path": c.config_path,
                "status": status,
            })
    paths["pareto_csv"] = str(csv_path)

    # 2) best_cluster_config.json (copy of the winning rendered config)
    if result.best is not None:
        src = REPO_ROOT / result.best.config_path
        dst = out_dir / "best_cluster_config.json"
        if src.is_file():
            shutil.copyfile(src, dst)
            paths["best_config"] = str(dst)
        (out_dir / "best_run.json").write_text(json.dumps({
            "run_id": result.best.run_id,
            "cli_args": result.best.cli_args,
            "metrics": result.best.metrics.as_row() if result.best.metrics else None,
        }, indent=2))

    # 3) report.md
    md_path = out_dir / "report.md"
    lines = ["# Planner report", ""]
    if result.dry_run:
        lines.append("> **dry run** — Stage-1 + rendering only (no simulation).")
        lines.append("")
    lines.append(f"- Candidates rendered: **{len(result.candidates)}**")
    passing = [c for c in result.candidates if c.passed]
    lines.append(f"- Passed SLO: **{len(passing)}**")
    lines.append(f"- Pareto front: **{len(result.pareto)}**")
    if result.best is not None:
        b = result.best
        lines += [
            "",
            "## Best (by weighted score)",
            f"- run_id: `{b.run_id}`",
            f"- config: `{b.config_path}`",
        ]
        if b.metrics:
            m = b.metrics
            lines.append(
                f"- TTFT={m.ttft_ms:.2f} ms, TPOT={m.tpot_ms:.2f} ms, "
                f"ITL-p99={m.itl_p99_ms:.2f} ms, throughput={m.throughput_toks_s:.1f} tok/s"
            )
    lines += ["", "## All candidates", "", f"See `pareto.csv` ({len(result.candidates)} rows)."]
    md_path.write_text("\n".join(lines) + "\n")
    paths["report_md"] = str(md_path)

    log.info("wrote reports to %s", out_dir)
    return paths
