#!/usr/bin/env python3
"""Post-hoc power calibration report (plan D5, §4.6 telemetry loop).

Input: JSONL history accumulated at deployment termination — one record per
deployment: {dep_id, model, hw, tp, predicted_power_w, measured_avg_w,
energy_wh, span_s}. Output: a markdown error table plus draft ``measured:``
entries to feed back into profiles/power/<HW>.yaml.

Usage:
    python scripts/power_calibration_report.py \
        --history output/power_calibration_history.jsonl \
        --out power_calibration_report.md [--yaml-out drafts/]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_history(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_report(records: list[dict]) -> tuple[str, dict[str, dict]]:
    """(markdown, {hw: draft power-profile fragment})."""
    lines = ["# Power calibration report", "",
             "| dep | model | hw | tp | predicted W | measured W | err% | Wh | span |",
             "|---|---|---|---|---|---|---|---|---|"]
    errs: list[float] = []
    by_hw: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pred, meas = r.get("predicted_power_w"), r.get("measured_avg_w")
        err = ((meas - pred) / pred * 100) if pred and meas else None
        if err is not None:
            errs.append(abs(err))
        lines.append(
            f"| {r['dep_id']} | {r['model']} | {r['hw']} | {r['tp']} "
            f"| {pred:.0f} | {meas:.0f} "
            f"| {err:+.1f}% | {r.get('energy_wh', 0):.1f} "
            f"| {r.get('span_s', 0):.0f}s |"
            if err is not None else
            f"| {r['dep_id']} | {r['model']} | {r['hw']} | {r['tp']} "
            f"| — | — | — | — | — |")
        if meas:
            by_hw[r["hw"]].append(r)
    if errs:
        lines += ["", f"Mean |err|: **{sum(errs) / len(errs):.1f}%** over "
                      f"{len(errs)} deployment(s)"]

    drafts: dict[str, dict] = {}
    for hw, rs in by_hw.items():
        # average measured power per (model, tp) — draft measured-table rows
        agg: dict[tuple, list[float]] = defaultdict(list)
        for r in rs:
            agg[(r["model"], r["tp"])].append(r["measured_avg_w"])
        drafts[hw] = {"measured": [
            {"model": model, "tp": tp, "load": 1.0,
             "avg_w": round(sum(v) / len(v), 1)}
            for (model, tp), v in sorted(agg.items())]}
        lines += ["", f"## profiles/power/{hw}.yaml draft feedback", "",
                  "```yaml", yaml.safe_dump(drafts[hw], sort_keys=False).strip(),
                  "```"]
    return "\n".join(lines) + "\n", drafts


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--out", default="power_calibration_report.md")
    p.add_argument("--yaml-out", default=None)
    args = p.parse_args(argv)

    records = load_history(args.history)
    if not records:
        print("no history records", file=sys.stderr)
        return 1
    md, drafts = build_report(records)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"wrote {args.out} ({len(records)} records)")
    if args.yaml_out:
        outdir = Path(args.yaml_out)
        outdir.mkdir(parents=True, exist_ok=True)
        for hw, doc in drafts.items():
            (outdir / f"{hw}.measured.yaml").write_text(
                yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            print(f"wrote {outdir / f'{hw}.measured.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
