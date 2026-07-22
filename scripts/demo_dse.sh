#!/bin/bash
# Phase 5 demo runner — one-line e2e for PLAN_webapp_dse_detail.md §5.4.
# Default: smoke spec (A6000 only, fast). Override arg 1 for 70B demo.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SPEC="${1:-examples/dse/spec_llama8b_smoke.yaml}"
JOB_NAME="${2:-demo}"

echo "▶ DSE demo: $SPEC (job-name=$JOB_NAME)"

python3 -m webapp.dse.cli explore --spec "$SPEC" --job-name "$JOB_NAME"

LATEST=$(ls -td "$REPO_ROOT"/output/dse_jobs/*-"$JOB_NAME" 2>/dev/null | head -1)
echo
echo "Artifacts: $LATEST"
if [ -f "$LATEST/top_n.json" ]; then
    python3 -c "
import json
top = json.load(open('$LATEST/top_n.json'))
print('\nTop-N:')
for i, r in enumerate(top, 1):
    m = r.get('metrics', {})
    print(f'  {i}. {r[\"label\"]:40s} score={r.get(\"score\")} ttft={m.get(\"p99_ttft_ms\")} tp={m.get(\"total_token_tp\")} energy={m.get(\"total_energy_wh\")}')
"
fi
