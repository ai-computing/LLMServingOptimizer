#!/bin/bash
# Sweeps all 13 parallelism configurations for Llama-3.1-8B on 4x A6000.
# Usage: bash script/run_a6000_4_sweep.sh [smoke|full|both]
# Default: both (smoke first, then full)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ASTRA-Sim binary links against the locally-extracted libprotobuf.so.23
export LD_LIBRARY_PATH="/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# graph_generator.py invokes 'python' (not python3); expose the symlink
export PATH="$HOME/.local/bin:$PATH"

PHASE="${1:-both}"
SMOKE_DATASET="dataset/example_trace.jsonl"
FULL_DATASET="dataset/sharegpt_req100_rate10_llama.jsonl"
SMOKE_OUT="output/a6000_4_sweep/smoke"
FULL_OUT="output/a6000_4_sweep/full"
CONFIG_DIR="cluster_config/sweep_a6000_4"

mkdir -p "$SMOKE_OUT" "$FULL_OUT"

CONFIGS=(
    "01_tp1_pp1_dp1"
    "02_tp2_pp1_dp1"
    "03_tp1_pp2_dp1"
    "04_tp2_pp2_dp1"
    "05_tp1_pp4_dp1"
    "06_tp1_pp1_dp2"
    "07_tp2_pp1_dp2"
    "08_tp1_pp2_dp2"
    "09_tp1_pp1_dp4"
    "10_pd_1p1d_tp1"
    "11_pd_1p2d_tp1"
    # configs 12 (pd_1p1d_tp2d) and 13 (pd_1p1d_pp2d) excluded:
    # P/D + decode npu_num>1 causes NPU topology mismatch in config_builder
    # (total_npu=4 but generated network config covers only 3 NPUs), crashing ASTRA-Sim.
)

run_one() {
    local label="$1"
    local dataset="$2"
    local outdir="$3"
    local num_req="$4"
    local cfg="${CONFIG_DIR}/${label}.json"
    local csv="${outdir}/${label}.csv"
    local log="${outdir}/${label}.log"

    echo "  Running ${label} (num-req=${num_req})..."
    python3 main.py \
        --cluster-config "$cfg" \
        --fp 16 --block-size 16 \
        --dataset "$dataset" \
        --output "$csv" \
        --num-req "$num_req" \
        --log-interval 1.0 \
        --log-level WARNING \
        > "$log" 2>&1

    if grep -q "Simulation results" "$log"; then
        echo "    OK: $label"
    else
        echo "    FAILED: $label — see $log" >&2
        return 1
    fi
}

if [[ "$PHASE" == "smoke" || "$PHASE" == "both" ]]; then
    echo "=== SMOKE PHASE (10 reqs) ==="
    for label in "${CONFIGS[@]}"; do
        run_one "$label" "$SMOKE_DATASET" "$SMOKE_OUT" 10
    done
    echo "Smoke phase complete."
fi

if [[ "$PHASE" == "full" || "$PHASE" == "both" ]]; then
    echo ""
    echo "=== FULL PHASE (100 reqs) ==="
    for label in "${CONFIGS[@]}"; do
        run_one "$label" "$FULL_DATASET" "$FULL_OUT" 100
    done
    echo "Full phase complete."
fi

echo ""
echo "All runs finished. Results in output/a6000_4_sweep/"
