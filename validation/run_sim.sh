#!/bin/bash
# 시뮬레이션 실행 (TP=1, TP=2) for A5000 validation
# Run from repo root: bash validation/run_sim.sh

set -e
DATASET="${DATASET:-dataset/sharegpt_req300_rate10_llama.jsonl}"
NUM_REQ="${NUM_REQ:-300}"

echo "=== [1/2] Simulation TP=1 ==="
python3 main.py \
  --cluster-config cluster_config/a5000_1gpu_validation.json \
  --fp 16 --block-size 16 \
  --dataset "$DATASET" \
  --output validation/sim_tp1_results.csv \
  --num-req "$NUM_REQ" \
  --log-interval 1.0 \
  2>&1 | tee validation/sim_tp1_stdout.txt

echo ""
echo "=== [2/2] Simulation TP=2 ==="
python3 main.py \
  --cluster-config cluster_config/a5000_2gpu_tp2_validation.json \
  --fp 16 --block-size 16 \
  --dataset "$DATASET" \
  --output validation/sim_tp2_results.csv \
  --num-req "$NUM_REQ" \
  --log-interval 1.0 \
  2>&1 | tee validation/sim_tp2_stdout.txt

echo ""
echo "Simulations complete."
echo "  TP=1: validation/sim_tp1_results.csv"
echo "  TP=2: validation/sim_tp2_results.csv"
