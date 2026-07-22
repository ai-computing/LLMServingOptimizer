#!/bin/bash
# End-to-end multi-node vLLM TP=16 ground-truth run across s8 (head) + s2 (worker) over IB.
# Run from s8 (repo root). Assumes the vLLM image and model weights are already present on
# both nodes (image: docker save|load over IB; weights: rsync over IB).
#
# Usage: validation/multinode/run_multinode_validation.sh <model_tag> <s8_model_path> <s2_model_path>
#   8B:  ... 8b  /home/bdsl/hyungyuJung/data/models/Llama-3.1-8B-Instruct  /home/swsok/models/Llama-3.1-8B-Instruct
set -euo pipefail
TAG="${1:-8b}"
S8_MODEL="${2:-/home/bdsl/hyungyuJung/data/models/Llama-3.1-8B-Instruct}"
S2_MODEL="${3:-/home/swsok/models/Llama-3.1-8B-Instruct}"
SERVED="meta-llama/Llama-3.1-8B"
PORT=8016
S8_IB=192.168.210.108
S2_IB=192.168.210.102
S2_SSH="ssh -p 10022 -o BatchMode=yes s2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATASET="${DATASET:-dataset/sharegpt_req100_rate10_llama.jsonl}"
NUM_REQ="${NUM_REQ:-100}"
RESULTS="$REPO/validation/vllm_a40_tp16_${TAG}_results.jsonl"

teardown() {
  echo "[teardown] stopping ray + containers on both nodes"
  docker exec vllm_mn ray stop >/dev/null 2>&1 || true
  docker rm -f vllm_mn >/dev/null 2>&1 || true
  $S2_SSH 'docker exec vllm_mn ray stop >/dev/null 2>&1; docker rm -f vllm_mn >/dev/null 2>&1' || true
}
trap teardown EXIT

echo "==> [1/5] Start Ray HEAD on s8"
bash "$HERE/run_cluster_node.sh" head "$S8_IB" "$S8_IB" ibs8 "$S8_MODEL"

echo "==> [2/5] Start Ray WORKER on s2 (over IB)"
$S2_SSH "bash -s" -- worker "$S2_IB" "$S8_IB" ibp194s0 "$S2_MODEL" < "$HERE/run_cluster_node.sh"

echo "==> [3/5] Verify Ray sees 16 GPUs"
for i in $(seq 1 24); do
  ngpu=$(docker exec vllm_mn ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | grep -oE "/[0-9.]+ GPU" | grep -oE "[0-9.]+" | head -1 || echo 0)
  echo "   ray GPUs: ${ngpu:-?}"
  [ "${ngpu%.*}" = "16" ] && break
  sleep 5
done

echo "==> [4/5] Launch TP=16 serve + wait for health"
bash "$HERE/serve_tp16.sh" "$SERVED" "$PORT"

echo "==> [5/5] Send $NUM_REQ requests"
cd "$REPO"
python3 validation/send_requests.py --tp 16 --port "$PORT" \
  --dataset "$DATASET" --num-req "$NUM_REQ" --output "$RESULTS"
echo "==> DONE. ground-truth results: $RESULTS"
