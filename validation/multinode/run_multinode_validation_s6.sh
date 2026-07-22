#!/bin/bash
# End-to-end multi-node vLLM TP=16 ground-truth run across s8 (head) + s6 (worker) over IB.
# s6 replaces s2 for the 70B run (s2 lacked disk for the 132 GB FP16 weights; s6 has 290 GB
# free and its mlx5_0 IB port is ACTIVE at 200 Gb/s on the SAME fabric as s8/s2).
#
# Prereq (run once, needs sudo/real TTY on s6):
#   sudo usermod -aG docker swsok           # docker without sudo
#   sudo ip addr add 192.168.210.106/24 dev ibs8 && sudo ip link set ibs8 up   # IPoIB up on .210
# Then image + weights staged on s6:
#   docker save nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 | ssh -p 10022 s6 docker load
#   rsync -aP -e 'ssh -p 10022' <s8 weights>/ 192.168.210.106:/home/swsok/models/Llama-3.1-70B-Instruct/
#
# Usage: validation/multinode/run_multinode_validation_s6.sh [model_tag] [s8_model_path] [s6_model_path]
set -euo pipefail
TAG="${1:-70b}"
S8_MODEL="${2:-/home/bdsl/hyungyuJung/data/models/Llama-3.1-70B-Instruct}"
S6_MODEL="${3:-/home/swsok/models/Llama-3.1-70B-Instruct}"
SERVED="${SERVED:-meta-llama/Llama-3.1-70B}"   # matches cluster_config model_name
PORT="${PORT:-8016}"
S8_IB=192.168.210.108
S6_IB=192.168.210.106
S8_IFACE=ibs8
S6_IFACE=ibs8            # s6's IPoIB interface is also named ibs8
S6_SSH="ssh -p 10022 -o BatchMode=yes s6"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATASET="${DATASET:-dataset/sharegpt_req100_rate10_llama.jsonl}"
NUM_REQ="${NUM_REQ:-100}"
RESULTS="$REPO/validation/vllm_a40_tp16_${TAG}_results.jsonl"

teardown() {
  echo "[teardown] stopping ray + containers on both nodes"
  docker exec vllm_mn ray stop >/dev/null 2>&1 || true
  docker rm -f vllm_mn >/dev/null 2>&1 || true
  $S6_SSH 'docker exec vllm_mn ray stop >/dev/null 2>&1; docker rm -f vllm_mn >/dev/null 2>&1' || true
}
trap teardown EXIT

echo "==> [0/5] IB path check: s8 -> s6 ($S6_IB)"
ping -c2 -W3 "$S6_IB" >/dev/null || { echo "ERROR: s6 IB $S6_IB unreachable — bring up ibs8 on s6 first"; exit 1; }

echo "==> [1/5] Start Ray HEAD on s8"
bash "$HERE/run_cluster_node.sh" head "$S8_IB" "$S8_IB" "$S8_IFACE" "$S8_MODEL"

echo "==> [2/5] Start Ray WORKER on s6 (over IB)"
$S6_SSH "bash -s" -- worker "$S6_IB" "$S8_IB" "$S6_IFACE" "$S6_MODEL" < "$HERE/run_cluster_node.sh"

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
  --model "$SERVED" --timeout "${REQ_TIMEOUT:-900}" \
  --dataset "$DATASET" --num-req "$NUM_REQ" --output "$RESULTS"
echo "==> DONE. ground-truth results: $RESULTS"
