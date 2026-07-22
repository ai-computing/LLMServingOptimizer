#!/bin/bash
# vLLM (A40, TP=N) benchmark for Llama-3.1-8B, with GPU power logging.
# Usage:  bash validation/run_vllm_a40_bench.sh [TP]      (TP defaults to 1)
# Uses a complete LOCAL copy of the weights (HF CDN here is throttled to ~0.6 MB/s).
# Llama-3.1-8B-Instruct == base architecture/tokenizer; with temperature=0 and fixed
# max_tokens the compute (hence latency/throughput/power) is identical to base.
set -euo pipefail

TP="${1:-1}"
LOCAL_MODEL="/home/bdsl/hyungyuJung/data/models/Llama-3.1-8B-Instruct"
SERVED_NAME="meta-llama/Llama-3.1-8B"
PORT=$((8000 + TP))                              # TP1->8001, TP2->8002
GPUS=$(seq -s, 0 $((TP-1)))                      # TP1->"0", TP2->"0,1"
DATASET="${DATASET:-dataset/sharegpt_req300_rate10_llama.jsonl}"
NUM_REQ="${NUM_REQ:-300}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PWRLOG="$REPO/validation/vllm_a40_tp${TP}_power.csv"
RESULTS="$REPO/validation/vllm_a40_tp${TP}_results.jsonl"
CNAME="vllm_a40_bench_tp${TP}"

cleanup() {
  echo "[cleanup] stopping power logger + container"
  [ -n "${PWR_PID:-}" ] && kill "$PWR_PID" 2>/dev/null || true
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$CNAME" >/dev/null 2>&1 || true

echo "==> Launching vLLM server (TP=$TP, GPUs=$GPUS, port=$PORT) from local weights ..."
docker run -d --name "$CNAME" \
  --gpus "\"device=$GPUS\"" \
  --ipc=host \
  -v "$LOCAL_MODEL:/model:ro" \
  -e "HF_HUB_OFFLINE=1" \
  -p "$PORT:$PORT" \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name "$SERVED_NAME" \
  --dtype float16 \
  --tensor-parallel-size "$TP" \
  --max-model-len 4096 \
  --port "$PORT"

echo "==> Waiting for vLLM to be ready (load + CUDA graph, up to 15 min) ..."
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "vLLM ready after ${i}*5s"; break; fi
  if ! docker ps --filter "name=$CNAME" --format '{{.Names}}' | grep -q "$CNAME"; then
    echo "ERROR: container exited early. Logs:"; docker logs --tail 40 "$CNAME"; exit 1
  fi
  sleep 5
done
curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "ERROR: not ready in time"; docker logs --tail 40 "$CNAME"; exit 1; }

echo "==> Starting GPU power logging on GPUs $GPUS -> $PWRLOG"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -lms 500 -i "$GPUS" > "$PWRLOG" &
PWR_PID=$!

echo "==> Sending $NUM_REQ requests ..."
cd "$REPO"
python3 validation/send_requests.py --tp "$TP" --port "$PORT" \
  --dataset "$DATASET" --num-req "$NUM_REQ" --output "$RESULTS"

echo "==> Benchmark done.  results: $RESULTS  power: $PWRLOG"
