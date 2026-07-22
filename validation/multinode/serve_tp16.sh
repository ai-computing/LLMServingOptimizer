#!/bin/bash
# Launch vLLM TP=16 serving spanning both nodes via the already-running Ray cluster.
# Run on the HEAD node (s8) AFTER run_cluster_node.sh has started on both nodes and
# `ray status` shows 16 GPUs.
#
# Usage: serve_tp16.sh <served_model_name> <port>
#   serve_tp16.sh meta-llama/Llama-3.1-8B 8016
set -euo pipefail
SERVED="${1:-meta-llama/Llama-3.1-8B}"
PORT="${2:-8016}"
CNAME="vllm_mn"
HEAD_IP="192.168.210.108"

echo "==> Ray cluster resources:"
docker exec "$CNAME" ray status 2>&1 | grep -iE "GPU|node_" | head

echo "==> Launching vLLM serve (TP=16, ray backend) on port $PORT ..."
docker exec -d "$CNAME" bash -c "VLLM_HOST_IP=$HEAD_IP vllm serve /model \
  --served-model-name '$SERVED' \
  --dtype float16 \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port $PORT > /serve.log 2>&1"

echo "==> Waiting for server health (up to 20 min: 16-way NCCL/IB init is slow) ..."
for i in $(seq 1 240); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM TP16 ready after ${i}*5s"; exit 0
  fi
  if ! docker exec "$CNAME" pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "ERROR: vllm serve process died. Last log:"; docker exec "$CNAME" tail -40 /serve.log; exit 1
  fi
  sleep 5
done
echo "ERROR: not ready in 20 min. Last log:"; docker exec "$CNAME" tail -60 /serve.log; exit 1
