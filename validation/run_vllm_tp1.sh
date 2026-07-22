#!/bin/bash
# vLLM TP=1 serving for validation
# Run from repo root, then in another terminal: python3 validation/send_requests.py --tp 1

set -e
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
PORT=8001
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/local/lib/ollama/mlx_cuda_v13/libnccl.so.2

echo "Starting vLLM server (TP=1) on port $PORT ..."
echo "In another terminal run:"
echo "  bash validation/measure_power.sh vllm_tp1"
echo "  python3 validation/send_requests.py --tp 1 --port $PORT"
echo ""

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port "$PORT" \
  --disable-log-requests
