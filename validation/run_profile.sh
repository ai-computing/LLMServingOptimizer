#!/bin/bash
# A5000 Llama-3.1-8B profiling: layers, attention, predictor, power
# Run from: llm_profile/ directory
#   cd ../llm_profile && bash ../validation/run_profile.sh

set -e
cd "$(dirname "$0")/../llm_profile"

MODEL="meta-llama/Llama-3.1-8B"
HW="A5000"
OUT_DIR="perf_models/${HW}/meta-llama/Llama-3.1-8B"

echo "=== [1/4] Layer profiling TP=1 ==="
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.layers.main \
  --hardware "$HW" --model "$MODEL" \
  --num-layers 1 --tp-size "1" \
  --warmup 10 --repeat 30 --max-len 10 --device cuda

echo "=== [2/4] Layer profiling TP=2 ==="
CUDA_VISIBLE_DEVICES=0,1 python3 -m profiler.layers.main \
  --hardware "$HW" --model "$MODEL" \
  --num-layers 1 --tp-size "2" \
  --warmup 10 --repeat 30 --max-len 10 --device cuda

echo "=== [3/4] Attention profiling TP=1 ==="
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.attention.main \
  --model "$MODEL" --hardware "$HW" \
  --max-len 2048 --tp-size "1" \
  --warmup 10 --repeat 50 --device cuda

echo "=== [4/4] Attention profiling TP=2 ==="
CUDA_VISIBLE_DEVICES=0,1 python3 -m profiler.attention.main \
  --model "$MODEL" --hardware "$HW" \
  --max-len 2048 --tp-size "2" \
  --warmup 10 --repeat 50 --device cuda

echo "=== [5/5] Build attention predictor TP=1,2 ==="
python3 -m profiler.predictor.main \
  --model "$MODEL" --hardware "$HW" \
  --tp-size "1, 2" \
  --kv-granularity 64 --chunk-granularity 32 \
  --max-len 2048 --max-batch 256

echo ""
echo "Profiling complete. Output: llm_profile/${OUT_DIR}/"
echo ""
echo "Next: measure GPU power (run in a separate terminal while profiling):"
echo "  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw \\"
echo "    --format=csv,noheader,nounits -lms 500 > ../validation/a5000_idle_power.csv"
