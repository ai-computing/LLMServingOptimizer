#!/bin/bash
# GPU 전력 측정 - 별도 터미널에서 실행
# Usage: bash measure_power.sh <label>
# Example: bash measure_power.sh idle
#          bash measure_power.sh vllm_tp1
#          bash measure_power.sh vllm_tp2

LABEL="${1:-power}"
OUT="$(dirname "$0")/${LABEL}_power.csv"

echo "Logging GPU power to: $OUT  (Ctrl+C to stop)"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  -lms 500 > "$OUT"
