#!/bin/bash
# vLLM (Llama-3.1-8B) NVLink-vs-PCIe ablation on a single A40 node (s8), any TP.
# Run 1 "nvlink": default NCCL  -> uses NVLink where present.
# Run 2 "pcie":   NCCL_P2P_DISABLE=1 -> no NVLink/P2P, all comm staged via host over PCIe.
# Everything else (image, weights, dtype, workload) is identical -> isolates NVLink.
#
# Usage: bash validation/run_vllm_tp_nvlink_ablation.sh [TP] [GPUS]
#   TP2 (NVLink pair):   ... 2 0,1
#   TP4 (2 NV pairs):    ... 4 0,1,2,3   (default)
set -euo pipefail

TP="${1:-4}"
GPUS="${2:-0,1,2,3}"
IMAGE="nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3"
LOCAL_MODEL="${LOCAL_MODEL:-/home/bdsl/hyungyuJung/data/models/Llama-3.1-8B-Instruct}"
SERVED="${SERVED:-meta-llama/Llama-3.1-8B}"
OUT_PREFIX="${OUT_PREFIX:-vllm_a40_tp${TP}}"     # output basename; e.g. vllm_a40_70b_tp4 for 70B
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PORT=$((8000 + TP))
DATASET="${DATASET:-dataset/sharegpt_req100_rate10_llama.jsonl}"
NUM_REQ="${NUM_REQ:-100}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

run_one() {
  local TAG="$1" P2P_DISABLE="$2"
  local CNAME="vllm_${OUT_PREFIX}_${TAG}"
  local RESULTS="$REPO/validation/${OUT_PREFIX}_${TAG}_results.jsonl"
  local SERVELOG="$REPO/validation/${OUT_PREFIX}_${TAG}_serve.log"
  local PWRLOG="$REPO/validation/${OUT_PREFIX}_${TAG}_power.csv"

  echo "############################################################"
  echo "# TP=$TP RUN '$TAG'  (NCCL_P2P_DISABLE=$P2P_DISABLE)  GPUs=$GPUS"
  echo "############################################################"
  docker rm -f "$CNAME" >/dev/null 2>&1 || true

  docker run -d --name "$CNAME" \
    --gpus "\"device=$GPUS\"" --ipc=host --shm-size=16g \
    --network host \
    -v "$LOCAL_MODEL:/model:ro" \
    -e HF_HUB_OFFLINE=1 \
    -e NCCL_P2P_DISABLE="$P2P_DISABLE" \
    -e NCCL_DEBUG=INFO \
    --entrypoint /bin/bash \
    "$IMAGE" -c "vllm serve /model --served-model-name '$SERVED' \
      --dtype float16 --tensor-parallel-size $TP \
      --distributed-executor-backend mp \
      --max-model-len 4096 --gpu-memory-utilization $GPU_MEM_UTIL \
      --port $PORT > /serve.log 2>&1"

  echo "==> waiting for health (up to 15 min) ..."
  local ok=0
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; echo "ready after ${i}*5s"; break; fi
    if ! docker ps --filter "name=$CNAME" --format '{{.Names}}' | grep -q "$CNAME"; then
      echo "ERROR: container exited. Last log:"; docker exec "$CNAME" tail -40 /serve.log 2>/dev/null || docker logs --tail 40 "$CNAME"; exit 1
    fi
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "ERROR: not ready"; docker exec "$CNAME" tail -60 /serve.log; docker rm -f "$CNAME"; exit 1; }

  docker exec "$CNAME" bash -c 'grep -iE "ustom.?all.?reduce|is disabled|via (NVL|P2P|SHM|direct)" /serve.log | grep -iE "ustom|isabled|Channel|->" | head -50' > "$SERVELOG" 2>/dev/null || true
  echo "==> NCCL transport lines (first few) -> $SERVELOG"
  head -6 "$SERVELOG" 2>/dev/null || true

  echo "==> power logging -> $PWRLOG"
  nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 -i "$GPUS" > "$PWRLOG" &
  local PWR_PID=$!

  echo "==> sending $NUM_REQ requests ..."
  cd "$REPO"
  python3 validation/send_requests.py --tp "$TP" --port "$PORT" \
    --model "$SERVED" --timeout "${REQ_TIMEOUT:-900}" \
    --dataset "$DATASET" --num-req "$NUM_REQ" --output "$RESULTS"

  kill "$PWR_PID" 2>/dev/null || true
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "==> TP=$TP run '$TAG' done -> $RESULTS"
  echo
}

trap 'docker rm -f vllm_${OUT_PREFIX}_nvlink vllm_${OUT_PREFIX}_pcie >/dev/null 2>&1 || true' EXIT

# ORDER controls run sequence (default nvlink first); set "pcie nvlink" to swap for order-effect control.
ORDER="${ORDER:-nvlink pcie}"
for tag in $ORDER; do
  case "$tag" in
    nvlink) run_one nvlink 0 ;;
    pcie)   run_one pcie   1 ;;
    *) echo "unknown run tag: $tag" ;;
  esac
done

echo "ALL DONE. results: validation/${OUT_PREFIX}_{nvlink,pcie}_results.jsonl"
