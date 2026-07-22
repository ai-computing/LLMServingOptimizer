#!/bin/bash
# Start a detached vLLM+Ray container on THIS node and form/join the Ray cluster over IB.
# All inter-node traffic (Ray bootstrap + NCCL) is pinned to the 200Gb/s InfiniBand fabric
# (mlx5_0, 192.168.210.x). RDMA needs /dev/infiniband + IPC_LOCK + unlimited memlock.
#
# Usage: run_cluster_node.sh <head|worker> <this_ib_ip> <head_ib_ip> <ib_iface> <model_host_path>
#   s8 (head):   run_cluster_node.sh head   192.168.210.108 192.168.210.108 ibs8       /home/bdsl/hyungyuJung/data/models/Llama-3.1-8B-Instruct
#   s2 (worker): run_cluster_node.sh worker 192.168.210.102 192.168.210.108 ibp194s0   /home/swsok/models/Llama-3.1-8B-Instruct
set -euo pipefail
ROLE="$1"; THIS_IP="$2"; HEAD_IP="$3"; IFACE="$4"; MODEL="$5"
IMAGE="nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3"
CNAME="vllm_mn"

docker rm -f "$CNAME" >/dev/null 2>&1 || true
# Explicitly pass each RDMA char device (a bare --device dir is not guaranteed to recurse);
# uverbs0 (user-verbs) + rdma_cm are what NCCL's IB transport opens.
IB_DEVS=""
for d in /dev/infiniband/uverbs0 /dev/infiniband/rdma_cm /dev/infiniband/issm0 /dev/infiniband/umad0; do
  [ -e "$d" ] && IB_DEVS="$IB_DEVS --device $d"
done
docker run -d --name "$CNAME" \
  --network host --ipc host --gpus all \
  $IB_DEVS --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --shm-size=16g \
  -v "$MODEL:/model:ro" \
  -e NCCL_IB_HCA=mlx5_0 \
  -e NCCL_SOCKET_IFNAME="$IFACE" \
  -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 \
  -e VLLM_HOST_IP="$THIS_IP" \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint /bin/bash \
  "$IMAGE" -c "sleep infinity"

# Ray must bind to the IB IP so the TP collective + control plane ride InfiniBand.
if [ "$ROLE" = "head" ]; then
  docker exec "$CNAME" ray start --head \
    --node-ip-address="$THIS_IP" --port=6379 --dashboard-host=0.0.0.0
else
  docker exec "$CNAME" ray start \
    --address="$HEAD_IP:6379" --node-ip-address="$THIS_IP"
fi
echo "[$ROLE] container '$CNAME' up, ray started on $THIS_IP (iface $IFACE)"
