"""
vLLM Docker 벤치마크: Single GPU vs 2-GPU DP

vLLM OpenAI 호환 서버를 Docker로 실행 후 HTTP 요청으로 처리량 측정.
HuggingFace Transformers 대비 vLLM의 continuous batching / PagedAttention 효과 측정.

실행 방법:
  python examples/benchmark_vllm_docker.py --mode single
  python examples/benchmark_vllm_docker.py --mode dp
  python examples/benchmark_vllm_docker.py --mode both
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_HOST_PATH  = "/home/swsok/models/Llama-3.1-8B-Instruct"
MODEL_IN_DOCKER  = "/models/Llama-3.1-8B-Instruct"
VLLM_IMAGE       = "vllm/vllm-openai:latest"

NUM_REQUESTS  = 100       # 총 요청 수
MAX_TOKENS    = 128       # 최대 생성 토큰
CONCURRENCY   = 16        # 동시 요청 수 (vLLM continuous batching 활용)
PROMPT        = "Explain the concept of neural networks in detail."
READY_TIMEOUT = 300       # 서버 준비 대기 최대 초


# ── Docker 헬퍼 ───────────────────────────────────────────────────────────────
_containers: list[str] = []   # 실행 중인 컨테이너 ID (cleanup용)

def _cleanup(sig=None, frame=None):
    """종료 시 모든 컨테이너 제거."""
    for cid in _containers:
        subprocess.run(["docker", "rm", "-f", cid],
                       capture_output=True, check=False)
    if sig is not None:
        sys.exit(0)

signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


def start_vllm_container(gpu_id: int, port: int, host_model_path: str) -> str:
    """vLLM 서버 컨테이너 시작 후 container ID 반환."""
    name = f"vllm_bench_gpu{gpu_id}"
    # 기존 동명 컨테이너 제거
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    cmd = [
        "docker", "run", "-d",
        "--gpus", f"device={gpu_id}",
        # Always mount to MODEL_IN_DOCKER so requests can use a fixed model name.
        "-v", f"{host_model_path}:{MODEL_IN_DOCKER}:ro",
        "-p", f"{port}:8000",
        "--name", name,
        "--shm-size", "10g",
        "--ipc", "host",
        # CUDA 12.9 compat lib in the image conflicts with host driver 595.
        # Force the toolkit-injected host libcuda.so to take priority.
        "-e", "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu",
        VLLM_IMAGE,
        "--model", MODEL_IN_DOCKER,
        "--dtype", "float16",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "64",
        "--disable-log-requests",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cid = result.stdout.strip()
    _containers.append(cid)
    return cid


def wait_for_ready(port: int, timeout: int = READY_TIMEOUT) -> bool:
    """vLLM /health 엔드포인트가 응답할 때까지 대기."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    interval = 2.0
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


def stop_container(cid: str):
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
    if cid in _containers:
        _containers.remove(cid)


# ── 추론 요청 ─────────────────────────────────────────────────────────────────
def _single_request(port: int) -> int:
    """completions API로 요청 1건 전송 → 생성 토큰 수 반환."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": MODEL_IN_DOCKER,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def benchmark_port(port: int, num_requests: int, label: str) -> dict:
    """지정 포트로 num_requests개 요청을 CONCURRENCY 단위로 전송."""
    total_tokens = 0
    errors = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(_single_request, port) for _ in range(num_requests)]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                total_tokens += fut.result()
            except Exception as e:
                errors += 1
                print(f"  [{label}] 요청 실패: {e}")

    elapsed = time.perf_counter() - t0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    return {"tokens": total_tokens, "elapsed": elapsed,
            "throughput": throughput, "errors": errors}


# ── 벤치마크 시나리오 ─────────────────────────────────────────────────────────
def benchmark_single(model_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"[Single GPU] GPU 0  —  {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")
    print(f"{'='*60}")

    print("  컨테이너 시작 중…")
    cid = start_vllm_container(0, 8100, model_path)

    print("  서버 준비 대기 중…", end="", flush=True)
    if not wait_for_ready(8100):
        print(" TIMEOUT")
        stop_container(cid)
        return {}
    print(" Ready")

    res = benchmark_port(8100, NUM_REQUESTS, "GPU0")

    print(f"  Elapsed    : {res['elapsed']:.2f} s")
    print(f"  Tokens     : {res['tokens']}  (errors={res['errors']})")
    print(f"  Throughput : {res['throughput']:.1f} tokens/s")

    stop_container(cid)
    return res


def benchmark_dp(model_path: str) -> dict:
    """GPU 0 + GPU 1 각각 NUM_REQUESTS/2개씩 동시에 처리."""
    half = NUM_REQUESTS // 2
    print(f"\n{'='*60}")
    print(f"[DP 2-GPU] GPU 0 + GPU 1  —  {NUM_REQUESTS} requests ({half}+{half}), concurrency={CONCURRENCY}")
    print(f"{'='*60}")

    print("  컨테이너 2개 시작 중…")
    cid0 = start_vllm_container(0, 8200, model_path)
    cid1 = start_vllm_container(1, 8201, model_path)

    print("  서버 준비 대기 중 (GPU 0, GPU 1)…", end="", flush=True)
    ready0 = wait_for_ready(8200)
    ready1 = wait_for_ready(8201)
    if not (ready0 and ready1):
        print(f" TIMEOUT (gpu0={ready0}, gpu1={ready1})")
        stop_container(cid0)
        stop_container(cid1)
        return {}
    print(" Ready")

    results: dict[str, dict] = {}
    wall_start = time.perf_counter()

    # 두 GPU에 동시에 요청 전송
    with ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(benchmark_port, 8200, half, "GPU0")
        f1 = pool.submit(benchmark_port, 8201, half, "GPU1")
        results[0] = f0.result()
        results[1] = f1.result()

    wall_elapsed = time.perf_counter() - wall_start
    total_tokens = results[0]["tokens"] + results[1]["tokens"]
    throughput   = total_tokens / wall_elapsed

    for gid in (0, 1):
        r = results[gid]
        print(f"  GPU {gid}: {r['tokens']} tokens in {r['elapsed']:.2f} s  "
              f"({r['throughput']:.1f} tok/s, errors={r['errors']})")
    print(f"  Wall-clock : {wall_elapsed:.2f} s")
    print(f"  Tokens     : {total_tokens}")
    print(f"  Combined DP throughput: {throughput:.1f} tokens/s")

    stop_container(cid0)
    stop_container(cid1)
    return {"elapsed": wall_elapsed, "tokens": total_tokens,
            "throughput": throughput}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global NUM_REQUESTS, MAX_TOKENS, CONCURRENCY

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["single", "dp", "both"], default="both")
    parser.add_argument("--model",       default=MODEL_HOST_PATH)
    parser.add_argument("--num-requests",  type=int, default=NUM_REQUESTS)
    parser.add_argument("--max-tokens",    type=int, default=MAX_TOKENS)
    parser.add_argument("--concurrency",   type=int, default=CONCURRENCY)
    args = parser.parse_args()

    NUM_REQUESTS = args.num_requests
    MAX_TOKENS   = args.max_tokens
    CONCURRENCY  = args.concurrency

    single_result = dp_result = None

    if args.mode in ("single", "both"):
        single_result = benchmark_single(args.model)

    if args.mode in ("dp", "both"):
        dp_result = benchmark_dp(args.model)

    if single_result and dp_result:
        speedup = dp_result["throughput"] / single_result["throughput"]
        print(f"\n{'='*60}")
        print(f"결과 비교 (vLLM Docker)")
        print(f"{'='*60}")
        print(f"  Single GPU throughput : {single_result['throughput']:.1f} tokens/s")
        print(f"  DP 2-GPU   throughput : {dp_result['throughput']:.1f} tokens/s")
        print(f"  Speedup (DP / Single) : {speedup:.2f}×")
        print(f"  이론적 최대 speedup   : 2.00×")
        print(f"  효율                  : {speedup/2*100:.1f}%")

    _cleanup()


if __name__ == "__main__":
    main()
