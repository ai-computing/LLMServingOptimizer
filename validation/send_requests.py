"""
vLLM 요청 전송 및 TTFT/TPOT 실측 스크립트.

Usage:
    python3 validation/send_requests.py --tp 1 [--port 8001] [--dataset ...] [--num-req 300]

동작:
  - JSONL 데이터셋의 arrival_time_ns를 재현해 vLLM에 요청을 전송한다.
  - SSE 스트리밍으로 per-request TTFT와 TPOT를 측정한다.
  - 결과를 validation/vllm_tp{n}_results.jsonl 에 저장한다.
"""

from __future__ import annotations  # PEP 604 (int | None) hints under Python 3.8

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

DATASET_DEFAULT = "dataset/sharegpt_req300_rate10_llama.jsonl"
MODEL = "meta-llama/Llama-3.1-8B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, required=True, choices=[1, 2, 4, 8, 16], help="Tensor parallel size")
    p.add_argument("--port", type=int, default=None, help="vLLM port (default: 8001 for tp=1, 8002 for tp=2)")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--num-req", type=int, default=None, help="Number of requests (default: all)")
    p.add_argument("--output", default=None, help="Output JSONL path")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s)")
    return p.parse_args()


def load_dataset(path: str, num_req: int | None):
    reqs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if num_req and i >= num_req:
                break
            reqs.append(json.loads(line))
    return reqs


async def send_one(session: aiohttp.ClientSession, url: str, model: str,
                   req: dict, req_idx: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": req["input_tok_ids"],   # OpenAI spec: prompt as token ID list
        "max_tokens": req["output_toks"],
        "temperature": 0.0,
        "stream": True,
    }

    t_send = time.perf_counter_ns()
    first_token_ns: int | None = None
    token_times: list[int] = []
    actual_output = 0

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "req_idx": req_idx,
                    "error": f"HTTP {resp.status}: {body[:200]}",
                    "input_toks": req["input_toks"],
                    "output_toks": req["output_toks"],
                    "arrival_time_ns": req["arrival_time_ns"],
                }

            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    token_text = choices[0].get("text", "")
                    finish = choices[0].get("finish_reason")
                    # Count only chunks that carry actual content or finish
                    if token_text or finish:
                        t_now = time.perf_counter_ns()
                        if first_token_ns is None:
                            first_token_ns = t_now
                        if token_text:
                            token_times.append(t_now)
                            actual_output += 1
                        if finish:
                            break
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        return {
            "req_idx": req_idx,
            "error": "timeout",
            "input_toks": req["input_toks"],
            "output_toks": req["output_toks"],
            "arrival_time_ns": req["arrival_time_ns"],
        }

    t_end = time.perf_counter_ns()

    ttft_ns = (first_token_ns - t_send) if first_token_ns is not None else None
    if first_token_ns is not None and len(token_times) > 1:
        itls = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]
        tpot_ns = int(sum(itls) / len(itls))
    elif first_token_ns is not None and actual_output > 0:
        tpot_ns = int((t_end - first_token_ns) / max(actual_output - 1, 1))
    else:
        tpot_ns = None

    return {
        "req_idx": req_idx,
        "input_toks": req["input_toks"],
        "output_toks": req["output_toks"],
        "actual_output_toks": actual_output,
        "arrival_time_ns": req["arrival_time_ns"],
        "ttft_ns": ttft_ns,
        "tpot_ns": tpot_ns,
        "total_latency_ns": t_end - t_send,
    }


async def run(args):
    port = args.port or (8001 if args.tp == 1 else 8002)
    server = f"http://localhost:{port}"
    url = f"{server}/v1/completions"
    out_path = args.output or f"validation/vllm_tp{args.tp}_results.jsonl"

    reqs = load_dataset(args.dataset, args.num_req)
    print(f"Loaded {len(reqs)} requests from {args.dataset}")
    print(f"Target server: {url}")
    print(f"Output: {out_path}")

    # Wait for server to be ready
    connector = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("Checking server health...")
        for attempt in range(60):
            try:
                async with session.get(f"{server}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1.0)
            if attempt % 10 == 9:
                print(f"  Still waiting... ({attempt+1}s)")
        else:
            print("ERROR: Server did not become ready within 60s")
            return

        print("Server ready. Starting request replay...")

        results: dict[int, dict] = {}
        tasks: list[asyncio.Task] = []
        t0 = time.perf_counter_ns()

        for idx, req in enumerate(reqs):
            # Sleep until the scheduled arrival time
            target_ns = t0 + req["arrival_time_ns"]
            now_ns = time.perf_counter_ns()
            if target_ns > now_ns:
                await asyncio.sleep((target_ns - now_ns) / 1e9)

            task = asyncio.create_task(
                send_one(session, url, args.model, req, idx, args.timeout)
            )
            tasks.append(task)

            if idx % 50 == 49:
                print(f"  Dispatched {idx+1}/{len(reqs)} requests...")

        print(f"All {len(reqs)} requests dispatched. Waiting for completion...")
        done = await asyncio.gather(*tasks)

    for r in done:
        results[r["req_idx"]] = r

    # Write results sorted by request index
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for idx in sorted(results):
            f.write(json.dumps(results[idx]) + "\n")

    # Summary
    valid = [r for r in done if "error" not in r and r.get("ttft_ns") is not None]
    errors = [r for r in done if "error" in r]
    if valid:
        ttfts_ms = sorted(r["ttft_ns"] / 1e6 for r in valid)
        tpots_ms = sorted(r["tpot_ns"] / 1e6 for r in valid if r.get("tpot_ns"))
        n = len(ttfts_ms)
        print(f"\n=== Results ({len(valid)}/{len(reqs)} succeeded, {len(errors)} errors) ===")
        print(f"TTFT  p50={ttfts_ms[n//2]:.1f}ms  p99={ttfts_ms[int(n*0.99)]:.1f}ms")
        if tpots_ms:
            m = len(tpots_ms)
            print(f"TPOT  p50={tpots_ms[m//2]:.1f}ms  p99={tpots_ms[int(m*0.99)]:.1f}ms")
        total_out = sum(r["actual_output_toks"] for r in valid)
        elapsed_s = (time.perf_counter_ns() - t0) / 1e9  # rough
        print(f"Throughput ≈ {total_out / elapsed_s:.1f} tok/s (output tokens)")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
