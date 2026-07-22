"""
Llama-3.1-8B 단일 GPU vs 2-GPU DP 처리량 벤치마크 (HuggingFace Transformers)

실행 방법:
  python examples/benchmark_dp.py --mode single --model /home/swsok/models/Llama-3.1-8B-Instruct
  python examples/benchmark_dp.py --mode dp     --model /home/swsok/models/Llama-3.1-8B-Instruct
  python examples/benchmark_dp.py --mode both   --model /home/swsok/models/Llama-3.1-8B-Instruct
"""
import argparse
import multiprocessing
import os
import time

MODEL_DEFAULT = "/home/swsok/models/Llama-3.1-8B-Instruct"
NUM_PROMPTS   = 50
MAX_NEW_TOKENS = 128
BATCH_SIZE    = 8
PROMPT        = "Explain the concept of neural networks in detail."


def run_on_gpu(gpu_id: int, model_path: str, prompts: list, result_queue):
    """단일 GPU에서 Transformers로 추론 후 (토큰 수, 경과 시간) 반환."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()

    total_tokens = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to("cuda:0")
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Count only newly generated tokens
            total_tokens += sum(
                len(out) - len(inp)
                for out, inp in zip(outputs, inputs["input_ids"])
            )

    elapsed = time.perf_counter() - t0
    result_queue.put((total_tokens, elapsed, gpu_id))


def benchmark_single(model: str) -> dict:
    print(f"\n{'='*60}")
    print(f"[Single GPU] GPU 0, {NUM_PROMPTS} requests (batch={BATCH_SIZE})")
    print(f"{'='*60}")

    prompts = [PROMPT] * NUM_PROMPTS
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_on_gpu, args=(0, model, prompts, q))
    p.start()
    p.join()

    total_tokens, elapsed, _ = q.get()
    throughput = total_tokens / elapsed

    print(f"  Elapsed     : {elapsed:.2f} s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput  : {throughput:.1f} tokens/s")
    return {"elapsed": elapsed, "tokens": total_tokens, "throughput": throughput}


def benchmark_dp(model: str) -> dict:
    """GPU 0, GPU 1 각각 NUM_PROMPTS/2개씩 동시에 처리 (Data Parallelism)."""
    print(f"\n{'='*60}")
    print(f"[DP 2-GPU] GPU 0 + GPU 1, {NUM_PROMPTS} requests split evenly")
    print(f"{'='*60}")

    half = NUM_PROMPTS // 2
    q = multiprocessing.Queue()
    p0 = multiprocessing.Process(target=run_on_gpu, args=(0, model, [PROMPT]*half, q))
    p1 = multiprocessing.Process(target=run_on_gpu, args=(1, model, [PROMPT]*half, q))

    wall_start = time.perf_counter()
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    wall_elapsed = time.perf_counter() - wall_start

    results = [q.get() for _ in range(2)]
    total_tokens = sum(r[0] for r in results)
    throughput = total_tokens / wall_elapsed

    for tokens, elapsed, gid in sorted(results, key=lambda x: x[2]):
        print(f"  GPU {gid}: {tokens} tokens in {elapsed:.2f} s  ({tokens/elapsed:.1f} tok/s)")
    print(f"  Wall-clock  : {wall_elapsed:.2f} s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Combined DP throughput: {throughput:.1f} tokens/s")
    return {"elapsed": wall_elapsed, "tokens": total_tokens, "throughput": throughput}


def main():
    global NUM_PROMPTS, MAX_NEW_TOKENS, BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["single", "dp", "both"], default="both")
    parser.add_argument("--model",      default=MODEL_DEFAULT)
    parser.add_argument("--num-prompts",  type=int, default=NUM_PROMPTS)
    parser.add_argument("--max-tokens",   type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    NUM_PROMPTS    = args.num_prompts
    MAX_NEW_TOKENS = args.max_tokens
    BATCH_SIZE     = args.batch_size

    single_result = dp_result = None

    if args.mode in ("single", "both"):
        single_result = benchmark_single(args.model)

    if args.mode in ("dp", "both"):
        dp_result = benchmark_dp(args.model)

    if single_result and dp_result:
        speedup = dp_result["throughput"] / single_result["throughput"]
        print(f"\n{'='*60}")
        print(f"결과 비교")
        print(f"{'='*60}")
        print(f"  Single GPU throughput : {single_result['throughput']:.1f} tokens/s")
        print(f"  DP 2-GPU   throughput : {dp_result['throughput']:.1f} tokens/s")
        print(f"  Speedup (DP / Single) : {speedup:.2f}×")
        print(f"  이론적 최대 speedup   : 2.00×")
        print(f"  효율                  : {speedup/2*100:.1f}%")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
