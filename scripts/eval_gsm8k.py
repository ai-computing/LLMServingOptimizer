#!/usr/bin/env python3
"""Minimal GSM8K accuracy check against a running OpenAI-compatible server.

Speed numbers alone cannot rank quantization schemes — a scheme that is 2x
faster and 10 points less accurate is not "better". This is a deliberately
small, transparent harness (no lm-eval dependency in the serving venv):

* fixed 5-shot prompt, identical for every precision, greedy decoding
* answer = the last number in the completion, compared to GSM8K's gold value
* requests issued concurrently so a full run costs a couple of minutes

It is a *relative* check between precisions of the same model on the same
prompts, not an attempt to reproduce published GSM8K leaderboard numbers.

Usage:
    python scripts/eval_gsm8k.py --base-url http://localhost:8100 \
        --model <served model id> --n 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 5-shot exemplars (from the GSM8K train split, abbreviated reasoning)
FEWSHOT = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half "
     "as many clips in May. How many clips did Natalia sell altogether in "
     "April and May?",
     "In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 "
     "clips. The answer is 72."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?",
     "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned "
     "0.2 x 50 = $10. The answer is 10."),
    ("Betty is saving money for a new wallet which costs $100. Betty has only "
     "half of the money she needs. Her parents decided to give her $15 for "
     "that purpose, and her grandparents twice as much as her parents. How "
     "much more money does Betty need to buy the wallet?",
     "Betty has 100 / 2 = $50. Her grandparents gave 15 x 2 = $30. In total "
     "she has 50 + 15 + 30 = $95, so she needs 100 - 95 = $5. The answer is 5."),
    ("James writes a 3-page letter to 2 different friends twice a week. How "
     "many pages does he write a year?",
     "Each time he writes 3 x 2 = 6 pages. Twice a week that is 6 x 2 = 12 "
     "pages. In a year that is 12 x 52 = 624 pages. The answer is 624."),
    ("Mark has a garden with flowers. He planted plants of three different "
     "colors in it. Ten of them are yellow, and there are 80% more of those "
     "in purple. There are only 25% as many green flowers as there are yellow "
     "and purple flowers. How many flowers does Mark have in his garden?",
     "Purple flowers are 10 + 80% of 10 = 18. Yellow and purple together are "
     "10 + 18 = 28. Green flowers are 25% of 28 = 7. In total there are "
     "28 + 7 = 35 flowers. The answer is 35."),
]

_PROMPT_HEAD = (
    "Solve the grade school math problem. Reason briefly, then finish with a "
    "line of the form 'The answer is <number>.'\n\n"
)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def build_prompt(question: str) -> str:
    parts = [_PROMPT_HEAD]
    for q, a in FEWSHOT:
        parts.append(f"Question: {q}\nAnswer: {a}\n\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)


def extract_number(text: str) -> str | None:
    """Prefer the number after the last 'answer is', else the last number."""
    tail = text.rsplit("answer is", 1)
    hunt = tail[1] if len(tail) > 1 else text
    nums = _NUM.findall(hunt) or _NUM.findall(text)
    if not nums:
        return None
    return nums[0 if len(tail) > 1 else -1].replace(",", "").rstrip(".")


def gold_number(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def load_problems(n: int) -> list[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [(ds[i]["question"], gold_number(ds[i]["answer"]))
            for i in range(min(n, len(ds)))]


def _complete(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 0,
        "stop": ["\nQuestion:", "\n\nQuestion:"],
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


def evaluate_gsm8k(base_url: str, model: str, n: int = 200,
                   concurrency: int = 16, max_tokens: int = 256) -> dict:
    problems = load_problems(n)

    def one(item):
        q, gold = item
        try:
            out = _complete(base_url, model, build_prompt(q), max_tokens)
        except Exception as e:  # a failed request counts as wrong, not fatal
            return {"ok": False, "error": f"{type(e).__name__}"}
        pred = extract_number(out)
        return {"ok": pred is not None and _same(pred, gold),
                "pred": pred, "gold": gold}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, problems))
    correct = sum(1 for r in results if r["ok"])
    errors = sum(1 for r in results if r.get("error"))
    return {"task": "gsm8k", "shots": len(FEWSHOT), "n": len(results),
            "correct": correct, "accuracy": round(correct / len(results), 4),
            "request_errors": errors, "greedy": True}


def _same(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred == gold


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8100")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = evaluate_gsm8k(a.base_url, a.model, a.n, a.concurrency)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
