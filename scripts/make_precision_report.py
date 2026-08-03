#!/usr/bin/env python3
"""Turn the precision-study JSON records into a markdown report.

    python scripts/make_precision_report.py --out docs/PRECISION_STUDY_A5000.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "output" / "precision_study"

PRECISIONS = ["fp16", "fp8", "int8", "int4"]
GROUPS = [
    ("Llama-3.1-8B-Instruct (TP1)", "llama8b-{p}", PRECISIONS),
    ("Qwen2.5-14B-Instruct (TP2)", "qwen14b-{p}-tp2", PRECISIONS),
    ("Qwen2.5-14B-Instruct (TP1)", "qwen14b-{p}-tp1", ["int8", "int4"]),
]
#: nominal weight footprint, GiB (params x bits/8, int4 at ~4.5 effective bits)
WEIGHTS_GIB = {"llama8b": {"fp16": 15.0, "fp8": 7.5, "int8": 7.5, "int4": 4.2},
               "qwen14b": {"fp16": 27.5, "fp8": 13.8, "int8": 13.8, "int4": 7.8}}


def load() -> dict[str, dict]:
    out = {}
    for f in sorted(IN_DIR.glob("*.json")):
        if f.name == "study_summary.json":
            continue
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return out


def pt(rec: dict, c: int) -> dict | None:
    for p in rec.get("points", []):
        if p["concurrency"] == c:
            return p
    return None


def peak(rec: dict) -> dict:
    return max(rec["points"], key=lambda p: p["thr_toks_s"])


def fmt(v, spec="{:.1f}", dash="—"):
    return dash if v is None else spec.format(v)


def table_capacity(recs: dict) -> list[str]:
    L = ["| 구성 | 정밀도 | 가중치(추정) | KV 캐시 토큰 | idle W | 로드된 GPU |",
         "|---|---|---|---|---|---|"]
    for title, pat, precs in GROUPS:
        family = "llama8b" if "llama8b" in pat else "qwen14b"
        for p in precs:
            r = recs.get(pat.format(p=p))
            if not r:
                continue
            L.append(f"| {title} | **{p}** | {WEIGHTS_GIB[family][p]:.1f} GiB | "
                     f"{r.get('kv_cache_tokens') or '—':,} | {r['idle_w']:.1f} | "
                     f"{r['tp']}x A5000 |".replace("'", ""))
    return L


def table_sweep(recs: dict, title: str, pat: str, precs: list[str]) -> list[str]:
    cs = sorted({p["concurrency"] for pr in precs
                 if (r := recs.get(pat.format(p=pr))) for p in r["points"]})
    if not cs:
        return []
    L = [f"#### {title}", "",
         "| 동시성 | " + " | ".join(f"{p} tok/s" for p in precs) +
         " | " + " | ".join(f"{p} TPOT(ms)" for p in precs) + " |",
         "|" + "---|" * (1 + 2 * len(precs))]
    for c in cs:
        thr, tpot = [], []
        for pr in precs:
            r = recs.get(pat.format(p=pr))
            q = pt(r, c) if r else None
            thr.append(fmt(q["thr_toks_s"] if q else None, "{:.0f}"))
            tpot.append(fmt(q["tpot_ms"] if q else None, "{:.1f}"))
        L.append(f"| {c} | " + " | ".join(thr) + " | " + " | ".join(tpot) + " |")
    return L + [""]


def table_efficiency(recs: dict) -> list[str]:
    L = ["| 구성 | 정밀도 | 피크 처리량 | @동시성 | 그때 전력 | **tok/J** | "
         "1M토큰 에너지 | vs fp16 |", "|---|---|---|---|---|---|---|---|"]
    for title, pat, precs in GROUPS:
        base = recs.get(pat.format(p="fp16")) or recs.get(
            pat.replace("{p}", "int8"))
        base_tpj = None
        if base:
            b = peak(base)
            base_tpj = b["thr_toks_s"] / b["avg_w"] if b["avg_w"] else None
        for p in precs:
            r = recs.get(pat.format(p=p))
            if not r:
                continue
            q = peak(r)
            tpj = q["thr_toks_s"] / q["avg_w"] if q["avg_w"] else None
            wh = (1e6 / q["thr_toks_s"] * q["avg_w"] / 3600) if q["thr_toks_s"] else None
            rel = f"{tpj / base_tpj:.2f}x" if (tpj and base_tpj) else "—"
            L.append(f"| {title} | **{p}** | {q['thr_toks_s']:.0f} tok/s | "
                     f"c={q['concurrency']} | {q['avg_w']:.0f} W | "
                     f"**{fmt(tpj, '{:.2f}')}** | {fmt(wh, '{:.1f}')} Wh | {rel} |")
    return L


def table_quality(recs: dict) -> list[str]:
    L = ["| 구성 | 정밀도 | GSM8K (200문제, 5-shot, greedy) | fp16 대비 | 요청 오류 |",
         "|---|---|---|---|---|"]
    for title, pat, precs in GROUPS:
        base = recs.get(pat.format(p="fp16"))
        bacc = (base or {}).get("quality", {}).get("accuracy") if base else None
        for p in precs:
            r = recs.get(pat.format(p=p))
            q = (r or {}).get("quality") or {}
            acc = q.get("accuracy")
            delta = (f"{(acc - bacc) * 100:+.1f}pp" if (acc is not None and bacc)
                     else "—")
            L.append(f"| {title} | **{p}** | "
                     f"{fmt(acc * 100 if acc is not None else None, '{:.1f}')}% "
                     f"({q.get('correct', '—')}/{q.get('n', '—')}) | {delta} | "
                     f"{q.get('request_errors', '—')} |")
    return L


def build(recs: dict) -> str:
    import datetime
    stack = next((r.get("stack") for r in recs.values() if r.get("stack")), "?")
    L = [
        "# A5000 x2 정밀도 실험: fp16 / fp8 / int8 / int4",
        "",
        f"날짜: {datetime.date.today().isoformat()} · 하드웨어: NVIDIA RTX A5000 24GB x2 "
        "(Ampere SM 8.6, NVLink 없음 — 호스트 브리지 PCIe) · 스택: " + str(stack),
        "",
        "## 요약", "", "<!--SUMMARY-->", "",
        "## 실험 설계", "",
        "- **모델**: Llama-3.1-8B-Instruct(모든 정밀도가 1장에 적재 → TP 교란 없는 순수 "
        "정밀도 비교), Qwen2.5-14B-Instruct(fp16이 27.5 GiB라 **TP2 필수** → 양자화가 "
        "최소 병렬도를 바꾸는 사례)",
        "- **부하**: vLLM `bench serve`, random 데이터셋 입력 256 / 출력 256 토큰, "
        "`--ignore-eos`, 동시성 스윕. 각 동시성마다 nvidia-smi로 전력을 병행 샘플링",
        "- **품질**: GSM8K 테스트 200문제, 5-shot 고정 프롬프트, greedy, 마지막 숫자 "
        "매칭(`scripts/eval_gsm8k.py`). 리더보드 재현이 아니라 **동일 프롬프트에서 "
        "정밀도 간 상대 비교**가 목적",
        "- **공통 설정**: `--dtype auto`(체크포인트의 양자화 설정 존중), "
        "max-model-len 4096, max-num-seqs 128, max-num-batched-tokens 2048, "
        "block-size 16, gpu-memory-utilization 0.90",
        "",
        "### A5000(Ampere)의 정밀도 지원 특성", "",
        "vLLM에 직접 질의한 결과 `supports_fp8() = False`(compute capability 8.6)입니다. "
        "즉 **FP8 연산기가 없어** FP8 체크포인트는 Marlin weight-only 경로로 로드되어 "
        "메모리만 절약하고 계산은 FP16으로 수행됩니다. 반면 **INT8 W8A8은 INT8 Tensor "
        "Core를 실제로 사용**하고, INT4(AWQ)는 weight-only 양자화라 가중치 판독량을 "
        "1/4로 줄이는 방식으로 이득을 냅니다.",
        "",
        "## 1. 메모리와 KV 용량", "",
    ]
    L += table_capacity(recs)
    L += ["", "## 2. 처리량과 지연", ""]
    for title, pat, precs in GROUPS:
        L += table_sweep(recs, title, pat, precs)
    L += ["## 3. 전력 효율 (이 프로젝트의 핵심 지표)", "",
          "`tok/J` = 피크 처리량 / 그 시점 평균 전력. 1M토큰 에너지는 해당 "
          "처리량·전력으로 100만 토큰을 생성할 때의 에너지입니다.", ""]
    L += table_efficiency(recs)
    L += ["", "## 4. 품질 (GSM8K)", ""]
    L += table_quality(recs)
    L += ["", "## 5. 분석", "", "<!--ANALYSIS-->", "",
          "## 6. 한계", "",
          "- GSM8K 200문제·5-shot 단일 프롬프트이므로 절대 점수는 공개 리더보드와 "
          "다릅니다. 정밀도 간 **상대 비교**로만 해석하십시오(±3pp 내 차이는 표본 "
          "오차와 구분되지 않습니다).",
          "- 입력/출력 길이를 256/256으로 고정했습니다. 긴 컨텍스트에서는 KV 용량 "
          "차이가 더 크게 작용합니다.",
          "- 전력은 nvidia-smi 샘플링(GPU만)이며 호스트 전력은 포함하지 않습니다.",
          "- 동일 모델의 양자화 체크포인트는 서로 다른 조직이 서로 다른 캘리브레이션 "
          "데이터로 만든 것이라, 정밀도 자체의 효과와 체크포인트 품질 차이가 섞여 "
          "있습니다.",
          "",
          "## 재현", "",
          "```bash",
          "python scripts/run_precision_study.py              # 전체 매트릭스",
          "python scripts/run_precision_study.py llama8b-int4 # 개별 설정",
          "python scripts/make_precision_report.py --out docs/PRECISION_STUDY_A5000.md",
          "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/PRECISION_STUDY_A5000.md"))
    a = ap.parse_args()
    recs = load()
    if not recs:
        print("no records in", IN_DIR)
        return 1
    Path(a.out).write_text(build(recs), encoding="utf-8")
    print(f"wrote {a.out} from {len(recs)} record(s): {sorted(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
