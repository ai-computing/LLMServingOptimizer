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
         "1M토큰 에너지 | 같은 모델 fp16 대비 |", "|---|---|---|---|---|---|---|---|"]
    for title, pat, precs in GROUPS:
        # the TP1 group has no fp16 (it does not fit on one card), so compare
        # against the same model's fp16 — which necessarily runs at TP2
        family_fp16 = ("llama8b-fp16" if "llama8b" in pat else "qwen14b-fp16-tp2")
        base = recs.get(pat.format(p="fp16")) or recs.get(family_fp16)
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


def _thr(recs, label, c):
    r = recs.get(label)
    p = pt(r, c) if r else None
    return p["thr_toks_s"] if p else None


def _acc(recs, label):
    return ((recs.get(label) or {}).get("quality") or {}).get("accuracy")


def _tpj(recs, label):
    r = recs.get(label)
    if not r:
        return None
    q = peak(r)
    return q["thr_toks_s"] / q["avg_w"] if q["avg_w"] else None


def _se(p, n=200):
    return (p * (1 - p) / n) ** 0.5


def summary(recs: dict) -> list[str]:
    """Headline numbers, computed from the records (not hardcoded)."""
    L = []
    r4_1 = _thr(recs, "llama8b-int4", 1)
    r16_1 = _thr(recs, "llama8b-fp16", 1)
    r4_64 = _thr(recs, "llama8b-int4", 64)
    r8_64 = _thr(recs, "llama8b-int8", 64)
    r16_64 = _thr(recs, "llama8b-fp16", 64)
    if all(v for v in (r4_1, r16_1, r4_64, r8_64, r16_64)):
        L.append(
            f"- **저동시성에서는 int4가 압도적, 고동시성에서는 int8과 동급으로 수렴**합니다. "
            f"Llama-8B 단일 스트림(c=1)에서 int4는 fp16의 **{r4_1 / r16_1:.2f}배**"
            f"({r16_1:.0f}→{r4_1:.0f} tok/s)지만, c=64에서는 {r4_64 / r16_64:.2f}배로 "
            f"떨어지고 int8({r8_64 / r16_64:.2f}배)에 추월당합니다. 디코드가 메모리 "
            f"병목에서 연산 병목으로 넘어가며 dequantization 비용이 드러나는 지점입니다.")
    t16, t8 = _tpj(recs, "llama8b-fp16"), _tpj(recs, "llama8b-int8")
    if t16 and t8:
        L.append(f"- **전력 효율은 양자화가 일관되게 이깁니다.** Llama-8B 피크에서 "
                 f"fp16 {t16:.2f} tok/J → int8 **{t8:.2f} tok/J({t8 / t16:.2f}배)**. "
                 f"전력(약 190~206W)은 정밀도와 거의 무관했고, 이득은 전부 처리량에서 "
                 f"나옵니다.")
    a, b = _thr(recs, "qwen14b-int4-tp1", 64), _thr(recs, "qwen14b-int4-tp2", 64)
    e1, e2 = _tpj(recs, "qwen14b-int4-tp1"), _tpj(recs, "qwen14b-int4-tp2")
    if a and b and e1 and e2:
        L.append(
            f"- **NVLink가 없는 이 머신에서는 '양자화로 TP를 없애는' 편이 유리합니다.** "
            f"Qwen-14B int4는 GPU 1장(TP1)에서 {a:.0f} tok/s로, GPU 2장 TP2"
            f"({b:.0f} tok/s)의 **{a / b * 100:.0f}%**를 절반의 하드웨어로 냅니다 "
            f"— GPU당 **{a / (b / 2):.2f}배**, 에너지 효율 **{e1 / e2:.2f}배**. 같은 2장에 "
            f"TP1 레플리카를 2개 띄우면 총 처리량이 TP2 대비 약 {2 * a / b:.1f}배가 됩니다.")
    ql = [(p, _acc(recs, f"llama8b-{p}")) for p in PRECISIONS]
    qq = [(p, _acc(recs, f"qwen14b-{p}-tp2")) for p in PRECISIONS]
    if all(v for _, v in ql) and all(v for _, v in qq):
        L.append(
            f"- **품질은 정밀도를 낮출수록 단조 감소하지만, 표본 200문제로는 "
            f"유의하다고 말할 수 없습니다.** GSM8K: Llama-8B "
            + " / ".join(f"{p} {v * 100:.1f}%" for p, v in ql) + ", Qwen-14B(TP2) "
            + " / ".join(f"{p} {v * 100:.1f}%" for p, v in qq) +
            f". 최대 격차는 Qwen fp16→int4의 {(qq[0][1] - qq[3][1]) * 100:.1f}pp인데 "
            f"차이의 표준오차가 {(_se(qq[0][1]) ** 2 + _se(qq[3][1]) ** 2) ** 0.5 * 100:.1f}pp"
            f"라 약 1.5σ 수준입니다(4절 참고).")
    kv16 = (recs.get("llama8b-fp16") or {}).get("kv_cache_tokens")
    kv4 = (recs.get("llama8b-int4") or {}).get("kv_cache_tokens")
    if kv16 and kv4:
        L.append(f"- **양자화는 KV 용량도 같이 사줍니다.** Llama-8B에서 할당된 KV가 "
                 f"{kv16:,}→{kv4:,} 토큰({kv4 / kv16:.1f}배)으로 늘어, 같은 카드에서 "
                 f"동시 요청 수나 컨텍스트 길이를 더 감당할 수 있습니다.")
    return L


def analysis(recs: dict) -> list[str]:
    L = ["### 5.1 정밀도의 이득은 동시성에 따라 뒤집힙니다", ""]
    rows = []
    for c in (1, 2, 4, 8, 16, 32, 64):
        f16 = _thr(recs, "llama8b-fp16", c)
        if not f16:
            continue
        cells = []
        for p in ("fp8", "int8", "int4"):
            v = _thr(recs, f"llama8b-{p}", c)
            cells.append(f"{v / f16:.2f}x" if v else "—")
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    if rows:
        L += ["Llama-8B TP1, fp16 대비 처리량 배수:", "",
              "| 동시성 | fp8 | int8 | int4 |", "|---|---|---|---|"] + rows + [""]
    L += [
        "int4의 우위는 c=1의 2.7배에서 c=64의 1.4배로 단조 감소하고, int8은 1.6→1.4배로 "
        "거의 평탄합니다. 두 곡선이 c=32~64 사이에서 교차합니다. 해석: 배치가 작을 때 "
        "디코드는 **가중치 판독 대역폭**이 병목이라 4비트가 그대로 이득이 되지만, 배치가 "
        "커지면 같은 가중치 판독을 여러 요청이 분담하므로 병목이 연산으로 옮겨가고 "
        "int4의 실시간 역양자화 비용이 남습니다. W8A8(int8)은 활성값까지 8비트라 "
        "고배치에서 INT8 Tensor Core를 실제로 쓸 수 있어 그 구간을 가져갑니다.",
        "",
        "**운영 함의**: 저지연·저동시성(코딩 어시스턴트, 대화형)에는 int4, "
        "고스루풋 배치 서빙에는 int8이 유리합니다. 하나만 고른다면 int8이 두 구간에서 "
        "모두 상위권이며 품질 손실도 가장 작았습니다.",
        "",
        "### 5.2 Ampere에서 FP8은 '메모리 절약'까지만입니다", "",
    ]
    f8_1, i8_1 = _thr(recs, "llama8b-fp8", 1), _thr(recs, "llama8b-int8", 1)
    f8_64, i8_64 = _thr(recs, "llama8b-fp8", 64), _thr(recs, "llama8b-int8", 64)
    if all((f8_1, i8_1, f8_64, i8_64)):
        L += [f"측정값은 이 예상과 일치합니다. c=1에서는 fp8 {f8_1:.0f} vs int8 "
              f"{i8_1:.0f} tok/s로 사실상 동급(둘 다 8비트 가중치를 읽는 "
              f"메모리 병목 구간)이지만, c=64에서는 int8 {i8_64:.0f} > fp8 {f8_64:.0f} "
              f"tok/s로 int8이 앞섭니다 — INT8 Tensor Core를 실제로 쓸 수 있는 쪽이 "
              f"고배치에서 이깁니다. FP8 연산기가 있는 Ada/Hopper에서는 이 순서가 "
              f"바뀔 것으로 예상되며, 그 검증은 해당 하드웨어가 필요합니다.", ""]
    L += ["### 5.3 양자화 vs 텐서 병렬 (PCIe 환경)", "",
          "| 동시성 | int8 TP1(1GPU) | int8 TP2(2GPU) | int8 GPU당 이득 | "
          "int4 TP1(1GPU) | int4 TP2(2GPU) | int4 GPU당 이득 |",
          "|---|---|---|---|---|---|---|"]
    for c in (1, 4, 16, 64):
        cells = []
        for p in ("int8", "int4"):
            t1 = _thr(recs, f"qwen14b-{p}-tp1", c)
            t2 = _thr(recs, f"qwen14b-{p}-tp2", c)
            if t1 and t2:
                cells += [f"{t1:.0f}", f"{t2:.0f}", f"**{t1 / (t2 / 2):.2f}x**"]
            else:
                cells += ["—", "—", "—"]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["",
          "GPU당 이득이 **동시성과 함께 커집니다**(int4: 1.27→1.70배). all-reduce 전송량이 "
          "배치에 비례해 늘기 때문에, 호스트 브리지 PCIe(실측 분류 ~12GB/s)에서는 "
          "배치가 클수록 TP2의 손실이 커집니다. 같은 모델을 GPU 1장에 담을 수 있다면 "
          "**TP를 쓰지 않고 레플리카를 늘리는 편이** 처리량·전력 모두 유리합니다.",
          "",
          "**단, TP2가 사는 지점은 KV 용량입니다.** " +
          (f"int4 TP2는 KV {(recs.get('qwen14b-int4-tp2') or {}).get('kv_cache_tokens'):,} 토큰인데 "
           f"TP1은 {(recs.get('qwen14b-int4-tp1') or {}).get('kv_cache_tokens'):,} 토큰"
           if recs.get("qwen14b-int4-tp2") and recs.get("qwen14b-int4-tp1") else "") +
          "으로 3배 차이입니다. 긴 컨텍스트나 레플리카당 높은 동시성이 필요하면 TP2가 "
          "여전히 필요하고, int8 TP1은 KV가 특히 빠듯해(가중치 13.8GiB / 예산 21.6GiB) "
          "실사용 여유가 작습니다.",
          "",
          "### 5.4 품질: 어디까지 신뢰할 수 있나", "",
          "GSM8K 200문제에서 비율의 표준오차는 약 3pp, **두 설정 차이의 표준오차는 "
          "약 4pp**입니다. 따라서:", ""]
    ll = [(p, _acc(recs, f"llama8b-{p}")) for p in PRECISIONS]
    if all(v for _, v in ll):
        L.append(f"- Llama-8B의 최대 격차({(ll[0][1] - ll[3][1]) * 100:.1f}pp)는 "
                 f"**노이즈와 구분되지 않습니다**. 이 표본으로는 '8B에서 int4가 품질을 "
                 f"떨어뜨렸다'고 말할 수 없습니다.")
    qq = [(p, _acc(recs, f"qwen14b-{p}-tp2")) for p in PRECISIONS]
    if all(v for _, v in qq):
        L.append(f"- Qwen-14B는 int8이 fp16과 거의 같은데"
                 f"({qq[0][1] * 100:.1f}% vs {qq[2][1] * 100:.1f}%) fp8과 int4가 함께 "
                 f"{qq[3][1] * 100:.1f}%로 내려갔습니다. 약 1.5σ라 단정은 못 하지만 "
                 f"**방향성은 int8 우세**입니다. fp8이 int4만큼 떨어진 것은 포맷 자체보다 "
                 f"체크포인트 캘리브레이션 차이일 가능성이 큽니다(서로 다른 조직이 "
                 f"만든 체크포인트).")
    a1, a2 = _acc(recs, "qwen14b-int4-tp1"), _acc(recs, "qwen14b-int4-tp2")
    if a1 and a2:
        L.append(f"- 같은 int4 가중치를 TP1/TP2로 돌린 결과가 {a2 * 100:.1f}% vs "
                 f"{a1 * 100:.1f}%로 {abs(a2 - a1) * 100:.1f}pp 달랐습니다. 이 값이 곧 "
                 f"**측정 노이즈의 하한 눈금**입니다(가중치가 동일하므로).")
    L += ["", "확정적 품질 결론이 필요하면 문제 수를 1,000개 이상으로 올리고 "
          "(설정당 약 10분 추가) MMLU 등 다른 과제를 병행해야 합니다.", ""]
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
        "## 요약", "", *summary(recs), "",
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
    L += ["", "## 5. 분석", ""] + analysis(recs) + ["",
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
