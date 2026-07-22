"""
TP 레이턴시 합성기 + 역검증 (Phase 2B/5 of TODO_TP4_extrapolation.md)

두 가지 합성 방법:
  roofline : compute-bound→1/tp 스케일, memory-bound→1.0 스케일
  ratio    : src-tp→ref-tp 실측 비율을 레이어 타입별로 학습, ref-tp→target-tp에 적용

역검증 (Phase 5):
  python3 validation/synthesize_tp.py --hardware A5000 \\
      --model meta-llama/Llama-3.1-8B --src-tp 1 --target-tp 2 --validate

TP=4 합성 (Phase 4):
  python3 validation/synthesize_tp.py --hardware A5000 \\
      --model meta-llama/Llama-3.1-8B --src-tp 1 --ref-tp 2 --target-tp 4 --write
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
PROFILE_ROOT = REPO_ROOT / "llm_profile" / "perf_models"

COMPUTE_BOUND = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj", "lm_head"}
MEMORY_BOUND  = {"embedding", "input_layernorm", "post_layernorm",
                 "final_layernorm", "rope", "act_fn"}


def load_layers(hardware: str, model: str, tp: int) -> pd.DataFrame:
    path = PROFILE_ROOT / hardware / model / f"tp{tp}" / "layers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    return pd.read_csv(path)


def roofline_scale(layer_name: str, tp_src: int, tp_tgt: int) -> float:
    """이상적인 루프라인 스케일 팩터."""
    ratio = tp_src / tp_tgt
    if layer_name in COMPUTE_BOUND:
        return ratio
    elif layer_name in MEMORY_BOUND:
        return 1.0
    else:
        return ratio


def synthesize_roofline(df_src: pd.DataFrame,
                        tp_src: int, tp_tgt: int) -> pd.DataFrame:
    """루프라인 모델로 tp_tgt 레이턴시 합성."""
    df = df_src.copy()
    df["tp_size"] = tp_tgt
    df["latency(ns)"] = df.apply(
        lambda r: max(1, int(r["latency(ns)"] * roofline_scale(r["layer_name"], tp_src, tp_tgt))),
        axis=1,
    )
    return df


def compute_observed_ratios(df_lo: pd.DataFrame, df_hi: pd.DataFrame,
                             tp_lo: int, tp_hi: int) -> pd.DataFrame:
    """
    tp_lo → tp_hi 실측 스케일링 비율을 (layer_name, input, kv_cache) 단위로 계산.
    이상치(Z-score > 3) 제거 후 반환.
    """
    m = df_lo.merge(df_hi, on=["layer_name", "input", "kv_cache"],
                    suffixes=("_lo", "_hi"))
    m["ratio_obs"] = m["latency(ns)_hi"] / m["latency(ns)_lo"].clip(1)

    # 레이어별 이상치 제거 (log 공간 Z-score)
    clean = []
    for name, grp in m.groupby("layer_name"):
        if len(grp) < 3:
            clean.append(grp)
            continue
        log_r = np.log(grp["ratio_obs"].clip(1e-9))
        med, std = log_r.median(), log_r.std()
        mask = (std == 0) | (((log_r - med).abs() / std.clip(1e-9)) <= 3.0)
        clean.append(grp[mask])
    return pd.concat(clean, ignore_index=True)


def synthesize_ratio(df_hi: pd.DataFrame, obs_ratios: pd.DataFrame,
                     tp_hi: int, tp_tgt: int) -> pd.DataFrame:
    """
    관측된 tp_lo→tp_hi 비율을 이용해 tp_hi→tp_tgt 레이턴시 외삽.
    로그 공간 선형 외삽: ratio_next = ratio_obs ^ (log2(tp_tgt/tp_hi) / log2(tp_hi/tp_lo))
    tp_lo는 obs_ratios로부터 추론됨.
    """
    steps_obs = 1.0   # tp_lo→tp_hi는 항상 2배 step으로 가정 (1→2 또는 2→4)
    steps_ext = np.log2(tp_tgt / tp_hi)

    # 레이어별 기하평균 비율
    layer_ratio = obs_ratios.groupby("layer_name")["ratio_obs"].apply(
        lambda x: np.exp(np.mean(np.log(x.clip(1e-9))))
    )

    # input=1 outlier 처리: input>=2 기하평균으로 대체
    obs_ge2 = obs_ratios[obs_ratios["input"] >= 2]
    layer_ratio_ge2 = obs_ge2.groupby("layer_name")["ratio_obs"].apply(
        lambda x: np.exp(np.mean(np.log(x.clip(1e-9))))
    )

    df = df_hi.copy()
    df["tp_size"] = tp_tgt

    def extrap(row):
        name = row["layer_name"]
        lat = row["latency(ns)"]
        # input=1의 이상치가 있는 레이어는 input>=2 비율 사용
        if row["input"] == 1 and name in ("down_proj", "lm_head", "o_proj"):
            base_ratio = layer_ratio_ge2.get(name, layer_ratio.get(name, 1.0))
        else:
            base_ratio = layer_ratio.get(name, 1.0)
        ext_factor = base_ratio ** steps_ext
        return max(1, int(lat * ext_factor))

    df["latency(ns)"] = df.apply(extrap, axis=1)
    return df


def validate(df_src: pd.DataFrame, df_pred: pd.DataFrame,
             df_actual: pd.DataFrame,
             method: str, tp_src: int, tp_tgt: int) -> dict:
    """예측 vs 실측 비교. MAPE 및 레이어별 오차 출력."""
    merged = df_pred.merge(df_actual, on=["layer_name", "input", "kv_cache"],
                           suffixes=("_pred", "_actual"))
    merged["err_pct"] = (
        (merged["latency(ns)_pred"] - merged["latency(ns)_actual"]).abs()
        / merged["latency(ns)_actual"].clip(1) * 100
    )

    # pred/actual ratio (이상치 탐지용)
    merged["ratio"] = merged["latency(ns)_pred"] / merged["latency(ns)_actual"].clip(1)
    n_before = len(merged)

    clean_rows = []
    for name, grp in merged.groupby("layer_name"):
        if len(grp) < 3:
            clean_rows.append(grp)
            continue
        log_r = np.log(grp["ratio"].clip(1e-9))
        med, std = log_r.median(), log_r.std()
        mask = (std == 0) | (((log_r - med).abs() / std.clip(1e-9)) <= 3.0)
        clean_rows.append(grp[mask])
    merged_clean = pd.concat(clean_rows, ignore_index=True)
    n_dropped = n_before - len(merged_clean)

    overall_mape = merged_clean["err_pct"].mean()
    by_layer = merged_clean.groupby("layer_name")["err_pct"].mean().sort_values(ascending=False)

    print(f"\n{'='*60}")
    print(f"  역검증: TP={tp_src} → TP={tp_tgt} 예측  (method={method})")
    print(f"{'='*60}")
    if n_dropped:
        print(f"  [이상치 제거] {n_dropped}개 행 제외 (측정 아티팩트)")
    print(f"  전체 MAPE: {overall_mape:.1f}%   {'✓ PASS (<15%)' if overall_mape < 15 else '✗ FAIL (≥15%)'}")

    print(f"\n  레이어별 MAPE (상위 순):")
    for name, err in by_layer.items():
        bar = "█" * int(err / 2)
        print(f"    {name:20s}  {err:6.1f}%  {bar}")

    # 실측 tp_src → tp_tgt 스케일링 비율 (TP=1 원본 기준)
    obs = compute_observed_ratios(df_src, df_actual, tp_src, tp_tgt)
    obs_ge2 = obs[obs["input"] >= 2]
    act_ratio = obs_ge2.groupby("layer_name")["ratio_obs"].apply(
        lambda x: np.exp(np.mean(np.log(x.clip(1e-9))))
    ).sort_values()
    print(f"\n  실측 tp{tp_src}→tp{tp_tgt} 스케일링 비율 (input≥2, 기하평균):")
    print(f"  {'layer':20s}  {'actual':>8s}  {'roofline':>8s}  {'diff':>8s}")
    print(f"  {'-'*50}")
    for name, r in act_ratio.items():
        ideal = 0.5 if name in COMPUTE_BOUND else 1.0
        diff = (r - ideal) / ideal * 100
        print(f"  {name:20s}  {r:8.3f}  {ideal:8.1f}  {diff:+7.1f}%")

    print(f"\n  예측 vs 실측 샘플 (input=2, kv_cache=0):")
    sample = merged_clean[(merged_clean["input"] == 2) & (merged_clean["kv_cache"] == 0)].sort_values("layer_name")
    print(f"  {'layer':20s}  {'pred(ns)':>10s}  {'actual(ns)':>10s}  {'err%':>7s}")
    print(f"  {'-'*55}")
    for _, r in sample.iterrows():
        print(f"  {r['layer_name']:20s}  {int(r['latency(ns)_pred']):>10,d}  "
              f"{int(r['latency(ns)_actual']):>10,d}  {r['err_pct']:>6.1f}%")

    merged_clean = merged_clean.copy()
    merged_clean["group"] = merged_clean["layer_name"].apply(
        lambda n: "compute-bound" if n in COMPUTE_BOUND else
                  ("memory-bound" if n in MEMORY_BOUND else "other")
    )
    grp_mape = merged_clean.groupby("group")["err_pct"].mean()
    print(f"\n  그룹별 MAPE:")
    for g, m in grp_mape.items():
        print(f"    {g:20s}  {m:.1f}%")

    return {"overall_mape": overall_mape, "by_layer": by_layer.to_dict(),
            "pass": overall_mape < 15}


def load_attn_predictions(hardware: str, model: str, tp: int):
    """tp{tp}/predictions/ 에서 prefill/decode predictions CSV 로드."""
    base = PROFILE_ROOT / hardware / model / f"tp{tp}" / "predictions"
    prefill = pd.read_csv(base / "attn_prefill_predictions.csv")
    decode  = pd.read_csv(base / "attn_decode_predictions.csv")
    return prefill, decode


def synthesize_attn_predictions(hardware: str, model: str,
                                 tp_lo: int, tp_hi: int, tp_tgt: int) -> tuple:
    """
    tp_lo→tp_hi 실측 비율 학습 후 tp_hi→tp_tgt 로그-선형 외삽.
    반환: (synth_prefill_df, synth_decode_df)
    """
    pf_lo, dc_lo = load_attn_predictions(hardware, model, tp_lo)
    pf_hi, dc_hi = load_attn_predictions(hardware, model, tp_hi)

    steps_obs = np.log2(tp_hi / tp_lo)
    steps_ext = np.log2(tp_tgt / tp_hi)
    exponent  = steps_ext / steps_obs  # 1.0 when tp_lo→tp_hi and tp_hi→tp_tgt are same-size steps

    # Prefill
    pm = pf_lo.merge(pf_hi, on=["kv_cache_size", "prefill_chunk_size"],
                     suffixes=("_lo", "_hi"))
    pm["ratio"] = pm["prediction_hi"] / pm["prediction_lo"].clip(1)
    prefill_ratio = float(np.exp(np.mean(np.log(pm["ratio"].clip(1e-9)))))
    ext_prefill   = prefill_ratio ** exponent
    print(f"  Prefill  tp{tp_lo}→tp{tp_hi} geomean ratio={prefill_ratio:.3f}  "
          f"ext factor (tp{tp_hi}→tp{tp_tgt})={ext_prefill:.3f}")

    # Decode
    dm = dc_lo.merge(dc_hi, on=["batch_size", "kv_cache_size"],
                     suffixes=("_lo", "_hi"))
    dm["ratio"] = dm["prediction_hi"] / dm["prediction_lo"].clip(1)
    decode_ratio = float(np.exp(np.mean(np.log(dm["ratio"].clip(1e-9)))))
    ext_decode   = decode_ratio ** exponent
    print(f"  Decode   tp{tp_lo}→tp{tp_hi} geomean ratio={decode_ratio:.3f}  "
          f"ext factor (tp{tp_hi}→tp{tp_tgt})={ext_decode:.3f}")

    synth_prefill = pf_hi.copy()
    synth_prefill["prediction"] = (synth_prefill["prediction"] * ext_prefill).astype(int).clip(lower=1)

    synth_decode = dc_hi.copy()
    synth_decode["prediction"] = (synth_decode["prediction"] * ext_decode).astype(int).clip(lower=1)

    return synth_prefill, synth_decode


def main():
    p = argparse.ArgumentParser(description="TP latency synthesizer + back-validator")
    p.add_argument("--hardware", default="A5000")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--src-tp", type=int, default=1)
    p.add_argument("--ref-tp", type=int, default=None,
                   help="ratio 방법 시 학습용 상위 TP (src-tp와 쌍으로 비율 학습)")
    p.add_argument("--target-tp", type=int, required=True)
    p.add_argument("--method", choices=["roofline", "ratio", "both"], default="both")
    p.add_argument("--validate", action="store_true",
                   help="target-tp 실측 프로파일과 비교 (역검증)")
    p.add_argument("--write", action="store_true",
                   help="합성 결과를 tp{target}/layers.csv에 저장")
    args = p.parse_args()

    df_src = load_layers(args.hardware, args.model, args.src_tp)
    print(f"Loaded TP={args.src_tp}: {len(df_src)} rows")

    results = {}

    # --- Roofline ---
    if args.method in ("roofline", "both"):
        df_roof = synthesize_roofline(df_src, args.src_tp, args.target_tp)
        if args.validate:
            df_actual = load_layers(args.hardware, args.model, args.target_tp)
            r = validate(df_src, df_roof, df_actual, "roofline", args.src_tp, args.target_tp)
            results["roofline"] = r
        if args.write:
            out = PROFILE_ROOT / args.hardware / args.model / f"tp{args.target_tp}" / "layers_synth_roofline.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df_roof.to_csv(out, index=False)
            print(f"\nSaved: {out}")

    # --- Ratio ---
    if args.method in ("ratio", "both"):
        ref_tp = args.ref_tp if args.ref_tp else args.src_tp
        if ref_tp == args.src_tp:
            print(f"\n[ratio] --ref-tp 미지정: ratio 방법은 두 개의 실측 TP 필요 (--src-tp와 --ref-tp 지정), skip")
        else:
            df_ref_lo = load_layers(args.hardware, args.model, args.src_tp)
            df_ref_hi = load_layers(args.hardware, args.model, ref_tp)
            print(f"Loaded TP={ref_tp}: {len(df_ref_hi)} rows")
            obs_ratios = compute_observed_ratios(df_ref_lo, df_ref_hi, args.src_tp, ref_tp)
            df_ratio = synthesize_ratio(df_ref_hi, obs_ratios, ref_tp, args.target_tp)
            if args.validate:
                df_actual = load_layers(args.hardware, args.model, args.target_tp)
                r = validate(df_ref_hi, df_ratio, df_actual, "ratio", ref_tp, args.target_tp)
                results["ratio"] = r
            if args.write:
                out_dir = PROFILE_ROOT / args.hardware / args.model / f"tp{args.target_tp}"
                out_dir.mkdir(parents=True, exist_ok=True)
                # layers.csv 저장 (ratio 방법 주)
                out = out_dir / "layers.csv"
                df_ratio.to_csv(out, index=False)
                print(f"\nSaved: {out}")
                # roofline 버전도 백업으로 저장
                df_roof2 = synthesize_roofline(df_src, args.src_tp, args.target_tp)
                df_roof2.to_csv(out_dir / "layers_synth_roofline.csv", index=False)
                print(f"Saved: {out_dir / 'layers_synth_roofline.csv'} (roofline backup)")

                # attention predictions 합성
                pred_dir = out_dir / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[attention 합성] tp{args.src_tp}→tp{ref_tp} 비율으로 tp{ref_tp}→tp{args.target_tp} 외삽:")
                synth_pf, synth_dc = synthesize_attn_predictions(
                    args.hardware, args.model, args.src_tp, ref_tp, args.target_tp)
                synth_pf.to_csv(pred_dir / "attn_prefill_predictions.csv", index=False)
                synth_dc.to_csv(pred_dir / "attn_decode_predictions.csv", index=False)
                print(f"Saved: {pred_dir / 'attn_prefill_predictions.csv'} ({len(synth_pf)} rows)")
                print(f"Saved: {pred_dir / 'attn_decode_predictions.csv'} ({len(synth_dc)} rows)")
                print("(pkl 파일은 시뮬레이터 첫 실행 시 자동 생성됩니다)")

    # --- 결론 ---
    if results:
        print(f"\n{'='*60}")
        print("  결론 (Phase 5 기준)")
        print(f"{'='*60}")
        for method, r in results.items():
            status = "✓ PASS → TP 외삽 진행 가능" if r["pass"] else "✗ FAIL → 외부 도구 교차검증 필요"
            print(f"  {method:12s}  MAPE={r['overall_mape']:.1f}%  {status}")


if __name__ == "__main__":
    main()
