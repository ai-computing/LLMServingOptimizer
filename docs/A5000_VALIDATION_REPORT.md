# A5000 실측 검증: upstream LLMServingSim 시뮬레이션 vs 실제 vLLM

날짜: 2026-07-21
장비: RTX A5000 24GB × 2 (GPU간 연결: PCIe **NODE** — NVLink 없음, CPU 브리지 경유), CUDA 12.8

## 파이프라인

upstream(casys-kaist v1.1.0+, `c84e58b`)의 공식 검증 플로우를 그대로 사용:

1. **프로파일링**: `python -m profiler profile meta-llama/Llama-3.1-8B --hardware A5000 --tp 1,2` (vLLM 0.19.0, layerwise_profile)
2. **시뮬레이션**: `python -m serving` + A5000 클러스터 구성 (24GB/768GB/s/link_bw 32GB/s)
3. **실측**: `python -m bench run` — 동일 워크로드를 실제 vLLM AsyncLLM으로 재생 (출력 길이 고정 재현)
4. **비교**: `python -m bench validate`

- 워크로드: `sharegpt_req100_rate10_llama.jsonl` (100 req, ~10 req/s, 최대 시퀀스 1,293 tok)
- 공통 설정: bf16, block 16, max-num-seqs 128, max-num-batched-tokens 2048, chunked prefill/prefix caching ON (vLLM v1 기본)
- bench TP1은 `--max-model-len 16384` 필요 (24GB 단일 GPU에서 기본 131072가 KV 16GiB를 요구해 기동 실패)

## 프로파일링 통계 (총 13시간 36분)

| TP | dense | per_seq | attention | skew | 소요 |
|---|---|---|---|---|---|
| 1 | 152샷/33초 | 40샷/23초 | 8,643샷/57분 | 13,009샷/5h52m + alpha 피팅 | ~6h50m |
| 2 | 152샷/30초 | 40샷/22초 | 8,643샷/57분 | 13,009샷/5h48m + alpha 피팅 | ~6h46m |

산출: `profiler/perf/A5000/meta-llama/Llama-3.1-8B/bf16/{meta.yaml, tp1/, tp2/}` (skew_fit.csv 포함)

## 결과

### TP=1 (단일 GPU) — 잘 맞음

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 120.3 | 97.9 | **-18.6%** |
| TTFT P90 (ms) | 202.4 | 203.1 | **+0.3%** |
| TPOT Mean (ms) | 39.5 | 36.4 | **-7.8%** |
| TPOT P99 (ms) | 78.4 | 70.9 | -9.6% |
| Latency Mean (s) | 8.38 | 7.90 | **-5.7%** |
| Latency P99 (s) | 20.14 | 19.61 | -2.6% |

### TP=2 (PCIe 2-GPU) — 시뮬레이터가 크게 낙관적

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 99.2 | 46.3 | **-53.3%** |
| TPOT Mean (ms) | 26.2 | 17.9 | **-31.6%** |
| Latency Mean (s) | 5.77 | 4.13 | **-28.5%** |

## 분석

1. **TP1 정확도는 우수**: E2E latency -6%, TPOT -8%, TTFT 꼬리(p90/p95)는 ±5% 이내. 신형 vLLM 기반 프로파일 + skew-attention 모델이 단일 GPU 커널 시간을 잘 잡는다.
2. **TP2 오차의 원인은 collective 비용 과소평가**:
   - 프로파일러의 TP emulation은 **단일 GPU에서 hf_overrides 샤딩**으로 per-GPU 커널 시간만 측정 — 실제 NCCL allreduce 비용은 프로파일에 없음.
   - allreduce는 ASTRA-Sim analytical 모델 + 클러스터 구성(link_bw 32GB/s, latency 0)이 담당하는데, 이 장비의 GPU간 연결은 NVLink 없는 **PCIe NODE(CPU 브리지 경유)** 라 실제 비용이 훨씬 큼.
   - vLLM 실측 TPOT: TP1 39.5ms → TP2 26.2ms (1.51× 개선). 시뮬: 36.4 → 17.9ms (2.03× 개선, 거의 이상적 스케일링) — 차이가 곧 collective 오버헤드.
3. **fork의 기존 검증과 비교** (동일 장비, 구버전 시뮬 + 구 프로파일러, 300req 워크로드라 직접 비교는 참고용):

| | fork 구버전 (TTFT/TPOT MAPE) | upstream 신버전 (TTFT/TPOT mean diff) |
|---|---|---|
| TP1 | 65.1% / 15.9% | **18.6% / 7.8%** |
| TP2 | 59.3% / 42.1% | **53.3% / 31.6%** |

   → **신버전이 TP1에서 극적으로, TP2에서도 유의미하게 개선**. 그러나 TP>1 collective 과소평가는 구버전·신버전 공통의 구조적 문제 (fork가 collective-overhead 캘리브레이션 모델을 자체 개발했던 것과 정확히 같은 원인).

## 개선 여지 (다음 단계 후보)

- 클러스터 구성의 `link_bw`/`link_latency`를 실측 기반으로 캘리브레이션 (PCIe NODE 실효 대역폭 ~10-15GB/s + NCCL 오버헤드).
- upstream v1.1.0의 per-dimension link 설정 활용.
- fork의 load-dependent collective-overhead 모델(commit `49556c6`)을 upstream에 포팅하는 것도 검토 가치 있음.

## 산출물

- 프로파일: `/home/swsok/LLMServingSim-upstream/profiler/perf/A5000/…` (+ `profiler/a5000_profile.log`)
- 시뮬: `outputs/a5000_validation/sim_tp{1,2}.{csv,log}`
- 실측: `bench/results/a5000_tp{1,2}/` (meta.json, requests.jsonl, timeseries.csv)
- 비교: `bench/results/a5000_tp{1,2}/validation/` (summary.txt + throughput/requests/latency PNG)
