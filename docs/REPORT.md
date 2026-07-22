# LLMServingSim 버전 비교: fork(현재 코드베이스) vs upstream 최신 (casys-kaist)

날짜: 2026-07-20

## 비교 대상

| | fork (이 코드베이스) | upstream 최신 |
|---|---|---|
| 커밋 | `3e22070` (planner-milp-maxflow) | `c84e58b` (casys-kaist/main, v1.1.0+) |
| 코드 구조 | `main.py` + `inference_serving/` | `python -m serving` + `serving/core/` |
| 프로파일 | `llm_profile/perf_models/` (구 포맷, PyTorch Profiler) | `profiler/perf/` (신 포맷, vLLM layerwise_profile) |
| 설치 위치 | `/home/swsok/LLMServingSim` | `/home/swsok/LLMServingSim-upstream` (+ `.venv`) |

## 선택한 시뮬레이션 구성

이 코드베이스의 대표 구성(README 예제, `output/sharegpt_req100_rate10.csv`)과 최대한 일치시킴:

- 모델: `meta-llama/Llama-3.1-8B`, 단일 노드 / 단일 인스턴스 / TP=1
- 워크로드: `dataset/sharegpt_req100_rate10_llama.jsonl` (100 requests, ~10 req/s) — **양쪽 동일 파일**
- 플래그 통일: block-size 16, prefix caching OFF, chunked prefill OFF, RR 라우팅, max-num-batched-tokens 2048
- **불가피한 차이 (핵심 주의사항)**:
  - **하드웨어 프로파일**: fork = A6000 (구 포맷), upstream = **RTXPRO6000 Blackwell** (신 포맷은 RTXPRO6000만 배포됨; 구 포맷 프로파일은 신버전이 읽지 못함 → 동일 하드웨어 비교 불가능)
  - dtype variant: fork = fp16, upstream = bf16 (배포된 유일한 variant; 메모리 회계 동일)
  - 프로파일링 방법 자체가 다름 (PyTorch Profiler vs vLLM layerwise)

## 실행 검증

- 100개 요청 전부 arrival time 일치, 요청별 디코드 스텝 수(ITL 개수) 100/100 일치 → **동일 워크로드가 동일하게 스케줄링됨**
- CSV `output` 칼럼 semantics 차이: fork = input+output 합, upstream = 순수 output (분석에서 보정)

## 결과

### 요약 (100 requests)

| 메트릭 | fork fresh (A6000) | fork 5/6 기존 (A6000) | upstream (RTXPRO6000) | up/fresh 비율 |
|---|---|---|---|---|
| TTFT mean (ms) | 128.02 | 94.84 | **28.23** | 0.221× |
| TTFT p50 (ms) | 105.60 | 77.93 | 25.25 | |
| TTFT p99 (ms) | 353.68 | 284.52 | 75.32 | |
| TPOT mean (ms) | 42.26 | 26.53 | **13.51** | 0.320× |
| TPOT p99 (ms) | 111.73 | 81.94 | 14.85 | |
| E2E latency mean (s) | 8.83 | 5.77 | **3.15** | 0.357× |
| Queuing delay mean (ms) | 34.66 | 23.58 | 7.55 | 0.218× |
| Request throughput (req/s) | 3.54 | – | 6.18 | 1.75× |
| Total token throughput (tok/s) | 1571.8 | – | 2742.2 | 1.74× |

- per-request TTFT 비율(up/fork): mean 0.293, stdev 0.147 (0.065–0.737)
- per-request TPOT 비율(up/fork): mean 0.337, stdev 0.067 (0.119–0.440)

### 해석

1. **절대값 차이의 주 원인은 하드웨어**: RTXPRO6000(96GB, 1597GB/s, Blackwell)이 A6000(768GB/s 설정)보다 대략 2–3× 빠른 것은 합리적 범위. 시뮬레이터 버전 차이만의 효과로 해석하면 안 됨.
2. **분포 형태 차이는 시뮬레이터 개선을 반영**:
   - upstream의 TPOT 분포가 극히 타이트함 (p50 13.70 / p99 14.85ms, 변동계수 ~3%). fork는 p99가 mean의 2.6× (111.73ms). 신버전의 vLLM 기반 프로파일 + skew-attention 모델 + 외삽(extrapolation) 룩업이 배치 크기 변화에 따른 latency를 더 매끄럽게 모델링.
   - per-request TPOT 비율의 낮은 분산(stdev 0.067)은 두 시뮬레이터가 구조적으로 유사하게 스케줄링함을 시사 — 스텝 수 100% 일치와 부합.
3. **fork 내부 드리프트**: fork 최신 코드(fresh)가 5/6 결과보다 TTFT +35%, TPOT +59% 높음 — 5월 이후 fork 자체 커밋들(collective-overhead 모델, scheduler/trace_generator 수정)의 효과. 과거 결과와 비교할 때 이 점 유의.

## 재현 방법

```bash
# fork (이 코드베이스)
python3 main.py --cluster-config cluster_config/single_node_single_instance.json \
  --fp 16 --block-size 16 --dataset dataset/sharegpt_req100_rate10_llama.jsonl \
  --output output/compare_v0_a6000_tp1_fresh.csv --num-req 100 --log-interval 1.0

# upstream (별도 클론)
cd /home/swsok/LLMServingSim-upstream && .venv/bin/python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype bfloat16 --block-size 16 \
  --dataset workloads/sharegpt_req100_rate10_llama.jsonl \
  --output outputs/compare_v1_rtxpro_tp1.csv --num-reqs 100 --log-interval 1.0 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --request-routing-policy RR
```

## 산출물

- fork fresh 결과: `output/compare_v0_a6000_tp1_fresh.csv` (+ `.log`)
- upstream 결과: `/home/swsok/LLMServingSim-upstream/outputs/compare_v1_rtxpro_tp1.csv` (+ `.log`)
- fork 기존(5/6) 결과: `output/sharegpt_req100_rate10.csv`

## upstream 환경 셋업 기록

- 클론: `git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git /home/swsok/LLMServingSim-upstream`
- venv: `.venv` (rich, numpy, pandas, pyyaml, pyinstrument, **msgspec**, chakra, **protobuf>=7.35** — chakra가 protobuf==6.* 을 핀하지만 배포 gencode가 7.35라 업그레이드 필요)
- ASTRA-Sim: `bash astra-sim/build/astra_analytical/build.sh` 후 `build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra` → `bin/AstraSim_Analytical_Congestion_Unaware` 심링크 필요 (fork와 동일한 퀴크)
