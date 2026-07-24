# A40 실측 검증: upstream LLMServingSim 시뮬레이션 vs 실제 vLLM

날짜: 2026-07-24
장비: NVIDIA A40 48GB × 8, CUDA 12.x
모델: meta-llama/Llama-3.1-8B (bf16) — 실측 가중치는 ungated 미러 `NousResearch/Meta-Llama-3.1-8B` (아키텍처 config 동일)

## GPU 인터커넥트 토폴로지 (결과 해석의 핵심)

`nvidia-smi topo -m` 기준 — A40에는 NVSwitch가 없고 **2-GPU 단위 NVLink 페어**만 존재:

| 페어 | 연결 |
|---|---|
| GPU 0–1, 2–3, 4–5, 6–7 | **NV4** (NVLink, 페어 내부) |
| 같은 NUMA 노드 내 페어 간 (0–3, 4–7) | NODE (PCIe, CPU 브리지) |
| 노드 간 (0–3 ↔ 4–7) | SYS (소켓 간) |

즉 TP가 커질수록 collective가 NVLink→PCIe→소켓간으로 **계층적으로 느려지는** 하드웨어다.
이 점이 A5000(NVLink 전무, 순수 PCIe)과 결정적으로 다르다.

## 파이프라인

upstream(casys-kaist v1.1.0+, `c84e58b`) 공식 검증 플로우:

1. **프로파일링**: `python -m profiler profile meta-llama/Llama-3.1-8B --hardware A40 --tp 1,2,4,8` (vLLM 0.19.0, layerwise_profile + skew sweep). 산출: `profiles/upstream/A40/meta-llama/Llama-3.1-8B/bf16/{meta.yaml, tp{1,2,4,8}/}`
2. **시뮬레이션**: `python -m serving` + A40 클러스터 구성 (아래)
3. **실측**: `python -m bench run` — 동일 워크로드를 실제 vLLM AsyncLLM으로 재생 (입출력 길이 고정)
4. **비교**: `python -m bench validate` (Diff% = (sim − vLLM) / vLLM × 100)

- 워크로드: `dataset/sharegpt_req100_rate10_llama.jsonl` (100 req, ~10 req/s)
- 공통 설정: bf16, block 16, max-num-seqs 128, max-num-batched-tokens 2048, chunked prefill/prefix caching ON (vLLM v1 기본), bench는 `--max-model-len 16384`
- 프로파일: 단일 GPU에서 TP emulation (hf_overrides 샤딩), TP=1,2,4,8 모두 커버. skew sweep TP당 ~4.5h, 총 ~19.5h.

### 클러스터 구성 (`cluster_config/upstream/a40_tp{1,2,4,8}_validation.json`)

npu_mem 48GB / 696GB/s, cpu_mem 128GB/256GB/s 공통. `link_bw`는 **각 TP가 실제로 쓰는 GPU 세트의 지배적 인터커넥트**를 반영:

| TP | bench GPU 세트 | 실제 인터커넥트 | 구성 link_bw |
|---|---|---|---|
| 1 | GPU 0 | (collective 없음) | 112 (미사용) |
| 2 | GPU 0,1 | NVLink 페어 | 112 GB/s |
| 4 | GPU 0–3 | NVLink 페어 2개 + PCIe NODE 브리지 | 32 GB/s |
| 8 | GPU 0–7 | + 소켓 간 SYS | 32 GB/s |

## 결과

### TP=1 (단일 GPU) — 매우 정확

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 112.5 | 90.4 | **-19.6%** |
| TTFT P95 (ms) | 201.8 | 206.4 | +2.2% |
| TPOT Mean (ms) | 42.0 | 41.9 | **-0.1%** |
| TPOT P99 (ms) | 54.0 | 60.9 | +12.9% |
| Latency Mean (s) | 9.34 | 9.32 | **-0.1%** |
| Latency P99 (s) | 23.77 | 23.88 | +0.4% |

### TP=2 (NVLink 페어) — 양호, A5000 대비 개선

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 65.2 | 43.6 | **-33.1%** |
| TPOT Mean (ms) | 22.1 | 20.1 | **-9.1%** |
| Latency Mean (s) | 5.08 | 4.64 | **-8.8%** |

### TP=4 (NVLink 페어 + PCIe 브리지) — 시뮬 크게 낙관적

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 73.6 | 26.4 | **-64.1%** |
| TPOT Mean (ms) | 21.2 | 11.0 | **-48.2%** |
| Latency Mean (s) | 4.75 | 2.55 | **-46.3%** |

### TP=8 (소켓 간 SYS) — 시뮬 붕괴 (추세 자체가 반대)

| 메트릭 | vLLM 실측 | 시뮬 | Diff% |
|---|---|---|---|
| TTFT Mean (ms) | 870.9 | 17.4 | **-98.0%** |
| TPOT Mean (ms) | 99.0 | 6.2 | **-93.8%** |
| Latency Mean (s) | 15.76 | 1.45 | **-90.8%** |

### 스케일링 추세 — 이번 검증의 핵심

| Latency Mean (s) | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| **vLLM 실측** | 9.34 | 5.08 | 4.75 | **15.76** |
| **시뮬** | 9.32 | 4.64 | 2.55 | **1.45** |

| TPOT Mean (ms) | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| **vLLM 실측** | 42.0 | 22.1 | 21.2 | **99.0** |
| **시뮬** | 41.9 | 20.1 | 11.0 | **6.2** |

실측 하드웨어는 **TP=2에서 이득, TP=4는 정체, TP=8에서 급격히 악화(negative scaling)** — 8-GPU
allreduce가 소켓 간 PCIe/SYS를 타면서 collective 비용이 폭증하기 때문. 반면 시뮬은 단일
`link_bw` 기반 이상적 스케일링을 가정해 **TP가 커질수록 계속 빨라진다고 예측** — TP=8에서는
실측과 추세가 정반대다.

## 분석

1. **TP=1 정확도 우수**: TPOT/E2E latency 오차 -0.1%, TTFT tail ±2%. 신형 vLLM layerwise
   프로파일 + skew-attention 모델이 단일 GPU 커널 시간을 거의 완벽히 잡는다. (TTFT mean만 -20%
   낮은데, prefill 스케줄링/큐잉 오버헤드가 프로파일에 없어서로 추정 — tail은 잘 맞음.)

2. **오차의 단일 원인은 collective 비용 과소평가이며, TP와 함께 단조 증가**:
   - TP=2 (NVLink) −9% → TP=4 (+PCIe 브리지) −48% → TP=8 (+소켓간) −91%.
   - 프로파일러의 TP emulation은 **단일 GPU에서 per-GPU 커널 시간만** 측정하고 NCCL allreduce는
     ASTRA-Sim analytical 모델 + `link_bw`가 담당한다. A40처럼 계층적(NVLink/PCIe/소켓간) 토폴로지는
     단일 `link_bw` 스칼라로 표현할 수 없어, TP가 커질수록 실제 병목(느린 링크)을 못 잡는다.

3. **A5000 대비 TP=2 개선**: A5000 TP2는 −53%/−32% (TTFT/TPOT)였으나 A40 TP2는 −33%/−9%.
   A40의 TP=2가 **NVLink 페어**라 실제 collective가 훨씬 싸고, 그만큼 시뮬의 이상적 가정과
   가까워졌기 때문. 즉 이번 결과는 "인터커넥트가 좋을수록 시뮬이 잘 맞는다"는 것을 직접 보여준다.

4. **실측 TP=8 negative scaling**: A40 8장은 NVSwitch가 없어 8-way allreduce가 소켓 간 PCIe를
   경유 → TTFT 871ms, TPOT 99ms로 TP=4(74ms/21ms)보다 크게 나빠진다. 대규모 TP를 이 하드웨어에서
   쓰면 안 된다는 실측 근거이자, 시뮬이 이를 경고하지 못한다는 한계다.

## 개선 여지 (다음 단계 후보)

- 클러스터 구성에 **계층적 링크 모델** 도입: NVLink 페어 vs PCIe 브리지 vs 소켓간을 구분하는
  per-dimension `link_bw`(upstream v1.1.0의 multi-dim topology 활용) 또는 실측 NCCL allreduce
  대역폭/지연으로 캘리브레이션.
- fork의 load-dependent collective-overhead 모델(commit `49556c6`)을 upstream에 포팅 검토.
- TP=4/8 `link_bw`를 실측 NCCL allreduce로 역산해 재검증.

## 산출물

- 프로파일: `profiles/upstream/A40/meta-llama/Llama-3.1-8B/bf16/{meta.yaml, tp{1,2,4,8}/}`
- 클러스터 구성: `cluster_config/upstream/a40_tp{1,2,4,8}_validation.json`
- 시뮬: `output/a40_validation/sim_tp{1,2,4,8}.{csv,log}`
- 실측: `output/a40_validation/bench_tp{1,2,4,8}/` (meta.json, requests.jsonl, timeseries.csv)
- 비교: `output/a40_validation/bench_tp{1,2,4,8}/validation/` (summary.txt + throughput/requests/latency PNG)
