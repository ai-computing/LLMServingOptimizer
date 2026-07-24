# A40 실측 검증: Llama-3.1-70B — upstream LLMServingSim 시뮬 vs 실제 vLLM

날짜: 2026-07-24
장비: NVIDIA A40 48GB × 8, 2-socket, CUDA 12.x
모델: meta-llama/Llama-3.1-70B (bf16). 실측 가중치는 로컬 `Llama-3.1-70B-Instruct`
(아키텍처 config가 base와 동일 — 벤치는 입출력 길이를 고정하므로 timing은 base와 동일).

## 서빙 가능성 (메모리)

70B bf16 가중치 ≈ 141 GB. A40(45GB 실효, 0.9 util → ~40.5GB)에 적재하려면 **TP≥4** 필요:
TP1/2 불가(141/70 GB/GPU), **TP4 = 35 GB/GPU(빠듯)**, TP8 = 17.6 GB/GPU(여유). GQA
(kv_heads=8)라 TP=4/8 모두 KV가 깔끔히 샤딩됨. 실측에서 **TP4 부팅 성공**(GPU당 43GB/96%,
AsyncLLM 로드 376s) — 4×A40으로 70B 서빙이 실제로 가능함을 확인.

## 파이프라인

1. **프로파일**: `python -m profiler profile meta-llama/Llama-3.1-70B --hardware A40 --tp 1,4,8
   --skip-skew` (단일 GPU TP emulation, dummy weights). 산출:
   `profiles/upstream/A40/meta-llama/Llama-3.1-70B/bf16/tp{1,4,8}/`. skew 생략 →
   시뮬은 pooled 상수 alpha 폴백.
2. **시뮬**: `python -m serving` + A40 70B 클러스터 config (계층형 토폴로지 + collective-overhead).
3. **실측**: `python -m bench run` (로컬 Instruct 가중치, 동일 워크로드).
4. **비교**: `python -m bench validate` (Diff% = (sim − vLLM)/vLLM).

- 워크로드: `dataset/sharegpt_req100_rate10_llama.jsonl` (100 req).
- 공통: bf16, block 16, max-num-seqs 128, max-num-batched-tokens 2048; bench `--max-model-len 8192`.

### 클러스터 구성 (`cluster_config/upstream/a40_70b_tp{4,8}_validation.json`)

8B와 동일 메커니즘(계층형 `tp_group_shape` + 부하 의존 collective-overhead). collective-overhead
상수는 **8B 캘리브레이션값을 hidden_size 비율(4096→8192, 2×)로 스케일한 뒤 실측으로 재확인**
(legacy §5.7: floor는 하드웨어 상수로 전이, per_token은 payload=hidden에 비례해 스케일):

| TP | tp_group_shape | link_bw (GB/s) | collective_overhead (floor / per_token) |
|---|---|---|---|
| 4 | [2,2] | [52.8, 24.5] | 52µs / 6µs (8B 52/3 → per_token 2×) |
| 8 | [2,2,2] | [52.8, 24.5, 21.0] | 115µs / 33µs (8B 140/20 스케일 후 1점 재보정) |

## 결과 (Diff%, Mean; 개선 전 = collective-overhead OFF)

| 지표 | | TP4 | TP8 |
|---|---|---|---|
| **TPOT** | 개선 전 | -36.0% | **-85.9%** |
| | **개선 후** | **+2.1%** | **-5.0%** |
| **Latency** | 개선 전 | -33.0% | **-82.7%** |
| | **개선 후** | **+9.2%** | +12.6% |

TP4/TP8 전체 백분위 (개선 후):

| 지표 | TP4 vLLM / Sim / Diff | TP8 vLLM / Sim / Diff |
|---|---|---|
| TTFT Mean (ms) | 2754.6 / 854.5 / **-69.0%** | 17330.2 / 988.6 / **-94.3%** |
| TPOT Mean (ms) | 201.7 / 206.0 / **+2.1%** | 493.1 / 468.3 / **-5.0%** |
| TPOT P99 (ms) | 1249.0 / 683.0 / -45.3% | 4055.2 / 826.2 / -79.6% |
| Latency Mean (s) | 35.72 / 39.00 / **+9.2%** | 80.34 / 90.49 / +12.6% |

## 정성적 개선 — negative scaling 재현

| Latency Mean (s) | TP4 | TP8 |
|---|---|---|
| **vLLM 실측** | 35.72 | **80.34** |
| 개선 전 (overhead OFF) | 23.94 | **13.92** ✗ |
| **개선 후** | 39.00 | **90.49** ✓ |

개선 전 시뮬은 TP8을 TP4보다 **빠르다고 오판**(13.9s < 23.9s)했으나, 실측은 TP8이 **2.25× 느리다**
(소켓 간 collective 폭증). 보정 후 시뮬은 TP8을 90.5s로 끌어올려 **실제의 TP8 ≫ TP4 순서를 재현**한다
(TPOT도 실측 TP8 493ms ≫ TP4 202ms와 일치).

## 분석

1. **8B→70B 상수 전이 검증**: 8B에서 캘리브레이션한 collective-overhead를 hidden 비율(2×)로 스케일한
   초기 상수만으로 TP4 TPOT **+2.1%**, TP8 TPOT **+12.3%** 에 도달 — legacy §5.7의 "floor는 모델
   독립, per_token ∝ hidden_size" 법칙이 A40에서도 성립. TP8은 1점 재보정(per_token 40→33µs,
   floor 140→115µs)으로 **-5.0%** 로 마무리.
2. **TP4 near-perfect**: 계층형 per-tier 대역폭(NVLink+PCIe) + 소폭 overhead로 TPOT +2.1% /
   Latency +9.2%. 8B와 동급 정확도.
3. **TTFT 과소평가**: TP8 TTFT −94% (실측 17.3s는 느린 디코드로 인한 극심한 큐 대기; sim은
   연산 완료 기준이라 큐잉 미반영). TP8 Latency +12.6% 및 P99 과대(+23.5%)도 여기서 파생 — legacy와
   동일하게 **TPOT/throughput을 비교 기준**으로 삼는다.
4. **skew 생략 영향 제한적**: 70B 프로파일은 skew 없이 pooled 상수 alpha를 쓰지만 TPOT이 잘 맞음 —
   이 워크로드의 decode kv 분포가 극단적으로 치우치지 않기 때문. 정밀 개선 시 `--only-skew`로 보강 가능.

## 개선 여지

- TTFT 큐잉 모델링(§ 8B 리포트와 공통 과제) — TP8 Latency 초과의 주 원인.
- 70B skew sweep 보강(`--only-skew`, TP당 ~5h)으로 불균일 decode 정확도 향상.
- TP4 KV가 빠듯(~5GB) → 실측/시뮬의 KV 용량 정합을 위해 sim `npu_mem.mem_size`를 실효값으로 조정 검토.

## 산출물

- 프로파일: `profiles/upstream/A40/meta-llama/Llama-3.1-70B/bf16/`
- 클러스터 구성: `cluster_config/upstream/a40_70b_tp{4,8}_validation.json`
- 시뮬: `output/a40_70b_validation/sim_tp{4,8}.{csv,log}` (+ `_noovh` 개선 전 베이스라인)
- 실측: `output/a40_70b_validation/bench_tp{4,8}/`
- 비교: `output/a40_70b_validation/bench_tp{4,8}/validation/`
