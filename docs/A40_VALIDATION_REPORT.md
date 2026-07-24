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

## 개선 적용: 계층형 토폴로지 + 부하 의존 collective-overhead 포팅 (2026-07-24)

legacy fork의 보정(`validation/HETEROGENEOUS_NETWORK_REPORT_KO.md`)을 upstream에 포팅했다.
두 부분: (1) `tp_group_shape`로 TP 그룹을 ASTRA-Sim 다차원 위계로 분해(A40 3-tier
`[2,2,2]`, `link_bw=[52.8,24.5,21.0]` = NVLink/PCIe/QPI 실측 busbw), (2) o_proj·down_proj
op latency에 `floor_ns + per_token_ns × n_decode`를 더하는 부하 의존 collective-overhead
(opt-in, 기본 OFF, 미설정 시 트레이스 **바이트 동일** — 회귀 없음 확인).

구현: `serving/core/config_builder.py`(tp_group_shape → `_compute_network_dims`,
per-tier `link_bw`; `collective_overhead`를 cluster에 전달),
`serving/core/trace_generator.py`(`_collective_overhead_ns`를 `_emit_layer`의 TP
all-reduce 레이어에 적용), `serving/__main__.py`(cluster-global 설정 주입).
(upstream 서브모듈 브랜치 `feat/hetero-collective-overhead`.)

### 상수 캘리브레이션 (이 A40 / vLLM 0.19 / ShareGPT-100)

직접 NCCL all-reduce 실측(`validation/nccl_allreduce_bench.py`)으로 tier별 지연 floor를
확인: 16 KB(디코드) all-reduce floor = **NVLink 70.7µs / PCIe 52µs / QPI 71.5µs**, 유효
busbw NVLink ~40 / PCIe ~8.8 / QPI ~1 GB/s. QPI floor 71.5µs·per-token≈8µs는 legacy의
70µs/10µs를 독립적으로 재확인. 이후 실측 대비 5점 스윕(TPOT·throughput 기준)으로 확정:

| tier(TP) | tp_group_shape | link_bw (GB/s) | collective_overhead (floor / per_token) |
|---|---|---|---|
| TP2 (NVLink) | — | 52.8 | 미적용 |
| TP4 (PCIe) | [2,2] | [52.8, 24.5] | 52µs / 3µs |
| TP8 (QPI) | [2,2,2] | [52.8, 24.5, 21.0] | 140µs / 20µs |

### 결과 (Diff% = (sim − vLLM)/vLLM, Mean)

| 지표 | | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|---|
| **TPOT** | 개선 전 | -0.1% | -9.1% | **-48.2%** | **-93.8%** |
| | **개선 후** | -0.1% | **-7.1%** | **+5.3%** | **-8.6%** |
| **Latency** | 개선 전 | -0.1% | -8.8% | **-46.3%** | **-90.8%** |
| | **개선 후** | -0.1% | -6.9% | **+6.8%** | +21.5% |

### 정성적 개선 — TP4↔TP8 순서 역전 해소

| Latency Mean (s) | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| vLLM 실측 | 9.34 | 5.08 | 4.75 | **15.76** |
| 개선 전 (단일 link_bw) | 9.32 | 4.64 | 2.55 | **1.45** ✗ |
| **개선 후** | 9.32 | 4.73 | 5.07 | **19.14** ✓ |

개선 전 시뮬은 TP가 커질수록 계속 빨라진다고 오판했으나(TP8 1.45s < TP4 2.55s), 개선 후
**TP8 > TP4의 negative scaling을 실제와 일치하게 재현**한다(TPOT도 TP8 90.4ms ≫ TP4 22.3ms,
실측 99.0 ≫ 21.2와 일치).

- **TP4가 −46% → +7%로 교정**: 계층형 per-tier link_bw(PCIe 병목 반영) + 소폭 overhead.
- **TP8 TPOT −94% → −8.6%**: collective-overhead가 소켓 간 QPI all-reduce의 latency floor를
  임계경로에 주입. TPOT/throughput 기준(legacy와 동일 기준)에서 ±10% 이내.
- **TTFT는 여전히 과소평가**(TP8 −85%): sim은 연산 완료 기준, 실측은 client 수신(큐잉 포함)
  기준의 정의차 — legacy와 동일하게 TPOT·throughput·순서를 비교 기준으로 삼는다. TP8 Latency
  +21.5% 초과도 이 TTFT 정의차에서 파생.

### 전송 경로 검증 — 실측 NVLink 사용 + 시뮬 대역폭 반영 (2026-07-24)

모델링 전제(NVLink/PCIe/소켓간 위계)가 실제와 일치하는지 양쪽에서 직접 확인했다.

**(A) 실측 vLLM은 NVLink를 사용했다** (`NCCL_DEBUG=INFO` + `nvidia-smi nvlink` 카운터,
`validation/nvlink_probe.py`). NCCL 토폴로지 탐지가 GPU 페어 링크를 `NVL`(48), 페어 간을
`PHB`(PCIe, 24), 소켓 간을 `SYS`(16)로 인식:

| TP | GPU 세트 | NCCL 채널 transport | NVLink |
|---|---|---|---|
| 2 | 0,1 | 8/8 `P2P/CUMEM` (NVL 링크) | 100% NVLink |
| 4 | 0–3 | 16 `P2P/CUMEM` + 16 `SHM/direct` | 페어 NVLink + 브리지 SHM |
| 8 | 0–7 | 8 `P2P/CUMEM` + 8 `SHM/direct` | 페어 NVLink + 소켓간 SHM |

world=2에서 200×32MB all-reduce 동안 GPU0 NVLink Data Tx가 **+6,400 MiB** 증가(실트래픽
확인). NVSwitch가 없어 TP4/8은 **NVLink 페어 + 나머지 SHM(호스트 경유)** 의 혼합 전송 —
§ negative scaling의 물리적 원인이자 tier 위계의 근거. (NCCL이 IB 플러그인은 로드하나
단일 노드라 미사용.)

**(B) 시뮬은 이 tier별 버스 속도를 반영한다 — 단 디코드는 latency-bound**. TP8 시뮬이 생성한
ASTRA-Sim 입력은 `npus_count=[2,2,2]`, `bandwidth=[52.8,24.5,21.0]`,
`all-reduce-implementation=["ring","ring","ring"]`(3차원). overhead를 끈 채
tiered `[52.8,24.5,21.0]` vs uniform `[52.8,52.8,52.8]`로 돌리면 출력이 달라져(→ 대역폭이
실제 사용됨) 있으나 디코드 TPOT은 **6.1ms vs 5.8ms(~5%)** 로 거의 불변이다. 실측 TPOT은
99ms — 즉 **대역폭 항은 프리필(큰 메시지)에만 유효하고 디코드 collective 비용은 잡지 못한다**.
디코드 all-reduce는 작은 메시지라 `메시지/대역폭`이 수 µs에 불과하고 고정 latency floor가
지배하기 때문(§3.3의 legacy 결론과 동일). 디코드 갭 6→90ms는 전적으로 `collective_overhead`가
채운다. 이것이 보정이 **계층형 대역폭 + 부하 의존 overhead** 두 부분으로 구성된 이유다.

## 개선 여지 (다음 단계 후보)

- ~~계층적 링크 모델 + fork의 load-dependent collective-overhead 포팅~~ → **완료(위 "개선 적용" 절)**.
- **TTFT 모델링**: 현재 sim은 큐잉을 포함한 client-side TTFT를 과소평가(TP8 −85%). TP8 Latency
  +21.5% 초과도 여기서 파생 — decode 백프레셔로 인한 큐 대기 반영이 다음 과제.
- **`per_token_ns`의 모델 크기 스케일**(legacy §5.7): 8B에서 캘리브레이션한 상수를 70B 등 다른
  hidden_size로 옮길 때 per-token은 payload에 비례해 스케일해야 함(floor는 하드웨어 상수로 전이).
- 멀티노드(TP16, InfiniBand tier) 확장: `tp_group_shape`에 노드 tier 추가 + IB 상수 캘리브레이션.

## 산출물

- 프로파일: `profiles/upstream/A40/meta-llama/Llama-3.1-8B/bf16/{meta.yaml, tp{1,2,4,8}/}`
- 클러스터 구성: `cluster_config/upstream/a40_tp{1,2,4,8}_validation.json`
- 시뮬: `output/a40_validation/sim_tp{1,2,4,8}.{csv,log}`
- 실측: `output/a40_validation/bench_tp{1,2,4,8}/` (meta.json, requests.jsonl, timeseries.csv)
- 비교: `output/a40_validation/bench_tp{1,2,4,8}/validation/` (summary.txt + throughput/requests/latency PNG)
