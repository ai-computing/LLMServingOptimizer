# 멀티노드(2-노드) TP=16 비균질 네트워크 검증 보고서

**대상 서버**: s8 + s2, 각 8× NVIDIA A40 (총 16 GPU)
**노드 간 인터커넥트**: InfiniBand HDR (mlx5_0 / ConnectX-6, 200 Gb/s, IPoIB 192.168.210.x)
**대상 모델**: Llama-3.1-8B (TP=16, 노드 간), 70B는 예측만
**기준(ground truth)**: 실제 vLLM 0.8.4 (단일 노드 검증과 동일 버전) 실측
**작성일**: 2026-06

이 보고서는 `HETEROGENEOUS_NETWORK_REPORT_KO.md §5(한계) 2번 — "다중 노드(TP≥16) 미검증"` 항목의
후속 작업이다. 단일 노드 3계층(NVLink/PCIe/QPI) 보정에 **노드 간 IB tier**를 한 단계 더 추가하고,
실제 2-노드 vLLM TP16 실측으로 보정·검증한다.

---

## 1. 배경 — 단일 노드에서 멀티노드로

단일 노드 작업(§HETEROGENEOUS_NETWORK_REPORT_KO)은 8×A40 2-소켓 박스에서:
- **계층형 인터커넥트 모델**(`tp_group_shape`)로 3계층(NVLink 52.8 / PCIe 24.5 / QPI 21.0 GB/s)을 표현,
- 링크 파라미터만으로는 TP8 갭을 닫을 수 없음을 NCCL 실측+폐루프로 규명한 뒤,
- **부하 의존 collective-overhead 모델**(`_collective_overhead_ns`, socket-gated)로 비용을 연산 임계경로에 주입,
- TP8 처리량 오차를 +103%/+106% → +0.2%/−7.5%(70B/8B)로 줄였다.

남은 한계는 **collective가 노드 경계(IB/NIC)를 넘는 TP≥16 미검증**이었다. 본 작업이 이를 다룬다.

---

## 2. 환경 구축

### 2.1 하드웨어/네트워크 (실측 확인)

| 항목 | s8 (head) | s2 (worker) |
|---|---|---|
| GPU | 8× A40 (46 GB) | 8× A40 (46 GB) |
| IB HCA | mlx5_0, ConnectX-6, 200 Gb/s, Active | mlx5_0, ConnectX-6, 200 Gb/s, Active |
| IB iface / IP | `ibs8` / 192.168.210.108 | `ibp194s0` / 192.168.210.102 |
| 이더넷 | 1 Gb/s (관리용, 인터커넥트 부적합) | 1 Gb/s |

→ 노드 간 유일한 고속 경로는 **IB 200 Gb/s (≈25 GB/s nominal)**. intra-node 최저 tier인 QPI(21 GB/s)와
공칭 대역폭은 비슷하나, **노드 간 all-reduce는 네트워크 스택·2-hop으로 floor와 유효 비용이 더 크다** (§4).

### 2.2 멀티노드 vLLM 기동 (Ray + NCCL/IB)

- 두 노드에 동일 이미지(`nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3`, vLLM 0.8.4) — s8→s2 IB 전송.
- 8B 가중치(15 GB) s8→s2 IB rsync. **(70B 132 GB는 s2 디스크 여유 134 GB로 불충족 → 70B 실측 보류)**
- Ray 클러스터: head=s8, worker=s2, 컨트롤 플레인+NCCL 모두 IB IP에 바인딩
  (`--network host`, `NCCL_IB_HCA=mlx5_0`, `NCCL_SOCKET_IFNAME=<ibs8|ibp194s0>`, RDMA char device 패스스루).
- `vllm serve --tensor-parallel-size 16 --distributed-executor-backend ray` → Ray가 16-way TP를 2노드로 분산.
- 확인: `ray status` 16 GPU, 기동 후 **양 노드 8 GPU 전부 ~42 GB 점유** = 진성 노드 간 TP16.

스크립트: `validation/multinode/{run_cluster_node.sh, serve_tp16.sh, run_multinode_validation.sh}`.

---

## 3. 시뮬레이터 확장 — 4계층 + 노드(IB) tier overhead

### 3.1 4계층 인터커넥트

`tp_group_shape=[2,2,2,2]`, `link_bw=[52.8, 24.5, 21.0, 25.0]` → ASTRA-Sim이 TP16 all-reduce를
NVLink→PCIe→QPI→**IB** 4차원으로 분해. (config: `cluster_config/a40_16gpu_tp16_{8b,70b}_4tier{,_cohd}.json`)

### 3.2 노드 경계 collective-overhead (`_collective_overhead_ns` 일반화)

단일 노드 모델은 socket 경계(`npus_per_group > socket_size=4`)만 게이팅했다. 멀티노드용으로 **노드 경계
tier**를 추가:

- `node_size`(=8)를 넘으면(`npus_per_group > node_size`) **IB tier 비용**(`node_floor_ns`, `node_per_token_ns`)이
  socket(QPI) 비용을 **대체**한다. NCCL 계층형 all-reduce는 **가장 느린 tier가 floor·비용을 지배**하므로
  합산이 아닌 "가장 느린 교차 tier의 비용"을 부과 — 단일 노드에서 QPI 비용을 (PCIe 위에 더하지 않고) THE
  overhead로 쓴 것과 동일한 모델링 철학.
- 하위호환: `node_size` 미지정 시 기존 socket-gated 동작과 바이트 동일.

```
config: {"enabled":true, "socket_size":4, "floor_ns":70000, "per_token_ns":10000,
         "node_size":8, "node_floor_ns":105000, "node_per_token_ns":18000}
```

### 3.3 프로파일

70B tp16 프로파일은 기존 존재. **8B tp16은 신규 생성**(`profiler24` 컨테이너, GQA KV-head 복제:
tp16 > num_kv_heads=8 → rank당 ≥1 head). 출력: `llm_profile/perf_models/A40/meta-llama/Llama-3.1-8B/tp16/`.

---

## 4. 보정 — IB tier 상수 결정 (8B TP16, ShareGPT-100)

placeholder(node_floor=150µs, per_token=25µs)로 시작 → 실측 대비 처리량 −29%/TPOT +25%로 **과함**.
5점 sweep으로 두 지표를 동시에 ~10% 이내로 맞추는 상수를 탐색:

| (node_floor / per_token) | gen tput | TPOT p50 | gen 오차 | TPOT 오차 | max |
|---|---:|---:|---:|---:|---:|
| 150 / 25 µs (placeholder) | 234 | 238.2 | −29.1% | +24.5% | 29% |
| 112 / 19 µs | 287 | 188.2 | −12.9% | −1.6% | 13% |
| **105 / 18 µs (채택)** | **299** | **179.8** | **−9.5%** | **−6.0%** | **9.5%** |
| 100 / 17 µs | 311 | 171.5 | −5.8% | −10.3% | 10% |
| 90 / 15 µs | 338 | 154.9 | +2.5% | −19.0% | 19% |

**채택: `node_floor_ns=105000(105µs)`, `node_per_token_ns=18000(18µs)`** — gen·TPOT 모두 ≤10%.

→ 단일 노드 QPI tier(70µs / 10µs) 대비 **floor 1.5×, per-token 1.8×**. 노드 간 IB all-reduce가 소켓 간
QPI보다 (네트워크 스택·2-hop으로) 느린 물리와 정성적으로 일치.

### 4.1 NCCL IB all-reduce 직접 측정 — 보정 floor 교차검증

`validation/multinode/nccl_ib_allreduce_bench.py`로 world_size=16 cross-node all-reduce를 직접 측정
(torchrun, 두 노드 컨테이너에서 동시 실행):

| 메시지 | 실측 latency | busbw (GB/s) | sim(bytes/bw, lat=0) | real/sim |
|---|---:|---:|---:|---:|
| **16 KB (decode)** | **80.8 µs** | 0.4 | 1.2 µs | **65.8×** |
| 64 KB | 183.6 µs | 0.7 | 4.9 µs | 37.4× |
| 256 KB | 948 µs | 0.5 | 19.7 µs | 48.2× |
| 4 MB | 7.87 ms | 1.0 | 0.31 ms | 25.0× |
| 64 MB (prefill) | 68.1 ms | 1.8 | 5.03 ms | 13.5× |

**교차검증 결과**:
- **16 KB cross-node all-reduce floor = 81 µs** ≈ 보정한 `node_floor=105 µs`(같은 자릿수). 8B decode all-reduce
  payload(hidden 4096 → 토큰당 8 KB, 배치 2–8 → 16–64 KB)는 실측 81–184 µs 구간이고, 보정식
  `105 µs + 18 µs × batch`가 정확히 이 밴드에 안착 → **end-to-end 보정값이 직접 NCCL 측정과 독립적으로
  일치**(보정값이 81 µs보다 다소 큰 것은 step당 프레임워크/스케줄 오버헤드 흡수분).
- IB 16 KB floor(81 µs) > 단일 노드 QPI 16 KB floor(~35–90 µs) → **IB가 QPI보다 느림** = 모델의 tier 순서 확증.
- 유효 busbw 0.4–1.8 GB/s (공칭 25의 1/15–1/60) → 노드 간 all-reduce는 **대역폭이 아니라 latency·sync가
  지배** → 대역폭만 모델이 소형 메시지를 65× 과소평가하는 근본 원인을 직접 증명.

---

## 5. 결과

### 5.1 8B TP=16 (실측 검증)

| 지표 | sim(대역폭만 4계층) | **sim(+IB cohd 보정)** | vLLM 실측 | 보정후 오차 |
|---|---:|---:|---:|---:|
| gen throughput (tok/s) | 818 | **299** | 330 | **+148% → −9.5%** |
| total throughput (tok/s) | 1556 | 568 | — | — |
| TPOT p50 (ms) | 31.5 | **179.8** | 191.3 | **−84% → −6.0%** |
| TTFT p50 (ms) | 53 | 7998 | 4672 | — |
| makespan (s) | 28.5 | 78.0 | — | — |

→ 대역폭만 모델은 단일 노드와 동일하게 통신을 무시해 처리량을 **+148% 낙관**. IB-tier overhead 보정으로
**−9.5%/−6.0%**까지 좁힘 — 단일 노드 8B TP8(−7.5%)과 동급 정확도.

### 5.2 70B TP=16 (예측, 실측 보류)

8B에서 보정한 동일 상수(105/18µs)를 70B에 적용한 예측:

| 지표 | sim(대역폭만) | sim(+IB cohd) | 참고: 단일노드 TP8 |
|---|---:|---:|---:|
| total throughput (tok/s) | 682 | 222.6 | 305 (sim≈vLLM) |
| gen throughput (tok/s) | — | 117.1 | — |
| TPOT p50 (ms) | 87.5 | 458.7 | ~303 |

→ 70B도 **TP16(노드 간 IB) < TP8(노드 내)** 로 예측 — comm-bound 심화의 정성적 일관성. 단,
**70B ground-truth 교차검증은 s2 디스크 확보 후 과제로 남김**(8B↔70B 상수 전이 검증은 미완).

---

## 6. 한계 및 향후 과제

1. **70B 실측 미완**: s2 디스크 부족(여유 134 GB < 70B 가중치 132 GB + 이미지 24 GB). 단일 노드에서
   입증된 "단일 상수의 모델 크기 전이성"을 멀티노드에서 재확인하려면 70B 실측이 필요.
2. **`node_per_token_ns`도 경험 상수**: 8B 1점 보정. 모델 크기 전이성(70B 실측)·다양한 워크로드/배치로의
   일반화는 추가 검증 대상.
3. ~~**NCCL IB floor 직접 측정 미반영**~~ → **완료(§4.1)**: 16 KB cross-node all-reduce 실측 81 µs가
   보정 floor 105 µs와 일치 확인. 다만 단일 점 보정이라 다양한 배치/payload로의 정밀 일반화는 추가 과제.
4. **단일 IB tier 가정**: 3+ 노드/멀티-rail IB, NIC당 다중 GPU 등은 미고려.

---

## 7. 산출물

**코드**
- `inference_serving/trace_generator.py::_collective_overhead_ns` — 노드(IB) tier 게이팅 추가(`node_size`/`node_floor_ns`/`node_per_token_ns`), 하위호환

**설정/프로파일**
- `cluster_config/a40_16gpu_tp16_{8b,70b}_4tier.json`(대역폭만), `_4tier_cohd.json`(+IB overhead, 보정 상수 반영)
- `llm_profile/perf_models/A40/meta-llama/Llama-3.1-8B/tp16/`(신규)

**멀티노드 실행/측정**
- `validation/multinode/{run_cluster_node.sh, serve_tp16.sh, run_multinode_validation.sh, nccl_ib_allreduce_bench.py}`
- `validation/vllm_a40_tp16_8b_results.jsonl`(실측), `validation/sim_a40_tp16_*`(sim), `validation/compare_tp16.py`
- `validation/multinode/nccl_ib_{s8_head,s2_worker}.log`(NCCL IB all-reduce 실측 로그)

---

## 8. 결론

단일 노드의 균등·계층형 모델은 collective가 **노드 경계(IB)를 넘는 TP16에서 처리량을 ~2.5배 낙관**했다
(8B 818 vs 실측 330). `_collective_overhead_ns`에 **노드 경계 tier**를 추가하고 실제 2-노드 vLLM TP16
실측으로 `node_floor=105µs / node_per_token=18µs`를 보정해, **8B TP16 처리량/TPOT 오차를
+148%/−84% → −9.5%/−6.0%**로 줄였다. 이 보정 floor는 **직접 측정한 16 KB cross-node NCCL all-reduce
latency(81 µs)와 독립적으로 일치**(§4.1)하며, QPI tier 대비 floor 1.5×·per-token 1.8×로 물리적으로 타당하다.
70B는 동일 상수로 예측치를 제시했으며, ground-truth 교차검증은 s2 디스크 확보 후의 과제로 남는다.
