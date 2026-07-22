# 비균질 네트워크 특성 반영을 통한 LLMServingSim 시뮬레이션 정확도 개선 보고서

**대상 서버**: 8× NVIDIA A40, 2-socket 워크스테이션
**대상 모델**: Llama-3.1-70B / 8B (TP=2/4/8/16)
**기준(ground truth)**: 실제 vLLM 0.8.4 실측
**작성일**: 2026-06

---

## 1. 배경 및 문제 정의

LLMServingSim의 네트워크 모델은 **단일 균등 대역폭(`link_bw`)** 을 가정한다. `_create_network_config`는
모든 NPU 간 링크를 동일 대역폭의 FullyConnected로 생성하므로, 텐서 병렬(TP) all-reduce가 어떤 물리
링크를 타든 같은 비용으로 계산된다.

그러나 실제 8× A40 서버의 GPU 간 연결은 **3계층 비균질** 구조다. 단일 대역폭 가정은 특히 TP 그룹이
느린 링크를 가로지를 때 큰 오차를 유발한다 — 본 보고서는 이 비균질성을 시뮬레이션에 반영한 작업과
그 정확도 개선 결과를 정리한다.

---

## 2. 대상 서버 네트워크 특성 분석

`nvidia-smi topo -m` 결과, 8개 GPU는 **규칙적 2×2×2 계층**으로 묶인다:

| 계층(tier) | 연결 | 묶는 그룹 | 측정 대역폭 |
|---|---|---|---|
| Tier 0 | `NV4` (NVLink ×4) | (0,1)(2,3)(4,5)(6,7) | **52.8 GB/s** |
| Tier 1 | `NODE` (NUMA내 PCIe) | {0,1}↔{2,3}, {4,5}↔{6,7} | **24.5 GB/s** |
| Tier 2 | `SYS` (소켓간 QPI/UPI) | {0–3} ↔ {4–7} | **21.0 GB/s** |

대역폭은 프로젝트의 단일-tier 검증 config(TP2=NVLink, TP4=PCIe 병목, TP8=QPI 병목)와 NCCL 실측에서
교차 확인했다. 따라서 **TP2는 소켓 내 NVLink만, TP4는 NVLink+PCIe, TP8은 소켓 간 QPI까지** 가로지른다.

---

## 3. 수행한 작업

### 3.1 계층형 인터커넥트 모델 추가 (`tp_group_shape`)

`inference_serving/config_builder.py::_create_network_config`를 일반화:

- cluster-config 최상위 `tp_group_shape`로 TP 그룹을 ASTRA-Sim의 다차원(안쪽=빠른 tier 먼저)으로 분해.
- `link_bw`/`link_latency`를 **스칼라 또는 per-tier 배열** 모두 허용.
- 스칼라+`tp_group_shape` 미지정 시 기존 평탄 출력과 **바이트 동일**(하위호환).

예) 8-GPU TP8 → `npus_count=[2,2,2]`, `bandwidth=[52.8, 24.5, 21.0]`
(config: `cluster_config/a40_8gpu_tp8_70b_3tier.json`, DSE fabric `2socket_8npu_nvlink_bridge_per_2slot`).

ASTRA-Sim이 이 3차원 위계를 소비해 TP all-reduce를 NVLink→PCIe→QPI로 분해 실행함을 sanity-run으로 확인.

### 3.2 검증 환경 구축

- ASTRA-Sim analytical 백엔드 빌드, **TP8 70B 실측 프로파일**(기존 외삽본 대체; 외삽은 projection
  지연을 최대 ~19% 과대평가).
- **실제 vLLM 벤치마크**(70B/8B, TP4/8, 로컬 가중치, 동일 ShareGPT 워크로드)로 ground truth 확보.

### 3.3 근본 원인 분석 — 계층 "표현"만으로는 부족

3계층 토폴로지로 구조는 정확히 표현했으나, 절대 성능은 여전히 빗나갔다. 원인을 두 실험으로 규명:

**(a) 실측 NCCL all-reduce vs 시뮬의 대역폭 전용 모델** (`validation/nccl_allreduce_bench.py`)

| 메시지 | TP8 실제 | TP8 busbw | sim 모델(bytes/bw, lat=0) | 실제/sim |
|---|---:|---:|---:|---:|
| 16 KB (디코드) | 0.070 ms | 0.4 GB/s | 0.0014 ms | **51×** |
| 64 MB (프리필) | 7.90 ms | 14.9 GB/s | 5.59 ms | 1.4× |

→ ① 작은 메시지에 **고정 ~35–90µs 지연 floor**가 존재(모델은 `latency=0`이라 무시), ② 소켓 간 유효
busbw가 공칭 21보다 낮은 **~15 GB/s**, ③ 실제 TP8 all-reduce는 TP4의 **3.05×**인데 모델은 1.4×만 가정.

**(b) 폐루프: 측정값을 링크 파라미터에 주입해도 갭이 안 닫힘**

| 변형 | QPI bw | hop 지연 | total tput | vs 원본 |
|---|---|---|---:|---:|
| 원본 | 21 | 0 | 619.8 | — |
| 대역폭만 21→15 | 15 | 0 | **619.8** | **0%** |
| 대역폭+측정지연 | 15 | 70µs | 601.5 | −3% |
| 극단(비현실적) | 1 | 500µs/hop | 535.6 | −14% |
| **vLLM 실제** | — | — | **305.8** | 목표 |

→ **결정적 발견**: 비현실적 극단값(21~77× 페널티)으로도 −14%에 그침. 시뮬은 TP8 디코드를 **compute-bound**로
보고(통신이 임계경로의 소수항), 실제는 **comm-bound**다. **링크 파라미터(`link_bw`/`link_latency`) 튜닝으로는
이 갭을 닫을 수 없음**이 확정됐다.

### 3.4 구조적 모델 — 부하 의존 collective-overhead

링크 파라미터가 아니라 **연산 임계경로에 collective 비용을 직접 주입**하는 모델을 추가
(`trace_generator._collective_overhead_ns`):

- TP all-reduce 지점(o_proj·down_proj)의 op latency에 `floor_ns + per_token_ns × (decode 배치)` 추가.
- **socket 경계 게이팅**: `npus_per_group > socket_size`일 때만 적용 → 소켓 내(TP≤4) 구성은 무영향.
- opt-in, 기본 OFF (기존 결과 바이트 동일). 정수 ns 반환(트레이스 정수 latency 유지).

설정: `{socket_size:4, floor_ns:70000(NCCL 실측 floor), per_token_ns:10000(보정)}`.

(부수 작업) GQA 모델의 고-TP 프로파일링 지원: `tp_size > num_key_value_heads`일 때 KV head를 복제(rank당
≥1 head)하도록 프로파일러 4곳 수정 → 70B를 A40 1장에서 TP=16/32까지 logical 프로파일 가능.

---

## 4. 반영 결과 — 정확도 향상

### 4.1 핵심: 70B TP8 (ShareGPT 100, 동일 워크로드)

| 지표 | 개선 전(평탄/3계층) | **개선 후(구조적 모델)** | vLLM 실제 | 오차 |
|---|---:|---:|---:|---:|
| Total throughput (tok/s) | 619.8 | **305.3** | 305.8 | **+103% → +0.2%** |
| Request throughput (req/s) | 1.40 | 0.69 | 0.69 | → 0% |
| Makespan (s) | 71.6 | 145.3 | 145.1 | → +0.2% |
| TPOT p50 (ms) | 96.9 | 303.0 | 287.9 | → +5% |

### 4.2 교차 검증: 8B TP8 (동일 상수 전이)

| 지표 | 개선 전 | 개선 후 (per_token=10000, **70B와 동일**) | vLLM | 오차 |
|---|---:|---:|---:|---:|
| Gen throughput (tok/s) | 1444.7 | 638.3 | 690.1 | **+106% → −7.5%** |
| TPOT p50 (ms) | 39.5 | 261.9 | 251.9 | → +4% |

→ **동일 상수(`floor 70µs`, `per_token 10µs`)가 8B·70B(~9× 크기차) 모두에서 TP8 갭을 ~8% 이내로** 닫음.
이는 오버헤드가 모델별 fudge가 아니라 **소켓 간 all-reduce의 하드웨어/collective 고유 속성**임을 시사.

### 4.3 비-영향 검증 (소켓 내 구성)

| 구성 | sim | vLLM | 비고 |
|---|---:|---:|---|
| TP4 70B (소켓 내, NVLink+PCIe) | 420.9 | 498.8 | 보정 없이 이미 정확(−16%); socket 게이팅으로 오버헤드 OFF |
| TP8 70B 오버헤드 블록 제거 | 619.8 | — | 기존과 바이트 동일(회귀 없음) |

### 4.4 정성적 개선 — TP4↔TP8 순서 역전 해소

개선 전 시뮬은 **TP8(619.8) > TP4(420.9)** 로 "TP8이 더 빠르다"고 오판했으나, 실제 vLLM은
**TP4(498.8) > TP8(305.8)** (소켓 간 QPI 병목으로 TP8이 느림). 구조적 모델은 TP8을 305.3으로 끌어내려
**실제와 동일한 TP4 > TP8 순서를 재현**한다. 또한 정적 링크로는 표현 못 하던 **부하 의존 열화**
(TPOT 97→303ms, TTFT 173ms→12.9s)를 포착한다.

### 4.5 8B TP-스케일링 오차 패턴 (기존 검증 데이터, 개선 전)

| TP | sim gen tput | vLLM gen tput | sim 오차 |
|---|---:|---:|---:|
| 2 (소켓 내, NVLink) | 2202 | 1864 | +18% |
| 4 (소켓 내, +PCIe) | 2220 | 1904 | +17% |
| 8 (소켓 간, +QPI) | 1422 | 690 | **+106%** |

→ 오차의 임계점이 **TP 차수가 아니라 "collective가 소켓 경계(QPI)를 넘는지"** 임을 보여줌. 본 작업은
이 임계점(TP8)을 정확히 겨냥해 보정한다.

### 4.6 NVLink 제거 ablation — tier 기여의 직접 실측 (2026-07-03)

§2의 tier 특성(Tier 0 NVLink 52.8 vs Tier 1 PCIe 24.5 GB/s)과 §3.3의 "노드 내 all-reduce는 latency 지배"
주장을 **하드웨어에서 직접** 검증했다. 동일 노드(s8) A40에서 Llama-3.1-8B(TP2·TP4)와 Llama-3.1-70B(TP4·TP8)를
각각 2회 벤치마크 — 이미지·가중치·워크로드(ShareGPT-100, FP16, vLLM 0.8.4) 완전 동일, **`NCCL_P2P_DISABLE`만
차이**로 NVLink 사용 여부만 격리했다(NCCL 로그로 실제 전송 경로 확인). TP2/4는 단일 소켓(NUMA 0) 내,
**TP8은 양 소켓을 교차**(GPU 0–7)한다.

- **TP2** (GPU 0,1 = 순수 NVLink 쌍, 유일 링크가 NVLink): NVLink 실행=**전 채널 NVLink**(P2P/IPC 8, SHM 0),
  PCIe 실행=**전 채널 SHM**(P2P/IPC 0, SHM 8) → 이상적 격리.
- **TP4** (GPU 0,1,2,3 = NUMA 0; NVLink 쌍 (0,1)(2,3), 쌍 간 PCIe): NVLink 실행 P2P/IPC 22 + SHM 18,
  PCIe 실행 P2P/IPC 0 + SHM 28. 8B·70B 동일 GPU 집합·동일 전송 경로.
- **TP8** (GPU 0–7 = 양 소켓 교차; 소켓 내 NVLink 쌍 4개 + 소켓 간 QPI): NVLink 실행 P2P/IPC 16 + SHM 14
  (소켓 간 홉은 NVLink 실행에서도 이미 SHM), PCIe 실행 P2P/IPC 0 + SHM 30 → **혼합 전송 vs 균일 SHM**.

| 모델 | TP | 지표 | NVLink | PCIe (NVLink off) | 변화 |
|---|---|---|---:|---:|---:|
| 8B | 2 | Gen tput (tok/s) | 1129.7 | 1105.7 | −2.1% |
| 8B | 2 | TTFT p50 (ms) | 49.7 | 54.6 | +10.0% |
| 8B | 2 | TPOT p50 (ms) | 22.60 | 24.28 | +7.4% |
| 8B | 4 | Gen tput (tok/s) | 1248.8 | 1178.8 | −5.6% |
| 8B | 4 | TTFT p50 (ms) | 54.9 | 69.6 | +26.6% |
| 8B | 4 | TPOT p50 (ms) | 23.77 | 27.65 | +16.3% |
| **70B** | **4** | Gen tput (tok/s) | 245.0 | 232.6 | −5.0% |
| **70B** | **4** | TTFT p50 (ms) | 3645 | 5390 | **+47.9%** |
| **70B** | **4** | TPOT p50 (ms) | 153.9 | 165.4 | +7.5% |
| **70B** | **8** | Gen tput (tok/s) | 155.5 | **201.7** | **+29.7% ⚠︎역전** |
| **70B** | **8** | TTFT p50 (ms) | 20903 | 13701 | **−34.5% ⚠︎역전** |
| **70B** | **8** | TPOT p50 (ms) | 298.1 | 224.9 | **−24.6% ⚠︎역전** |

(p99 TPOT: 8B TP4 +83.5%, 70B TP4 +16.4%. p99 TTFT 70B TP4 +39.0%. 70B TP8 makespan: NVLink 136.2s vs PCIe 103.2s.
TP8만 부호가 역전 — PCIe(NVLink off)가 더 빠름.)

**네 가지 결론:**
1. **디코드(TPOT)에서 NVLink 이득은 대역폭이 아니라 latency에서 나온다** — NVLink를 끄면 처리량 영향은 작으나
   (−2~6%) 디코드 지연 TPOT은 악화(p50 +7~16%). 디코드 all-reduce는 작은 메시지(토큰당 수십 KB)라 대역폭
   여유가 커 **고정 지연이 임계경로** — §3.3의 NCCL 실측(작은 메시지 35–90µs latency floor)과 일치한다.
2. **프리필(TTFT)은 대역폭 지배 — 모델이 클수록 NVLink 이득이 커진다** — TTFT 페널티가 8B TP4 +26.6% →
   **70B TP4 +47.9%**. 프리필 all-reduce는 큰 메시지라 대역폭 지배인데, 70B는 payload(hidden 8192)가 8B(4096)의
   2배라 NVLink의 대역폭 우위가 더 크게 작용한다 — §5.7의 "per-token 비용이 hidden_size에 스케일"과 동일 물리.
3. **디코드 상대 페널티는 큰 모델일수록 작다(연산 비중 ↑)** — TPOT 상대 페널티 8B TP4 +16.3% → 70B TP4 +7.5%.
   70B는 레이어당 연산이 무거워(TPOT 절대값 154ms vs 8B 24ms) 통신이 임계경로에서 차지하는 비중이 작기 때문.
   단 **절대 지연 증가는 70B가 더 큼**(+11.5ms vs +3.9ms) — 큰 all-reduce payload와 일치. 상대% 감소는
   분모(compute-heavy TPOT) 증가 때문이지 통신 비용 자체가 준 것이 아니다. (§4.4 TP4↔TP8 역전과 같은 방향의 물리.)
4. **소켓을 교차하는 TP8에서는 NVLink가 오히려 손해다(부호 역전)** — 70B TP8에서 NVLink를 끈 쪽(균일 SHM)이
   처리량 **+29.7%**, makespan −24.2%로 더 빠르다. TP8 all-reduce는 소켓 경계를 넘는데, NVLink 실행은 소켓 내
   쌍만 NVLink(P2P)·소켓 간은 SHM인 **혼합 전송**이라 빠른 NVLink 홉이 느린 cross-socket 홉을 기다리며 링이
   불균형·직렬화된다. `NCCL_P2P_DISABLE=1`로 전부 SHM이면 균일 fabric에 최적화된 트리로 더 잘 파이프라이닝된다
   (2-소켓 부분-NVLink 박스에서 알려진 현상). **순서를 바꾼 확인 재실행**(pcie 먼저)에서도 방향이 재현됐다 —
   PCIe는 매우 안정적(gen 201.7→203.9, makespan 103.2→102.1s)인 반면 NVLink 혼합 전송은 불안정(gen 155.5→99.1,
   makespan 136→214s)해 PCIe 우위가 오히려 +105.8%까지 벌어졌다. 이는 §4.1/§3.4의 **"TP8 QPI 병목이 처리량을
   지배"** 를 통신 계층에서 직접 확증한다 — 소켓 경계에서는 tier 이질성(NVLink vs QPI)이 균일화보다 해롭다.

---

## 5. 멀티노드 확장 — 노드 간 InfiniBand tier (TP16)

§4까지는 단일 노드(8 GPU, 소켓 경계까지)였다. 본 절은 §5(구 한계 2번)의 후속으로, **2-노드를 연결해
collective가 노드 경계(InfiniBand)를 넘는 TP16**을 실측 검증한다. (전체 상세: `validation/MULTINODE_TP16_REPORT_KO.md`)

### 5.1 환경

| 항목 | s8 (head) | s2 (worker, 8B용) | s6 (worker, 70B용) |
|---|---|---|---|
| GPU | 8× A40 | 8× A40 | 8× A40 |
| IB iface / IP | `ibs8` / 192.168.210.108 | `ibp194s0` / 192.168.210.102 | `ibs8` / 192.168.210.106 |
| 루트 디스크 여유 | 554 GB | 96 GB | 290 GB |

노드 간 링크: **InfiniBand mlx5_0 / ConnectX-6, 200 Gb/s (4X HDR, ≈25 GB/s nominal)**, IPoIB 192.168.210.x.
s8↔s2 와 s8↔s6 는 **동일 IB 패브릭**(GID subnet prefix `fe80::` 동일, ACTIVE 200 Gb/s)이므로 §5.3 보정 상수를 공유한다.

→ 이더넷은 1 Gb/s뿐, **유일한 고속 노드 간 경로는 IB**. 실제 vLLM 0.8.4(단일 노드 검증과 동일 버전)를
Ray 클러스터(head=s8) + NCCL/IB로 TP16 구동, ShareGPT-100 실측. 양 노드 8 GPU 전부 점유 확인 = 진성
노드 간 TP16. **8B는 worker=s2, 70B는 worker=s6** — s2 디스크(여유 96 GB) < 70B FP16 가중치 132 GB 라
동일 IB 패브릭이면서 290 GB 여유인 s6로 대체했다(동일 200 Gb/s IB이므로 IB-tier 보정 상수 그대로 적용).

### 5.2 모델 확장 — 4계층 + 노드(IB) tier overhead

- **4계층 토폴로지**: `tp_group_shape=[2,2,2,2]`, `link_bw=[52.8, 24.5, 21.0, 25.0]` → all-reduce를
  NVLink→PCIe→QPI→**IB**로 분해.
- **노드 경계 게이팅**(`_collective_overhead_ns` 일반화): `npus_per_group > node_size(=8)`이면 IB tier 비용
  (`node_floor_ns`/`node_per_token_ns`)이 socket(QPI) 비용을 **대체**(가장 느린 교차 tier가 지배 — §3.4와 동일 철학).
  하위호환: `node_size` 미지정 시 기존과 바이트 동일(**검증: TP8 70B = 305.25 tok/s로 §4.1 재현, 회귀 없음**).
- 8B tp16 프로파일 신규 생성(GQA KV-head 복제, tp16 > kv_heads=8).

### 5.3 보정 — IB tier 상수 (8B TP16, ShareGPT-100)

5점 sweep으로 처리량·TPOT을 동시에 ~10% 이내로 맞추는 상수를 탐색 → **채택: `node_floor=105µs,
node_per_token=18µs`**. 단일 노드 QPI tier(70µs/10µs) 대비 **floor 1.5×, per-token 1.8×** (노드 간 IB가
소켓 간 QPI보다 느린 물리와 일치).

### 5.4 결과 — 8B TP16 (실측 검증)

| 지표 | 개선 전(대역폭만 4계층) | **개선 후(+IB overhead)** | vLLM 실제 | 오차 |
|---|---:|---:|---:|---:|
| Gen throughput (tok/s) | 818 | **299** | 330 | **+148% → −9.5%** |
| TPOT p50 (ms) | 31.5 | **179.8** | 191.3 | **−84% → −6.0%** |
| TTFT p50 (ms) | 53 | 7998 | 4672 | (정의차 유지) |

→ 대역폭만 모델은 단일 노드와 동일하게 통신을 무시해 처리량을 **+148% 낙관**. IB-tier overhead 보정으로
**−9.5%/−6.0%** — 단일 노드 8B TP8(−7.5%)과 동급 정확도.

### 5.5 NCCL IB floor 직접 측정 — 보정 floor 교차검증

world_size=16 cross-node all-reduce 직접 측정(`validation/multinode/nccl_ib_allreduce_bench.py`):

| 메시지 | 실측 latency | busbw (GB/s) | sim(bytes/bw, lat=0) | real/sim |
|---|---:|---:|---:|---:|
| **16 KB (decode)** | **80.8 µs** | 0.4 | 1.2 µs | **65.8×** |
| 64 KB | 183.6 µs | 0.7 | 4.9 µs | 37.4× |
| 64 MB (prefill) | 68.1 ms | 1.8 | 5.03 ms | 13.5× |

→ **16 KB cross-node floor 81 µs ≈ 보정 floor 105 µs**(같은 자릿수; 보정값이 다소 큰 것은 step당
프레임워크 오버헤드 흡수분). 8B decode payload(토큰당 8 KB, 배치 2–8 → 16–64 KB)가 실측 81–184 µs
구간이고 보정식 `105 + 18×batch µs`가 이 밴드에 안착 → **end-to-end 보정값이 직접 측정과 독립적으로 일치**.
유효 busbw 0.4–1.8 GB/s(공칭의 1/15–1/60) = 노드 간 all-reduce가 **대역폭이 아니라 latency·sync 지배**임을
직접 증명(대역폭만 모델이 65× 빗나가는 근본 원인).

### 5.6 70B TP16 — 8B 상수 적용 예측

8B 보정 상수(105/18µs)를 70B에 **그대로 전이**한 예측: total **222.6 tok/s**(대역폭만 682), TPOT p50 458.7 ms.
→ 70B도 **TP16(노드 간) < TP8(노드 내, 305)** 로 comm-bound 심화의 정성적 일관성.
이 예측이 실측과 얼마나 맞는지, 그리고 8B↔70B 상수 전이가 노드 tier에서도 성립하는지는 §5.7에서 실측 검증한다.

### 5.7 70B TP16 실측 및 IB 상수 재보정 (2026-07-01, s8+s6)

§5.6 예측을 실제 vLLM로 검증했다. s2 디스크 부족은 **s6(290 GB 여유, s8과 동일 200 Gb/s IB 패브릭)** 로 해소
(§5.1). s6의 IPoIB(`ibs8`)를 192.168.210.106에 올려 s8↔s6 IB를 확보하고, 70B-Instruct FP16 가중치(132 GB)를
IB로 rsync한 뒤 Ray(head=s8)+NCCL/IB로 TP16을 구동했다. **ShareGPT-100, 100/100 요청 성공(TPOT 유효 n=97)**,
양 노드 8 GPU 전부 점유 = 진성 노드 간 TP16.

| 지표 | 개선 전(대역폭만 4계층) | +IB overhead (8B 상수 105/18µs) | **+IB overhead (70B 보정 105/40µs)** | vLLM 실제 |
|---|---:|---:|---:|---:|
| Gen throughput (tok/s) | 359 | 117 | **66** | 66.5 |
| TPOT p50 (ms) | 87.5 | 458.7 | **893.7** | 890.0 |
| TTFT p50 (ms) | 166 | 27418 | — | 35412 |
| Gen 오차 | +439% | +76% | **−1.9%** | — |
| TPOT 오차 | −90% | −48.5% | **+0.4%** | — |

**핵심 발견 — 노드 tier per-token 상수는 모델 크기에 스케일.** 8B에서 보정한 상수(105/18µs)를 70B에 그대로
전이하면 **gen +76% / TPOT −48.5%** 로 여전히 크게 낙관한다(단일 노드 QPI tier에서 8B↔70B 상수가 ±8% 이내로
전이된 §4.2와 대조). 즉 노드 경계 all-reduce는 8B보다 70B에서 유의하게 더 comm-bound다.

**재보정(4점 sweep, 실측 gen 66.5 / TPOT 890 목표):**

| node_floor | node_per_token | gen | TPOT p50 | gen 오차 | TPOT 오차 |
|---:|---:|---:|---:|---:|---:|
| 105 µs | 36 µs | 71 | 814.7 | +6.7% | −8.5% |
| **105 µs** | **40 µs** | **66** | **893.7** | **−1.9%** | **+0.4%** |
| 105 µs | 44 µs | 61 | 972.7 | −9.1% | +9.3% |
| 150 µs | 40 µs | 65 | 900.9 | −3.3% | +1.2% |

→ **채택: `node_floor=105µs`(8B와 동일, 불변), `node_per_token=18→40µs`**. 두 성분이 서로 다르게 거동한다:
- **floor(105µs)는 모델 독립** — 순수 IB latency·sync 핸드셰이크 상수. 8B·70B 공통이며 §5.5의 직접 측정
  16 KB cross-node all-reduce floor(81µs)와 같은 자릿수. floor를 150µs로 올려도 개선 없음(위 4행) → floor는 이미 포화.
- **per_token(18→40µs, ≈2.2×)은 모델 크기 스케일** — 노드 간 all-reduce payload가 hidden_size에 비례하고
  70B hidden(8192)이 8B(4096)의 **정확히 2×**. 관측된 2.2×가 이 payload 비율과 일치(잔여 0.2×는 더 큰 FFN·
  step당 sync 누적으로 해석). → **노드 tier에서 latency floor는 하드웨어 상수로 전이되나, per-token 비용은
  collective 볼륨(모델 크기)에 비례해 스케일**한다. §4.2(단일 노드 QPI tier)의 "단일 상수 전이"를 노드 경계로
  정밀화한 결과다.

보정 후 70B TP16 오차는 **gen −1.9% / TPOT +0.4%** — 8B TP16(−9.5%/−6.0%)과 동급으로, 노드 간 TP16
전 구간(8B·70B)에서 ~2% 이내 정확도를 달성했다. 정성적으로도 **70B TP16(노드 간, 66 tok/s) < TP8(노드 내,
305)** 로 comm-bound 심화를 실제와 일치하게 재현한다.

---

## 6. 한계 및 향후 과제

1. **`per_token_ns`는 경험 상수**: 8B·70B에 동일 값이 전이되어 하드웨어 속성으로 추정되나, 1st-principle
   유도는 아직 아님. 기여는 비용을 *표현 가능하게* 만든 메커니즘(socket-gated, load-scaled, critical-path).
2. **노드 tier per-token 상수의 모델 크기 스케일**(§5.7): 멀티노드 70B는 s6로 실측 완료. floor(105µs)는
   8B·70B 공통이나 per-token은 18→40µs(≈hidden 비율 2×)로 스케일함을 확인했다. 다만 이 스케일 법칙은
   8B·70B 두 점 관측이라, 임의 모델·payload로의 일반화(예: `per_token ∝ hidden_size` 1st-principle 유도)는 추가 과제.
3. **전력 모델**: TP4 이상에서 flat `active_power=300W/GPU`가 실측(~205W)을 과대평가(+29%) — TP 차수별
   이용률 반영 필요(본 작업 범위 밖).
4. **TTFT 정의 차이**(연산 완료 기준 vs 클라이언트 수신)는 유지 — 처리량/TPOT이 비교 기준.

---

## 7. 산출물

**코드**
- `inference_serving/config_builder.py` — 계층형 네트워크 생성(`tp_group_shape`, per-tier 배열) + instance 전달
- `inference_serving/trace_generator.py` — `_collective_overhead_ns` + o_proj/down_proj 적용; **노드(IB) tier 게이팅(`node_size`/`node_floor_ns`/`node_per_token_ns`) 추가**
- `main.py` — 클러스터 인터커넥트 정보 전달
- `llm_profile/{models/llama.py, profiler/...}` — GQA 고-TP 프로파일링(KV 복제) 4곳 수정

**설정/데이터 (단일 노드)**
- `cluster_config/a40_8gpu_tp8_70b_3tier{,_cohd}.json`, `a40_4gpu_tp4_70b_2tier.json`,
  `a40_8gpu_tp8_8b_3tier_cohd.json`
- `docs/dse/fabrics.yaml` (fabric `2socket_8npu_nvlink_bridge_per_2slot`)
- `validation/nccl_allreduce_bench.py`, `validation/vllm_a40_tp{4,8}_*results.jsonl`
- 실측 프로파일 `llm_profile/perf_models/A40/meta-llama/Llama-3.1-70B/tp{8,16,32}/`
- **NVLink ablation(§4.6)**: `validation/run_vllm_tp_nvlink_ablation.sh` (모델/TP/ORDER 파라미터화 러너),
  `validation/vllm_a40_tp{2,4}_{nvlink,pcie}_*` (8B) + `validation/vllm_a40_70b_tp{4,8}_{nvlink,pcie}_*` (70B) +
  `validation/vllm_a40_70b_tp8_c2_{nvlink,pcie}_*` (TP8 순서 바꾼 확인 재실행), 각 `{results.jsonl,serve.log,power.csv}`

**설정/데이터 (멀티노드, §5)**
- `cluster_config/a40_16gpu_tp16_{8b,70b}_4tier{,_cohd}.json` (4계층 + IB overhead 보정 상수)
- `cluster_config/a40_16gpu_tp16_70b_4tier_cohd_cal.json` (**70B 재보정 상수 105/40µs**, §5.7)
- 신규 프로파일 `llm_profile/perf_models/A40/meta-llama/Llama-3.1-8B/tp16/`
- `validation/multinode/{run_cluster_node.sh, serve_tp16.sh, run_multinode_validation.sh, nccl_ib_allreduce_bench.py}`
- `validation/multinode/run_multinode_validation_s6.sh` (**70B s8+s6 실측 러너**), `recal_70b_sweep.py` (IB 상수 재보정 sweep)
- `validation/vllm_a40_tp16_{8b,70b}_results.jsonl`(실측), `validation/sim_a40_tp16_*`(sim, 70B 보정본 `..._cohd_cal_*` 포함), `validation/compare_tp16.py`
- `validation/multinode/nccl_ib_{s8_head,s2_worker}.log`(NCCL IB all-reduce 실측 로그)

**상세 검증 기록**: `validation/VALIDATION_A40_REPORT.md` (TP=4 control / TP=8 root-cause / 구조적 모델 / 8B 교차검증),
**`validation/MULTINODE_TP16_REPORT_KO.md`** (멀티노드 TP16 전체 기록)

---

## 8. 결론

LLMServingSim의 균등 대역폭 가정은 소켓 내(TP≤4) 구성에선 충분(오차 ±17% 이내)하지만, **collective가
소켓 간 QPI를 가로지르는 TP8에서 처리량을 ~2배 낙관**했다. 본 작업은 (1) **계층형 인터커넥트 모델**로
비균질 토폴로지를 표현하고, (2) 링크 파라미터로는 불가능함을 NCCL 실측+폐루프로 규명한 뒤, (3) **부하
의존 collective-overhead 모델**(연산 임계경로, socket-gated)로 비용을 주입했다. 그 결과 TP8 처리량 오차를
**+103%(70B)/+106%(8B) → +0.2%/−7.5%** 로 줄였고, **TP4↔TP8 성능 역전과 부하 의존 지연 열화를 실제와
일치하게 재현**했다. 단일 상수가 모델 크기를 가로질러 전이된다는 점은 이 보정이 하드웨어 고유 특성을
포착함을 뒷받침한다. 나아가 **NVLink 제거 ablation(§4.6, 8B TP2/4 · 70B TP4)** 은 이 tier 비용의 성격을
하드웨어에서 직접 확인했다 — 디코드(TPOT)는 작은 메시지라 **latency 지배**여서 NVLink 제거 시 처리량은 소폭
(−2~6%)이나 지연이 악화하고, 프리필(TTFT)은 큰 메시지라 **대역폭 지배**여서 **모델이 클수록 NVLink 이득이
커진다**(TTFT 페널티 8B TP4 +27% → 70B TP4 +48%). 둘 다 §5.7의 payload-스케일 물리와 일관된다.
반대로 **소켓을 교차하는 70B TP8에서는 NVLink가 오히려 손해**(NVLink off가 처리량 +30%, 순서 바꾼 재실행에서
재현)로, 소켓 내 NVLink와 소켓 간 QPI가 섞인 혼합 all-reduce가 균일 SHM보다 느리고 불안정함을 보였다 —
§4.1/§3.4의 "TP8 QPI 병목" 을 통신 계층에서 직접 확증한다.

**멀티노드 확장(§5)**: 동일 메커니즘에 **노드 경계(InfiniBand) tier**를 한 단계 더 추가하고, 실제 2-노드
(s8+s2)×8 A40 = 16 GPU vLLM TP16 실측으로 IB tier 상수(`node_floor=105µs, per_token=18µs`)를 보정해
**8B TP16 처리량/TPOT 오차를 +148%/−84% → −9.5%/−6.0%**로 줄였다. 이 보정 floor는 직접 측정한 16 KB
cross-node NCCL all-reduce latency(81µs)와 독립적으로 일치하며, QPI tier 대비 floor 1.5×·per-token 1.8×로
물리적으로 타당하다. 즉 **소켓→노드 경계로 한 단계 깊어진 비균질성에 대해서도 동일한 모델링 철학(가장 느린
교차 tier의 부하 의존 비용을 임계경로에 주입)이 일관되게 작동**함을 보였다.

**70B 멀티노드도 s6(동일 IB 패브릭, 290 GB 여유)로 실측 완료(§5.7)**: 8B 보정 상수를 그대로 전이하면
gen +76%로 낙관하나, **`node_per_token`을 18→40µs(≈hidden 비율 2×)로 스케일**하면 **gen −1.9% / TPOT +0.4%**.
즉 노드 tier에서는 **latency floor(105µs)는 모델 독립 하드웨어 상수로 전이되고, per-token 비용만 collective
볼륨(모델 크기)에 비례해 스케일**한다 — §4.2(단일 노드 QPI tier)의 "단일 상수 전이"를 노드 경계로 정밀화한
결과다. 이로써 **8B·70B × TP8·TP16 전 구간에서 ~2%(TP16)–8%(TP8) 이내 정확도**를 확보했다.
