# PLAN_execution_layer.md — v3 실행 계층 (토폴로지 그래프 + vLLM 배포·모니터링·해제)

> **목표**: 이종 GPU/NPU 클러스터를 노드 내/노드 간 연결망까지 표현하는 **토폴로지
> 그래프**로 모델링하고, 「모델 + SLO + 규모」 요청에 대해 가용 자원 중 **전력효율
> 최적 조합을 선택·잠금 → Docker로 vLLM 자동 기동 → UI에서 모니터링 → 종료 시
> 자원 해제**까지 전 수명주기를 관리하는 실행 계층을 구현한다.
> **설계 근거 문서**: `LLM_서빙_전력최적화_서비스_설계서_v3.docx` (§ 번호는 그 문서를 가리킴)
> **선행**: PLAN_power_service.md M0–M10 완료 상태를 전제한다 (추천·예약 파이프라인).

---

## 0. 문서 정보

| 항목 | 내용 |
|---|---|
| 대상 저장소 | `ai-computing/LLMServingOptimizer` (main) |
| 신규 모듈 | `service/topology/`, `service/deploy/`, `service/monitor/`, `service/api/deployment_routes.py`, `webapp/templates/service/`, `webapp/static/service/` |
| 수정 모듈 | `service/inventory/registry.py`(v2 스키마), `service/inventory/ledger.py`(deployments 테이블), `service/api/routes.py`(confirm 훅), `service/api/models.py`(스키마 확장), `webapp/app.py`(라우터 mount) |
| 금지 사항 | `backends/*`(submodule) 수정 금지. 기존 planner/DSE/service 공개 동작 회귀 금지. `inventory/views.py`의 planner 출력 스키마 변경 금지 |
| 테스트 러너 | `pytest` — 기존 마커 유지: `unit`(기본), `sim`(시뮬 필요), `hw`(실기기 필요, CI 제외). 신규 마커 `docker`(로컬 dockerd 필요, CI 옵션) 추가 |
| UI 테스트 | Playwright (D4). fixture vLLM 서버(가짜 /metrics·/health)로 실기기 없이 구동 |

**구현 순서는 D0→D5.** 각 마일스톤은 "구현 → 테스트 → DoD 체크" 순으로 완료해야
다음으로 넘어간다. 테스트가 실패한 채로 다음 마일스톤을 시작하지 않는다.

```
pytest -m unit                                        # 매 커밋
pytest -m "unit or sim" planner/tests/ tests/ service/tests/    # 마일스톤 완료 시
pytest -m docker service/tests/                       # dockerd 있는 환경에서
```

---

## 1. 설계 원칙 (모든 마일스톤 공통, §1.3)

1. **비침습**: 시뮬레이터 submodule은 블랙박스. 신규 기능은 `service/` 하위 모듈과
   webapp 라우터 추가로만 구현한다.
2. **단일 진실 원천**: 클러스터 상태(토폴로지·예약·배포)는 registry YAML + SQLite
   원장이 유일한 원천. 그래프·UI·모니터링은 모두 그 위의 파생 뷰다.
3. **추천과 실행의 분리**: planner는 Docker를 모른다. `vllm_launcher`가
   allocation → 실행 명세(DeploymentSpec) 번역을 전담한다.
4. **상태 기계 기반**: 배포 수명주기는 명시적 상태 기계로 관리하고 모든 전이를
   `deployment_events`에 기록한다. 서비스 재시작 후 이 기록만으로 복구 가능해야 한다.
5. **SLO는 계약**: 추천 시 약속한 SLO를 실행 중에도 30초 윈도우로 지속 검증하고
   위반을 1급 이벤트(DEGRADED 전이 + UI 경고)로 노출한다.

## 2. 신규 디렉토리 구조 (§3.3)

```
service/
├── topology/                  # D0–D1 (G1)
│   ├── graph.py               #  TopologyGraph (networkx MultiGraph 래퍼)
│   ├── schema.py              #  registry v2 pydantic 스키마 (+v1 자동 승격)
│   ├── discovery.py           #  nvidia-smi topo -m / lspci 파서 → 초안 YAML
│   └── views.py               #  free 서브그래프 → planner topology 직렬화
├── deploy/                    # D2 (G2)
│   ├── manager.py             #  DeploymentManager (상태 기계 구동, 워커 스레드)
│   ├── docker_driver.py       #  docker SDK 래퍼 (원격 dockerd, TLS/SSH)
│   ├── vllm_launcher.py       #  allocation → DeploymentSpec 번역
│   ├── state.py               #  DeploymentState enum + 허용 전이 표
│   └── store.py               #  deployments/deployment_events CRUD (ledger DB 공유)
├── monitor/                   # D3 (G3)
│   ├── collector.py           #  vLLM /metrics 폴러 (prometheus text 파서, 5s)
│   ├── power.py               #  nvidia-smi dmon / DCGM / furiosa-smi 폴러 (5s)
│   ├── slo_checker.py         #  30s 슬라이딩 윈도우 SLO 판정
│   ├── logs.py                #  docker logs --follow → 링버퍼 → SSE 브리지
│   └── store.py               #  링버퍼(1h 실시간) + SQLite(5분 집계 영속화)
└── api/
    ├── deployment_routes.py   #  D2–D3: REST + SSE
    └── models.py              #  (확장) Deployment 관련 pydantic 스키마

webapp/
├── templates/service/         # D4
│   ├── layout.html            #  Service 탭 공통 프레임(서브탭·SSE 연결 상태)
│   ├── cluster.html           #  화면 ① 클러스터 뷰
│   ├── request.html           #  화면 ② 서빙 요청
│   └── deployments.html       #  화면 ③ 배포 목록+상세
└── static/service/
    ├── common.js              #  SSE 클라이언트(지수 백오프 재연결)·포맷터·토스트·모달
    ├── cluster_view.js        #  d3-force 렌더러 + 상태 색상 매퍼
    ├── request_form.js        #  폼 검증·진행 스테퍼·추천 카드·대안 테이블
    ├── deploy_dashboard.js    #  Chart.js 차트·KPI·로그 뷰어·종료 플로우
    └── service.css            #  색상 토큰(CSS 변수) 단일 정의
```

외부 JS 의존은 **d3(force·zoom·drag)와 Chart.js 두 가지로 한정**하고
`webapp/static/vendor/`에 vendoring한다(오프라인 클러스터 환경 고려).

## 3. 핵심 데이터 구조 (§4 — 구현 시 이 정의를 그대로 따를 것)

### 3.1 토폴로지 그래프 (`service/topology/`)

무방향 **MultiGraph**. 정점/간선 속성은 pydantic으로 검증.

| 정점 | 주요 속성 |
|---|---|
| `HostNode` | id, hostname, host_base_w, docker{endpoint, tls}, numa_domains, state |
| `DeviceNode` | id(`node0/A40/3` — **기존 ledger 식별자 규약과 동일**), kind(gpu\|npu), hw, mem_gb, idle_w, active_w, pcie_gen, numa, state(free\|reserved\|running\|faulted) |
| `SwitchNode` | id, kind(nvswitch\|pcie_switch\|nic\|tor_switch), bandwidth |

| 간선 | 주요 속성 |
|---|---|
| `IntraLink` | kind(nvlink\|xgmi\|pcie\|npu_link), bandwidth_gbps, lanes/gen, latency_us |
| `InterLink` | kind(ethernet\|infiniband\|cxl), bandwidth_gbps, latency_us, rdma(bool) |

**registry YAML v2** (`cluster_registry.yaml`):

```yaml
version: 2
nodes:
  - id: node0
    hostname: gpu-a40-0.cluster
    host_base_w: 250
    docker: {endpoint: "tcp://gpu-a40-0:2376", tls: true}
    devices:
      - {id: node0/A40/0, kind: gpu, hw: A40, mem_gb: 48,
         idle_w: 60, active_w: 300, pcie: gen4x16, numa: 0}
    switches:
      - {id: node0/plx0, kind: pcie_switch}
    intra_links:
      - {a: node0/A40/0, b: node0/A40/1, kind: nvlink, bandwidth: 112GBps}
      - {a: node0/A40/0, b: node0/plx0,  kind: pcie,   bandwidth: 32GBps}
  - id: node1
    devices:
      - {id: node1/RNGD/0, kind: npu, hw: RNGD, mem_gb: 48, active_w: 150,
         runtime: furiosa, device_path: /dev/rngd0}
inter_links:
  - {a: node0/nic0, b: node1/nic0, kind: ethernet,
     bandwidth: 100Gbps, latency: 10us, rdma: false}
```

**v1 하위호환(중요)**: 현행 `cluster_registry.example.yaml`(v1)은 로더가 자동
승격한다 — devices `count` → 개별 DeviceNode 전개, `intra_node_bandwidth` →
완전연결 IntraLink, `links` → InterLink. **기존 v1 파일은 무수정 동작해야 한다.**

**파생 연산** (graph.py):
- `free_subgraph(ledger_snapshot)` — 예약 디바이스 제외 유도 부분그래프
- `affinity(device_set) -> float` — 집합 내부 최소 절단 대역폭(min-cut).
  TP 그룹 배치 시 NVLink 집합 > PCIe-only 집합 우선 점수
- `to_planner_topology()` — planner spec topology dict 직렬화.
  **기존 `inventory/views.py` 출력 스키마와 동일해야 함** (교체 후 골든 비교)

### 3.2 원장 확장 (`ledger.py` / `deploy/store.py`)

기존 `reservations` 테이블 유지. 추가:

```sql
CREATE TABLE deployments (
  id TEXT PRIMARY KEY,              -- dep-<ulid>
  reservation_id INTEGER NOT NULL REFERENCES reservations(id),  -- 1:0..1
  tenant TEXT NOT NULL,
  model TEXT NOT NULL,
  state TEXT NOT NULL,
  spec_json TEXT NOT NULL,          -- DeploymentSpec 직렬화
  endpoints_json TEXT,              -- {openai_url, metrics_url, ...}
  slo_json TEXT,                    -- 약속한 SLO (감시 기준)
  created_at TEXT, ready_at TEXT, terminated_at TEXT,
  last_error TEXT
);
CREATE TABLE deployment_events (    -- 상태 전이 감사 로그
  dep_id TEXT, ts TEXT, from_state TEXT, to_state TEXT, detail TEXT
);
```

### 3.3 Deployment 구조체 (`deploy/`)

```python
class ContainerSpec(BaseModel):
    node_id: str
    role: Literal["standalone", "head", "worker"]
    image: str                    # 버전 태그 고정 (vllm/vllm-openai:vX.Y)
    device_ids: list[str]         # [node0/A40/0, ...]
    gpu_indices: list[int]        # 호스트 로컬 인덱스 → --gpus 바인딩
    env: dict[str, str]           # NCCL_*, VLLM_HOST_IP 등
    ports: dict[str, int]         # {api: 8001, metrics: 8001}
    shm_size: str = "16g"

class DeploymentSpec(BaseModel):
    model: str; served_name: str
    engine_args: dict             # tp, dtype, max_model_len, gpu_memory_utilization...
    containers: list[ContainerSpec]
    health: HealthPolicy          # 주기·타임아웃·재시도 N·실패 정책

class Deployment(BaseModel):
    id: str; reservation_id: int; tenant: str
    spec: DeploymentSpec; state: DeploymentState
    endpoints: Endpoints | None; slo: SLOSpec
```

**상태 기계** (state.py — 허용 전이 외에는 `InvalidTransition` 예외):

```
PENDING → PULLING → STARTING → HEALTH_CHECK → READY
READY ↔ DEGRADED                     (SLO 지속 위반 / 헬스 간헐 실패 ↔ 회복)
READY|DEGRADED → DRAINING → STOPPED → RELEASED    (정상 종료)
임의 상태 → FAILED                    (재시도 소진·노드 다운; last_error 기록)
```

### 3.4 모니터링 데이터 (`monitor/`)

```python
class MetricSample(BaseModel):     # 5s, vLLM /metrics 파생
    ts: datetime; dep_id: str
    running: int; waiting: int
    kv_cache_usage: float          # 0..1
    gen_toks_per_s: float; prompt_toks_per_s: float
    ttft_p50_ms: float; ttft_p95_ms: float
    tpot_p50_ms: float; tpot_p95_ms: float

class PowerSample(BaseModel):      # 5s, DCGM/nvidia-smi/furiosa-smi
    ts: datetime; device_id: str
    power_w: float; util: float; mem_used_gb: float; temp_c: float

class SLOStatus(BaseModel):        # 30s 슬라이딩 윈도우
    dep_id: str; window_s: int
    targets: SLOSpec
    observed: dict[str, float]
    verdict: Literal["ok", "warn", "violated"]
    breaches: list[str]            # ["tpot_ms: 231 > 200"]
```

보존: 실시간은 배포당 인메모리 링버퍼(1시간), SQLite에는 5분 집계만.
종료 시 총 에너지(Wh)·평균 전력·SLO 준수율 요약 기록(추천 예측치와 사후 대조용).

### 3.5 API (§4.5 — `deployment_routes.py`)

| 메서드/경로 | 응답 | 비고 |
|---|---|---|
| POST `/api/serve-requests/{id}/confirm` | ConfirmOut + `deployment_id` | `auto_deploy=true`(기본)면 배포 생성 연쇄 |
| POST `/api/deployments` | Deployment | {job_id \| reservation_id, engine_overrides} — 수동 경로 |
| GET `/api/deployments` | list[DeploymentSummary] | 테넌트·상태 필터 |
| GET `/api/deployments/{id}` | Deployment + SLOStatus | 컨테이너·엔드포인트·최근 이벤트 포함 |
| GET `/api/deployments/{id}/metrics` | SSE | MetricSample+PowerSample+SLOStatus 5초 push |
| GET `/api/deployments/{id}/logs` | SSE (?tail=500) | stdout/stderr follow |
| POST `/api/deployments/{id}/terminate` | Deployment | {drain_timeout_s}. **멱등** |
| GET `/api/cluster/graph` | TopologyGraphOut | UI용: 정점/간선 + 상태 |

## 4. 핵심 동작 흐름 (§5)

### 4.1 전력최적 선택 (기존 파이프라인 + 신규 5단계)

1. `ledger.snapshot()` → (ver, 가용 디바이스) → `free_subgraph`
2. 규모 → Workload Synthesizer → demand_toks_per_s + 평가용 합성 워크로드 (기존 M3)
3. Stage-1 CP-SAT: `min Σ n_t·P(hw_t)·tp_t + Σ host_base_w·used[h]` s.t. 처리량≥수요 (기존 M2)
4. Stage-2: Fidelity Router가 고른 백엔드로 SLO·전력 최종 판정 (기존 M9)
5. **배치 세부화(신규)**: hw×수량 → 실제 device id 매핑 시 `affinity` 최대화.
   같은 TP 그룹은 NVLink/NVSwitch 연결 집합·같은 NUMA·같은 PCIe 스위치 하위 우선.
   동점이면 idle 디바이스가 적은 호스트 선택(호스트 대기전력 절약).
6. infeasible → 기존 `InfeasibleReport` (병목 제약 + 완화 시나리오)

### 4.2 배포 기동 시퀀스

```
confirm(job)                         # 원장 예약 (기존)
  └▶ manager.create(dep)             # PENDING, deployments insert
       ├▶ docker_driver.pull(...)    # PULLING  (노드 병렬)
       ├▶ docker_driver.run(...)     # STARTING (head→worker 순)
       ├▶ wait /health, /v1/models   # HEALTH_CHECK (타임아웃·재시도 정책)
       ├▶ state=READY, endpoints 공개
       └▶ monitor.attach(dep)
실패: 재시도 N회 → FAILED, 생성 컨테이너 전부 정리(rollback),
      예약은 유지 — 사용자에게 [동일 자원 재시도]/[자원 해제] 제시
```

번역 규칙(vllm_launcher):
- `tp` = 그룹당 디바이스 수, `gpu_memory_utilization` 기본 0.90
- GPU 바인딩: 호스트 로컬 인덱스로 `--gpus '"device=0,1"'`
- NCCL env는 그래프의 링크 종류(nvlink/pcie)로 결정
- NPU(RNGD): furiosa 런타임 이미지 + device_path 매핑, 동일 OpenAI 호환 서버
- 포트: 노드별 풀 8001–8099, 컨테이너 이름 = dep-id 기반 결정적 생성
- 멀티노드(P1/D5): Ray head→worker, `distributed-executor-backend=ray`

### 4.3 모니터링

- collector: `/metrics` 5s 폴링 — `vllm:num_requests_running/waiting`,
  `gpu_cache_usage_perc`, TTFT/TPOT 히스토그램 → p50/p95 계산
- power: DCGM(권장) 또는 nvidia-smi / furiosa-smi. 배포 전력 = Σ소속 디바이스 + 호스트 base 안분
- slo_checker: 30s 윈도우 p95 vs 약속 SLO. **3회 연속 위반 → violated + DEGRADED 전이**
- logs: `docker logs --follow` 노드별 스레드 → 링버퍼 → SSE. FAILED 시 마지막 200줄 보존
- 수집기 수명: READY에서 attach, STOPPED에서 detach.
  **서비스 재시작 시 deployments 테이블에서 READY를 찾아 재-attach**

### 4.4 종료·해제·장애

- terminate → DRAINING: 신규 차단 불가(P0, 게이트웨이 없음)이므로
  waiting=0 ∧ running=0까지 최대 drain_timeout_s 대기 → stop(SIGTERM, grace 30s) → rm
- RELEASED: `ledger.release`(멱등) + 에너지 요약 기록
- reconcile 주기 태스크: 원장 ↔ 실제 컨테이너 양방향 대조(키=dep-id 이름 규약)
- 장애 대응표: 컨테이너 비정상 종료→재기동 N회→FAILED / 모델 OOM→gpu_mem_util 하향 1회
  재시도 / SLO 지속 위반→DEGRADED / 노드 다운→FAILED + 디바이스 faulted(후속 추천 제외)
  / 서비스 재시작→테이블 기준 복원(중간 상태는 rollback 후 PENDING 재큐)

## 5. UI 상세 명세 (§6 — D4에서 이대로 구현)

### 5.1 공통

- Service 탭 하위 서브탭 3개: **클러스터 | 서빙 요청 | 배포**. 12-컬럼 그리드
  (주 콘텐츠 8 + 컨텍스트 패널 4), 1024px 미만 단일 컬럼 강등.
- 갱신: 클러스터 뷰 10s 폴링, 대시보드 SSE 5s, 액션은 낙관적 갱신 후 서버 확정 재렌더.
- 단위: 전력 W(정수)/kW(소수1), 에너지 Wh, 지연 ms, 처리량 tok/s. tabular 숫자 우측 정렬.
- SSE 연결 상태 헤더 상시 표시, 유실 시 지수 백오프 + 마지막 수신 시각.
- 파괴적 액션은 2단 확인 모달(버튼에 대상 이름 포함).
- **색상 토큰은 `service.css`의 CSS 변수로 단일 정의**, JS는 클래스만 부여:

| 토큰 | 값 | 용도 |
|---|---|---|
| `--state-free` | #9E9E9E | 디바이스 free (옅은 채움+회색 테두리) |
| `--state-reserved` | #FB8C00 | 디바이스 reserved |
| `--state-running` | #2E7D32 | 디바이스 running / 배포 READY |
| `--state-faulted` | #C62828 | 디바이스 faulted / 배포 FAILED |
| `--dep-progress` | #1E88E5 | PENDING~HEALTH_CHECK (배지+스피너) |
| `--dep-degraded` | #F9A825 | DEGRADED |
| `--dep-neutral` | #757575 | DRAINING/STOPPED |
| `--slo-ok/warn/violated` | 초록/황색/빨강 | SLO 게이지·목표선·배지 |

### 5.2 화면 ① 클러스터 뷰 (cluster.html + cluster_view.js)

```
┌─ 툴바: 필터[노드▾ 상태▾ 링크▾] □ 전력 오버레이 ────────────────┐
├─ 그래프 캔버스(8칸) ────────────────┬─ 요약 패널(4칸) ─────────┤
│  host hull 안에 디바이스 사각형      │ 총 디바이스/상태별 카운트  │
│  링크: NVLink 굵은초록, PCIe 회색파선│ 현재 총 전력 kW           │
│        Ethernet 갈색, IB 파랑 이중선 │ 호스트별 전력 막대         │
├─ 선택 상세 바(하단 고정) ────────────┴──────────────────────────┤
│ node0/A40/3 · running · dep-01H2…(teamA) · 287W · 44/48GB      │
│                                  [배포 상세로 이동 →]           │
└────────────────────────────────────────────────────────────────┘
```

렌더링 규칙: HostNode=둥근 사각 hull(라벨: id, host_base_w; 다운 시 옅은 적색),
GPU=사각형/NPU=모서리 깎인 사각형(채움=상태 토큰, 전력 오버레이 ON이면 하단에 실시간 W,
크기 active_w 비례 120–300W 스케일), Switch=원형(기본은 nic만 표시),
IntraLink 두께 = bandwidth 로그 스케일.

인터랙션: hover 툴팁(디바이스: id·hw·mem·W·상태·예약 테넌트/배포 | 링크: 종류·대역폭·지연),
클릭→하단 상세 바 고정, **하이라이트 모드**(추천 카드에서 진입 시 선택 디바이스 강조 +
나머지 30% dim + TP 그룹 내부 링크 강조), d3-force 좌표 localStorage 저장,
드래그 고정(더블클릭 해제)·줌·팬. 빈 상태: registry 없으면 안내 카드 + example 복사 명령.
그래프 로드 실패 시 테이블 폴백.

### 5.3 화면 ② 서빙 요청 (request.html + request_form.js)

좌측 폼(4칸) / 우측 결과(8칸).

폼: 모델(드롭다운=GET /api/models), 정밀도(fp16/bf16), SLO 3필드(placeholder 권장값,
>0 검증, 미입력=제약 없음), 규모(req/s 필수 + 프리셋 chat/summarize/agentic + duration),
고급 접힘(power_cap_w, exclude_hw 멀티셀렉트, force_backend, num_req_eval).
서버 422는 필드 인라인 에러로.

결과 영역 3상태:
1. 제출 전: 안내 + 최근 요청 이력
2. 진행 중: 단계 스테퍼 ①스냅샷→②워크로드 합성→③Stage-1→④Stage-2(n/K)→⑤완료 (SSE)
3. 완료: **BEST 카드** — 좌: 전력 W 대형 숫자 + 신뢰도 배지(HIGH/MED/LOW=router
   confidence) + 평가 백엔드명, 중: 구성 요약 + TTFT/TPOT 대 SLO 게이지,
   우: [확정+배포] 주 버튼. 아래 **대안 Top-K 테이블**(구성·전력·예상 TTFT/TPOT·SLO
   여유%·신뢰도; 행 hover→그래프 하이라이트, 행 클릭→카드 교체).

infeasible: 적색 카드 — 병목 제약 + 완화 시나리오 + "dep-X 종료 시 가능" 힌트(링크).

확정 모달: 잠글 디바이스·전력·스냅샷 ver 요약 + auto_deploy 체크(기본 ON).
ConflictError 시 자동 재플래닝 1회 → 변경점 diff(추가/제거 디바이스 강조) 재확인.
성공 시 배포 상세 화면으로 자동 전환.

### 5.4 화면 ③ 배포 대시보드 (deployments.html + deploy_dashboard.js)

목록: 필터(상태·테넌트·모델, □종료 포함) + 테이블
(상태 배지 | ID/모델 | 테넌트 | 디바이스 | 전력 W | SLO | 가동 | 액션).

상세:
```
┌ 헤더: dep-id · 모델 · 상태배지   endpoint[복사]  [종료 ⏻] ┐
├ KPI 스트립 ×4: 처리량 | TTFT p95 | TPOT p95 | 전력        │
│  (각 카드에 SLO 목표/추천 예측값 병기, 색상 규약)           │
├ 차트 2열: 좌 시계열 2개(처리량+대기열 / TTFT·TPOT p95+SLO  │
│  목표 점선, 위반 적색 음영·DEGRADED 황색 음영)              │
│  우: KV캐시·대기열 게이지 + 디바이스별 전력 스파크라인       │
├ 하단 탭: [로그] [이벤트] [구성]                            │
└───────────────────────────────────────────────────────────┘
```

- 차트: 60분 기본(5s 해상도), 15분/6시간 전환.
- KPI 전력 카드: 추천 예측 대비 편차 ±10% 초과 시 황색.
- 로그 뷰어: 고정폭, follow 토글(위로 스크롤 시 자동 해제, [↓최신] 복귀), 레벨 필터,
  검색, 링버퍼 5,000줄, ERROR/WARN 색상 강조.
- 이벤트 탭: deployment_events 상태 전이 타임라인. 구성 탭: spec JSON 읽기 전용 뷰어.
- **종료 플로우**: [종료]→모달(in-flight 수, drain_timeout_s 입력 기본 120, 확인
  체크박스 후 활성화)→DRAINING 진행 바(in-flight 실시간 감소 + 경과/제한)→RELEASED
  토스트("자원 n개 해제 · 총 에너지 x kWh")→목록 복귀.
- FAILED 상세 진입 시 last_error + 마지막 로그 200줄 자동 펼침 + [동일 자원 재시도]/[자원 해제].

## 6. 마일스톤

### D0. 토폴로지 그래프 + registry v2 (§4.1)

**구현**
- `service/topology/schema.py`: v2 pydantic 스키마 + v1 자동 승격 로더(§3.1).
- `service/topology/graph.py`: TopologyGraph + `free_subgraph`/`affinity`/`to_planner_topology`.
- `service/topology/views.py`가 기존 `inventory/views.py`를 대체하되 출력 스키마 동일.
- `GET /api/cluster/graph` 엔드포인트(TopologyGraphOut 직렬화).

**테스트 (DoD)**
- [ ] v1 registry(`cluster_registry.example.yaml`) 로드 → 승격 결과가 기대 그래프와 일치.
- [ ] v2 스키마 위반(음수 전력, 중복 id, 존재하지 않는 링크 끝점) → ValidationError.
- [ ] `free_subgraph`: 예약 2개 주입 시 해당 정점·부속 간선 제외 확인.
- [ ] `affinity`: NVLink 4-clique > NVLink 2+2(PCIe 브리지) > PCIe-only 순서 검증(손계산 대조).
- [ ] `to_planner_topology` 골든 비교: 기존 views.py 출력과 dict 동일.
- [ ] **회귀**: `pytest -m unit planner/tests/ tests/ service/tests/` 전부 green.

### D1. 자동 탐지 + affinity 배치 세부화 (§5.1-5)

**구현**
- `discovery.py`: `nvidia-smi topo -m` 출력 파서(NV#/PIX/PXB/PHB/SYS 분류) +
  `lspci`/`/sys` 보조 → registry v2 초안 YAML. `--dry-run`은 stdout 출력만.
- 배치 세부화: planner 결과(hw×수량) → device id 매핑에 affinity 점수 최대화 +
  동점 시 idle 최소 호스트. `service/api/routes.py`의 `_default_planner` 후처리로 삽입.

**테스트 (DoD)**
- [ ] fixture topo 출력(NVLink 쌍 2개 + PCIe 4장) → 초안 YAML의 링크 종류 정확.
- [ ] TP2 요청 시 NVLink 쌍이 PCIe 쌍보다 우선 선택됨.
- [ ] TP2 × 2그룹 요청 시 두 그룹이 서로 다른 NVLink 쌍에 분리 배치.
- [ ] (hw, CI 제외) 실장비 1노드에서 discovery 실행 → 사람 검토로 YAML 확인.

### D2. 단일노드 배포 경로 — P0 핵심 (§4.2-4.3, §5.2)

**구현**
- `deploy/state.py`(전이 표), `deploy/store.py`(테이블 §3.2), `deploy/docker_driver.py`
  (docker SDK; endpoint tcp+TLS / ssh:// 지원, pull/run/stop/rm/logs/events/ping),
  `deploy/vllm_launcher.py`(번역 규칙 §4.2), `deploy/manager.py`(워커 스레드 상태 기계).
- confirm 훅: `auto_deploy=true`면 예약 직후 `manager.create`.
- terminate 경로: DRAINING(대기열 소진 폴링) → STOPPED → RELEASED(ledger.release).
- reconcile 주기 태스크 + 서비스 재시작 복구(§4.4).
- API: POST/GET `/api/deployments*`, terminate (§3.5).

**테스트 (DoD)**
- [ ] (unit) 상태 기계: 허용 전이 전수 + 불허 전이 예외. 모든 전이가 events에 기록.
- [ ] (unit) vllm_launcher: allocation fixture → 기대 ContainerSpec(gpu_indices,
      env, 포트, 이름 규약) 골든 비교. NVLink/PCIe에 따른 NCCL env 분기.
- [ ] (unit) manager + fake driver(메모리 내 가짜 dockerd): 정상 경로
      PENDING→…→READY→…→RELEASED 완주, 실패 주입 시 재시도 N회 후 FAILED + rollback 호출 확인.
- [ ] (unit) 재시작 복구: READY 1건·STARTING 1건 저장 후 manager 재기동 →
      READY 재-attach, STARTING은 rollback 후 PENDING.
- [ ] (unit) reconcile: 고아 컨테이너(가짜 목록 주입) 정리, 원장만 있는 배포 감지.
- [ ] (docker) 로컬 dockerd + 초소형 모델(예: Qwen2.5-0.5B)로
      confirm→READY→terminate→RELEASED E2E.
- [ ] (docker) 동시성: 두 클라이언트가 겹치는 자원 confirm+deploy 경쟁 → 이중 할당 0건.

### D3. 모니터링 + SSE (§4.3, §5.3)

**구현**
- `monitor/collector.py`: prometheus text 파서(외부 의존 없이) + 히스토그램 p50/p95.
- `monitor/power.py`: nvidia-smi(`--query-gpu=power.draw,utilization.gpu,...` 원격 실행
  또는 DCGM), furiosa-smi 스텁. device_id 매핑.
- `monitor/slo_checker.py`: 30s 윈도우, 3회 연속 위반 → DEGRADED 전이 트리거.
- `monitor/logs.py` + `monitor/store.py`(링버퍼 + 5분 집계 SQLite, 종료 시 Wh 요약).
- SSE: `/api/deployments/{id}/metrics`, `/logs` (기존 serve-requests SSE 패턴 재사용).

**테스트 (DoD)**
- [ ] (unit) 파서: 실제 vLLM /metrics 덤프 fixture → MetricSample 필드 정확
      (히스토그램 p95 손계산 대조).
- [ ] (unit) slo_checker: 위반 2회→ok 유지, 3회 연속→violated + 전이 콜백 1회만.
      회복 시 READY 복귀.
- [ ] (unit) 에너지 적분: 고정 전력 fixture 10분 → Wh 손계산 일치.
- [ ] (unit) fixture vLLM 서버(FastAPI로 /metrics·/health 흉내)로 collector 폴링 E2E.
- [ ] (unit) SSE 스트림: TestClient로 5초 간격 이벤트 수신·스키마 검증.
- [ ] 대시보드 수동 확인(fixture 서버): 차트·게이지·로그가 갱신됨.

### D4. UI 3화면 + Playwright (§5 = 설계서 §6)

**구현**
- §5.1–5.4 명세대로 templates/static 구현. d3·Chart.js vendoring.
- 색상 토큰 CSS 변수 단일 정의. SSE 클라이언트 공통화(common.js).

**테스트 (DoD)**
- [ ] Playwright (fixture vLLM + fake driver로 실기기 없이):
      제출→스테퍼 진행→추천 카드→확정 모달→배포 상세 자동 전환→KPI/차트 갱신
      →종료 모달→DRAINING 진행→RELEASED 토스트→목록 복귀.
- [ ] 클러스터 뷰: 그래프 렌더·상태 색·hover 툴팁·하이라이트 모드 스크린샷 비교.
- [ ] infeasible 응답 → 적색 카드 + 완화 시나리오 표시.
- [ ] SSE 강제 절단 → 재연결 표시·복구.
- [ ] 1024px 미만 뷰포트 → 단일 컬럼 강등·테이블 폴백.
- [ ] 서비스 프로세스 재시작 후 대시보드 재진입 → READY 배포 모니터링 재개.

### D5. 확장 — 멀티노드·NPU·사후 대조 (P1)

**구현**
- 멀티노드: Ray head/worker 컨테이너 오케스트레이션(head→worker 조인 순서),
  `VLLM_HOST_IP`·`distributed-executor-backend=ray`.
- NPU(RNGD): furiosa 런타임 이미지 경로 + device_path 매핑 + furiosa-smi 전력.
- 사후 대조 리포트: 배포 종료 시 (추천 예측 전력, 실측 Wh/시간) 쌍을 축적 →
  `scripts/power_calibration_report.py`가 오차 표 생성, `profiles/power/*.yaml`
  measured 테이블 환류 PR 초안 출력.

**테스트 (DoD)**
- [ ] (unit) 멀티노드 launcher: 2노드 TP2×PP2 allocation → head/worker ContainerSpec
      순서·env 골든 비교.
- [ ] (docker) 2노드(또는 단일 호스트 2 dockerd) Ray 조인 스모크.
- [ ] (hw, CI 제외) RNGD 1대 배포 E2E — SDK 확보 시.
- [ ] 사후 대조: fixture 이력 5건 → 오차 표·환류 YAML 초안 정확.
- [ ] **P1 완료 기준**: 이종(GPU+NPU) 데모 시나리오 완주 + 예측 전력 오차 리포트 자동 생성.

## 7. 테스트 전략 요약

| 계층 | 도구 | 시점 |
|---|---|---|
| 그래프 연산·상태 기계·파서·SLO 판정 | pytest unit, 손계산·골든 비교 | 매 커밋 |
| planner 뷰 호환 | 골든 비교(기존 views.py 출력) | 매 커밋 |
| 배포 경로 | fake driver(unit) → 로컬 dockerd(docker 마커) | D2 완료 시 |
| 모니터링 | fixture vLLM 서버 (실기기 불필요) | D3 완료 시 |
| UI E2E | Playwright + fixture 스택 | D4 완료 시 |
| 실측 | 실장비 discovery·RNGD (hw 마커, CI 제외) | 하드웨어 가용 시 |

**전역 회귀 가드**: 모든 마일스톤에서 `pytest -m unit planner/tests/ tests/ service/tests/`
green 유지. 기존 예제 spec 산출 해는 골든 파일로 고정(기존 규약).

## 8. 리스크 메모 (구현 시 주의)

- **TP>1 collective 과소평가**(A5000 검증 리포트에서 확인된 시뮬 경향): 링크 종류를
  Stage-1 제약·배치 세부화에 반영하고, D5 사후 대조로 보정 루프를 닫는다.
- **원격 dockerd 보안**: TLS 상호인증 또는 ssh:// 전용. API 응답에 노드 자격증명·
  endpoint 원문 비노출. 컨테이너는 최소 권한(privileged 금지, 필요 디바이스만 매핑).
- **drain 불완전(P0)**: 게이트웨이가 없어 신규 유입 차단 불가 — 대기열 소진 대기 +
  타임아웃 명시가 P0 한계임을 UI에 고지. P1에서 경량 프록시 검토.
- **vLLM 메트릭 이름 드리프트**: vLLM 버전별 metric 명칭 차이 → collector에 버전별
  매핑 테이블 + 미지 메트릭 경고(크래시 금지). 이미지 태그 고정이 1차 방어.
- **SQLite 동시성**: manager 워커·monitor·API가 같은 DB 공유 — 기존 ledger의
  커넥션 규약(짧은 트랜잭션, busy_timeout) 준수. 장기 폴링 루프에서 커넥션 보유 금지.
- **legacy 백엔드 경로 quirk**(CLAUDE.md): 시뮬 관련 경로는 반드시 어댑터 경유 —
  실행 계층에서 시뮬 CLI를 직접 조립하지 말 것.
- **OR-Tools**: `IntVar.Proto()` introspection 금지(세그폴트 이력). 배치 세부화는
  CP-SAT 밖(후처리 그래프 알고리즘)에서 수행 — 솔버 수정 불필요.
- **UI**: 색상·상태 문자열을 JS에 하드코딩하지 말 것 — CSS 변수·서버 응답의 state
  문자열만 사용. SSE 이벤트 스키마는 pydantic 모델의 `model_dump_json`과 1:1.
