# PLAN.md — 이종 클러스터 자원 관리 플래너 (아이디어 #1: 정확해 구조 최적화, MILP · Max-Flow)

> **목표**: 다양한 GPU/NPU가 섞인 이종 클러스터에서, 사용자가 지정한 요구사항(SLO — TTFT/TPOT, Throughput, Toks/Wh)을 만족하는 **최적 자원 할당**을 찾는 **오프라인 플래너**를 구현한다.
> **핵심 접근**: 클러스터를 그래프로 모델링하여 **MILP/Max-Flow 솔버로 후보 배치를 정확히(구조적으로) 계산**하고, 그 후보들을 **LLMServingSim을 예측 백엔드로 사용해 검증·재랭킹**한다.
> **최종 산출물**: 검증된 최적 `cluster_config/*.json`(+ 실행 인자). 이후 실제 클러스터(vLLM/llm-d/Dynamo)에 이식·검증.

---

## 0. 문서 정보

| 항목 | 내용 |
|---|---|
| 대상 저장소 | `ai-computing/LLMServingSim` (fork of `casys-kaist/LLMServingSim`) |
| 기준 릴리스 | v1.0.0 (2026-02-25) |
| 신규 모듈 위치 | `planner/` (신규 최상위 디렉토리) |
| 시뮬레이터 연동 | `main.py` 서브프로세스 호출 + `output/*.csv` 파싱 (비침습적) |
| 라이선스 | MIT (신규 코드도 동일 라이선스 유지) |

> ⚠️ **버전 주의**: v1.0.0 기준 CLI 플래그·`cluster_config` 필드명을 사용한다. 상위 저장소(`casys-kaist`)의 이후 버전에서는 일부 플래그명(`--request-routing-policy` 기본값, 파싱 키)이 바뀔 수 있으므로, 구현 착수 시 **빌드 대상 커밋의 `README.md`·`cluster_config/README.md`·`main.py` 인자표를 먼저 재확인**한다.

> 📌 **정합성 검증 반영 (2026-07-13, 현재 `ai-computing/LLMServingSim` main 기준)**: 아래 항목은 실제 코드(`main.py`, `cluster_config/README.md`, `inference_serving/config_builder.py`, `output/*.csv`)와 대조해 확정·수정한 내용이다.
> 1. **CLI 인자 전부 실재 확인** — `--cluster-config/--fp/--block-size/--dataset/--output/--num-req/--log-interval/--max-num-batched-tokens/--request-routing-policy` 존재. `--request-routing-policy` 기본값은 현재 `RR`(선택지 `RR/RAND/CUSTOM`).
> 2. **TP/PP 필드 없음** — 인스턴스에 `tp`/`pp` 필드는 **존재하지 않는다.** 병렬화는 `npu_num`(인스턴스 총 NPU 수)+`npu_group`(TP 차수)로 표현한다(§5.3·§5.4 수정 반영). PP는 config 노브로 노출돼 있지 않으므로 Stage1 변수에서 제외/보류한다.
> 3. **계층적 TP 패브릭** — 상위 필드 `tp_group_shape`(TP 그룹을 NVLink/PCIe/cross-socket 티어로 분해)와 리스트형 `link_bw`/`link_latency`가 존재한다. 이종 토폴로지 모델링에 활용한다(§4·§5.2·§5.4 반영).
> 4. **전력은 CSV에 없음** — `output/*.csv`에는 전력/에너지 컬럼이 없고 표준출력 로그로만 나온다 → `toks_per_wh` 견고화를 위해 §8 패치가 사실상 필수.
> 5. **`ITL`은 배열 문자열** — CSV의 `ITL` 컬럼은 per-token 간격 리스트(`"[...]"` 문자열)라 파싱 후 p99 집계가 필요(§5.5 반영). 프로파일 경로는 `llm_profile/perf_models/`(§4 오타 수정).

---

## 1. 설계 원칙 — "비침습적 래퍼(Wrapper) 우선"

플래너는 시뮬레이터 코드를 **직접 수정하지 않고** 상위에서 감싸는 것을 원칙으로 한다. 이유:

- 시뮬레이터 upstream(`casys-kaist`)과의 병합 충돌을 최소화한다.
- 시뮬레이터를 **블랙박스 평가 함수** `f(allocation) → {TTFT, TPOT, ITL-p99, throughput, energy}` 로만 취급하면 최적화 로직과 시뮬 로직이 깔끔히 분리된다.
- 시뮬레이터가 이미 제공하는 입력(`cluster_config/*.json`)과 출력(`output/*.csv`)이 인터페이스로 충분하다.

즉 플래너는 다음 3가지만 한다:
1. **탐색 변수 → `cluster_config` JSON + CLI 인자** 로 렌더링(생성).
2. `python main.py ...` **서브프로세스 실행**.
3. **결과 CSV 파싱 → 목적/제약 점수 계산** → 최적화 엔진에 반환.

---

## 2. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                         planner/ (신규 모듈)                           │
│                                                                        │
│  [입력]                                                                │
│   cluster_spec.yaml   ── 클러스터 토폴로지·디바이스 프로파일·요구사항   │
│        │                                                               │
│        ▼                                                               │
│  ┌───────────────────┐   그래프 G=(V,E)                                │
│  │ 1) TopologyGraph   │   V=디바이스, E=링크(대역폭/지연)               │
│  │    (graph_model.py)│                                                │
│  └─────────┬──────────┘                                                │
│            ▼                                                           │
│  ┌───────────────────────────────┐  Stage 1: 정확해 구조 최적화        │
│  │ 2) StructuralSolver           │  MILP / Max-Flow (OR-Tools CP-SAT)  │
│  │    (milp_solver.py)           │  → Top-K 후보 배치                  │
│  └─────────┬─────────────────────┘                                    │
│            │ Top-K allocations                                         │
│            ▼                                                           │
│  ┌───────────────────────────────┐                                    │
│  │ 3) ConfigRenderer             │  allocation → cluster_config.json   │
│  │    (config_renderer.py)       │              + main.py CLI 인자     │
│  └─────────┬─────────────────────┘                                    │
│            ▼                                                           │
│  ┌───────────────────────────────┐  Stage 2: 시뮬 검증·재랭킹          │
│  │ 4) SimEvaluator               │  subprocess: python main.py ...     │
│  │    (sim_evaluator.py)         │  → output/*.csv 파싱                │
│  └─────────┬─────────────────────┘  → {TTFT,TPOT,ITL-p99,thpt,energy} │
│            ▼                                                           │
│  ┌───────────────────────────────┐                                    │
│  │ 5) Objective / Constraints    │  SLO 제약 검사 + 다목적 점수        │
│  │    (objective.py)             │  (ε-제약으로 파레토 스윕)           │
│  └─────────┬─────────────────────┘                                    │
│            ▼                                                           │
│  [출력] 최적 cluster_config.json + 성적표 리포트(csv/md)               │
└──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
        (이후) 실제 클러스터 이식: vLLM / llm-d / Dynamo
```

---

## 3. 디렉토리 & 파일 구조 (신규)

```
LLMServingSim/
├── planner/                       # ← 신규 최상위 모듈
│   ├── __init__.py
│   ├── cli.py                     # 진입점: `python -m planner.cli --spec ...`
│   ├── spec_schema.py             # 입력 스펙(YAML) 파싱·검증 (pydantic)
│   ├── graph_model.py             # 클러스터→그래프 변환 (networkx)
│   ├── milp_solver.py             # Stage 1: MILP/Max-Flow (OR-Tools CP-SAT)
│   ├── config_renderer.py         # allocation → cluster_config JSON + CLI 인자
│   ├── sim_evaluator.py           # Stage 2: main.py 서브프로세스 + CSV 파싱
│   ├── objective.py               # SLO 제약 + 다목적(Throughput, Toks/Wh) 점수
│   ├── search_orchestrator.py     # 전체 파이프라인 조율 (Stage1→render→Stage2→rank)
│   ├── report.py                  # 결과 리포트(csv/md) + 파레토 표
│   └── utils.py                   # 로깅·캐시·해시 유틸
│
├── planner/specs/                 # 예제 입력 스펙
│   ├── example_hetero_8gpu.yaml
│   └── README.md
│
├── planner/tests/                 # 단위·통합 테스트 (pytest)
│   ├── test_graph_model.py
│   ├── test_milp_solver.py
│   ├── test_config_renderer.py
│   ├── test_sim_evaluator_mock.py # 시뮬 없이 mock CSV로 파싱 검증
│   └── test_end_to_end_small.py   # 소규모 실제 시뮬 1~2회 (CI에선 optional)
│
├── planner/PLAN.md                # 본 문서
├── planner/README.md              # 사용법
└── requirements-planner.txt       # ortools, networkx, pydantic, pandas, pyyaml, pymoo(optional)
```

> `main.py`, `inference_serving/`, `cluster_config/`, `llm_profile/` 등 **기존 파일은 수정하지 않는다.** (예외는 §8 "선택적 시뮬레이터 확장" 참조)

---

## 4. 입력 스펙 정의 (`planner/specs/*.yaml`)

사용자가 작성하는 단일 진입 스펙. 플래너 내부에서 이 스펙을 그래프·탐색공간·목적함수로 변환한다.

```yaml
# example_hetero_8gpu.yaml
model:
  name: "meta-llama/Llama-3.1-70B"     # main.py 지원 모델
  fp: 16

workload:
  dataset: "dataset/sharegpt_req100_rate10_llama.jsonl"
  num_req: 100

# 1) 클러스터 내부 연결망 구조 및 속도
topology:
  nodes:
    - id: node0
      devices:
        - {name: H100,      count: 2, mem_gb: 80}
        - {name: A6000,     count: 4, mem_gb: 48}
    - id: node1
      devices:
        - {name: TPU-v6e-1, count: 2}
  links:                                # 엣지: 대역폭/지연
    - {src: node0, dst: node1, bandwidth: "200Gbps", latency: "0.0005ms"}
  intra_node_bandwidth: "600GBps"       # NVLink 등 노드 내부
  # 선택: 계층적 TP 패브릭. 렌더 시 cluster_config의 tp_group_shape + 리스트형 link_bw/link_latency로 직렬화.
  # 예) tp_group_shape=[2,2] → NVLink 쌍(dim0) + 노드 내 PCIe(dim1). 티어별 대역폭/지연을 innermost-first 리스트로.
  tp_group_shape: [2, 2]                # 생략 시 단일 flat FullyConnected TP (legacy)

# 2) 디바이스 프로파일 (llm_profile 결과 재사용; 경로만 참조)
profiles:
  perf_root: "llm_profile/perf_models"  # 프로파일러가 생성한 CSV 루트 (하드웨어별 하위 디렉토리: A100/A40/A6000/H100/RNGD/TPU-v6e-1 등)

# 3) 사용자 요구사항 (SLO / 성능 / 에너지)
requirements:
  ttft_ms:  {constraint: "<=", value: 500}    # hard 제약
  tpot_ms:  {constraint: "<=", value: 50}     # hard 제약
  itl_p99_ms: {constraint: "<=", value: 80}   # hard 제약
  objectives:                                 # 다목적 (파레토)
    - {metric: throughput,   direction: max, weight: 0.6}
    - {metric: toks_per_wh,  direction: max, weight: 0.4}

# 탐색 공간 제어
search_space:
  pd_disaggregation: true                # P/D 분리 허용
  tp_choices:  [1, 2, 4]                # → npu_group. tp가 npu_num의 약수여야 함
  pp_choices:  [1]                       # PP는 config 노브 미지원 → [1] 고정 (§5.3 참조)
  xpyd_prefill_range: [1, 4]             # prefill 인스턴스 수 범위
  xpyd_decode_range:  [1, 6]
  batch_tokens_choices: [1024, 2048, 4096]

solver:
  top_k: 8                               # Stage1이 넘길 후보 수
  time_limit_sec: 120                    # MILP 시간 제한
  pareto_epsilon_steps: 5                # ε-제약 스윕 단계
```

---

## 5. 컴포넌트별 상세 구현 계획

### 5.1 `spec_schema.py` — 입력 검증
- **역할**: 위 YAML을 pydantic 모델로 파싱·검증. 필수 필드 누락, 잘못된 디바이스명(프로파일 미존재) 등을 조기 차단.
- **의존**: `pydantic`, `pyyaml`.
- **검증 항목**: (1) `model.name`이 지원 모델인지, (2) 각 디바이스 프로파일 CSV가 `profiles.perf_root` 아래 존재하는지, (3) 제약/목적 메트릭 이름이 유효한지.

### 5.2 `graph_model.py` — 클러스터를 그래프로
- **역할**: 토폴로지 스펙 → `networkx.DiGraph`. 노드 속성 = 디바이스 타입/개수/메모리; 엣지 속성 = 대역폭/지연.
- **출력**: Helix식 "capacity graph". 이 그래프가 MILP 솔버의 입력이자, 이후 `cluster_config`의 `link_bw`/`link_latency`(스칼라 또는 티어별 리스트)와 `tp_group_shape`로 직렬화된다.
- **핵심 함수**:
  - `build_graph(spec) -> nx.DiGraph`
  - `device_inventory(graph) -> list[Device]`  (솔버용 평탄화된 디바이스 목록)
  - `to_cluster_config_topology(graph) -> dict`  (config 직렬화 보조)

### 5.3 `milp_solver.py` — Stage 1: 정확해 구조 최적화 ★ 핵심
- **역할**: 배치·병렬화·xPyD를 **정수계획법(MILP)**으로 최적화하여 상위 K개 후보를 산출.
- **솔버**: **Google OR-Tools CP-SAT** (오픈소스·강력). 대안: PuLP+CBC, Pyomo+HiGHS.
- **결정 변수**:
  - `x[d, i] ∈ {0,1}`: 디바이스 `d`를 인스턴스 `i`에 할당.
  - `role[i] ∈ {prefill, decode}`: 인스턴스 역할 (P/D 분리).
  - `tp[i] ∈ tp_choices`: 인스턴스별 TP 차수. **config에는 `npu_group`으로 직렬화**되며, `npu_num`은 인스턴스에 할당된 총 디바이스 수, `npu_num/tp[i]`가 복제(DP) 개수가 된다.
  - **`pp` 제외/보류**: 현재 `cluster_config`에는 파이프라인 병렬을 지정하는 노브가 없다(`npu_num`/`npu_group`만 존재, `npu_group ≤ npu_num` 강제). PP 지원은 코드에서 확인된 뒤 별도로 도입한다. §4 스펙의 `pp_choices`는 [1]로 고정하거나 제거한다.
  - `n_prefill, n_decode`: xPyD 비율.
- **제약**:
  - 각 디바이스는 최대 1개 인스턴스에 할당.
  - 병렬 차수 정합성: `tp[i]`는 `npu_num[i]`(할당 디바이스 수)의 약수여야 하며 `tp[i] ≤ npu_num[i]` (config_builder의 `npu_group ≤ npu_num` 검증과 일치).
  - 메모리 용량: 모델 가중치 + KV 캐시 예상치 ≤ 인스턴스 총 메모리.
  - 링크 대역폭: (선형화된) KV 전송량 ≤ 링크 용량 (Max-Flow 항).
  - SLO의 **선형 프록시** 제약(예: 파이프라인 단계 지연 상한). 정밀 SLO는 Stage 2에서 확정.
- **목적함수 (Stage1 근사)**: 선형화 가능한 프록시 — 예상 처리량 최대화 − λ·예상 전력. 다목적은 **ε-제약 스윕**(`pareto_epsilon_steps`)으로 여러 해를 생성.
- **출력**: 상위 `top_k`개의 `Allocation` 객체 리스트 (Solution Pool 활용).
- **핵심 함수**:
  - `solve(graph, spec) -> list[Allocation]`
  - `_add_placement_constraints(model, ...)`, `_add_memory_constraints(...)`, `_add_flow_constraints(...)`

> **참고 구현**: Helix(ASPLOS'25)의 max-flow+MILP 배치 공식, Mélange의 비용-aware 디바이스 선택 ILP. 레이어 동질성을 이용해 변수 수를 노드·링크에 선형으로 유지.

### 5.4 `config_renderer.py` — allocation → 시뮬 입력
- **역할**: `Allocation` 객체를 LLMServingSim의 `cluster_config/*.json`과 `main.py` CLI 인자로 변환.
- **매핑**:
  | Allocation 필드 | cluster_config / CLI |
  |---|---|
  | 노드·디바이스 배치 | `nodes[].instances[].hardware`, `num_nodes`, `nodes[].num_instances` |
  | `tp[i]` (TP 차수) | `instances[].npu_group` |
  | 할당 디바이스 수 | `instances[].npu_num` (복제 개수 = `npu_num/npu_group`) |
  | `role[i]` (P/D) | `instances[].pd_type` = `"prefill"`/`"decode"`/`null` |
  | 계층적 TP 패브릭 | 상위 `tp_group_shape` + 리스트형 `link_bw`/`link_latency` |
  | 링크 대역폭/지연 | `link_bw`(GB/s), `link_latency`(ns) — 스칼라 또는 innermost-first 리스트 |
  | 배치 토큰 | `--max-num-batched-tokens` |
  | 라우팅 | `--request-routing-policy` (기본 `RR`) |
- **주의**: 정확한 JSON 스키마는 `cluster_config/README.md`와 `cluster_config/*.json` 예제를 **템플릿으로 로드**하여 필드를 채우는 방식으로 구현(하드코딩 금지). 이렇게 하면 스키마 변경에 강건하다.
- **핵심 함수**: `render(allocation, spec) -> (config_path, cli_args)`

### 5.5 `sim_evaluator.py` — Stage 2: 시뮬 실행·파싱
- **역할**: 렌더된 config로 `main.py`를 서브프로세스 실행하고 결과 CSV를 파싱.
- **실행 예시** (README 기준):
  ```bash
  python main.py \
    --cluster-config 'cluster_config/<generated>.json' \
    --fp 16 --block-size 16 \
    --dataset '<dataset>' \
    --output 'output/<run_id>.csv' \
    --num-req 100 --log-interval 1.0
  ```
- **파싱**: `output/<run_id>.csv`의 per-request 지표 → p99·평균 집계. 실제 CSV 헤더:
  ```
  instance id,request id,model,input,output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL
  ```
  - `TTFT`, `TPOT`는 스칼라(ns). `ITL`은 **per-token 간격 리스트가 담긴 문자열**(`"[198130302, 197318925, ...]"`)이므로 `ast.literal_eval`/`json.loads`로 파싱 후 p99 집계한다.
  - **전력/에너지 컬럼은 CSV에 없다** → 표준출력 로그(전력 요약)를 캡처·파싱하여 `toks_per_wh` 계산. 파싱이 취약하므로 §8 패치(전력→CSV 컬럼)를 우선 검토한다.
  - 시간 단위는 ns(1 GHz). ms 제약(`ttft_ms` 등)과 비교 시 단위 변환 필요.
- **견고성**: 타임아웃, 비정상 종료(메모리 초과 등) 처리 → 해당 후보는 "infeasible"로 표시하고 계속 진행.
- **캐시**: `(config 해시) → 결과`를 디스크 캐시하여 동일 후보 재평가를 방지.
- **핵심 함수**: `evaluate(config_path, cli_args) -> Metrics | Infeasible`

### 5.6 `objective.py` — 제약·다목적 점수
- **역할**: 파싱된 `Metrics`를 사용자 요구사항에 대입.
  - **Hard 제약**: `TTFT ≤`, `TPOT ≤`, `ITL-p99 ≤`, 메모리 → 하나라도 위반 시 후보 탈락.
  - **목적**: `throughput`, `toks_per_wh` → 파레토 지배 판정 + 가중합 스칼라 점수.
- **핵심 함수**: `check_constraints(metrics, req) -> bool`, `score(metrics, req) -> float`, `pareto_front(candidates) -> list`

### 5.7 `search_orchestrator.py` — 파이프라인 조율
- **흐름**: `spec → graph → MILP(top_k) → [render → sim → objective] × K → 파레토 랭킹 → 최적 config 출력`.
- **병렬화**: K개 후보의 Stage 2 시뮬을 프로세스 풀로 병렬 실행(디바이스 자원과 무관, CPU 시뮬이므로 코어 수만큼).
- **핵심 함수**: `run(spec_path) -> PlannerResult`

### 5.8 `report.py` — 결과 리포트
- **출력물**:
  - `planner_out/best_cluster_config.json` — 최종 추천 배치.
  - `planner_out/pareto.csv` — 후보별 (배치, TTFT, TPOT, ITL-p99, throughput, Toks/Wh, 통과여부).
  - `planner_out/report.md` — 사람이 읽는 요약 + 파레토 표.

### 5.9 `cli.py` — 진입점
```bash
python -m planner.cli \
    --spec planner/specs/example_hetero_8gpu.yaml \
    --out-dir planner_out/ \
    --jobs 8
```

---

## 6. 구현 단계별 마일스톤

| 단계 | 내용 | 산출물 | 검증 기준 |
|---|---|---|---|
| **M0. 스캐폴딩** | `planner/` 구조·`requirements-planner.txt`·CLI 뼈대 | import 가능한 빈 모듈 | `python -m planner.cli --help` 동작 |
| **M1. 래퍼 검증** | `config_renderer` + `sim_evaluator`로 **손수 만든 배치 1개**를 시뮬 돌려 CSV 파싱 | 단일 배치 평가 성공 | 기존 `run.sh` 예제와 동일 결과 재현 |
| **M2. 목적/제약** | `objective.py` — SLO 제약·Toks/Wh 계산 | 후보 점수화 | mock CSV로 단위테스트 통과 |
| **M3. MILP Stage1** | `graph_model` + `milp_solver` — Top-K 후보 생성 | 후보 리스트 | 소규모(≤8 디바이스)에서 수동 최적해와 일치 |
| **M4. E2E 파이프라인** | `search_orchestrator` — Stage1→Stage2→랭킹 | `best_cluster_config.json` | 예제 스펙에서 SLO 만족 배치 산출 |
| **M5. 다목적·병렬** | ε-제약 파레토 스윕 + 병렬 시뮬 | `pareto.csv` | 파레토 프론트 ≥3점, 병렬 speedup 확인 |
| **M6. 리포트·문서** | `report.py` + `planner/README.md` + 예제 | md 리포트 | 재현 가능한 예제 1건 |
| **M7. (선택) 실측 이식** | 최적 config → vLLM/llm-d 인자 변환 초안 | 이식 스크립트 | 실제 소규모 클러스터 1건 대조 |

---

## 7. 테스트 전략

- **단위 테스트** (`pytest`, 시뮬 불필요):
  - `test_graph_model`: 토폴로지 스펙 → 그래프 정합성(노드·엣지 수, 속성).
  - `test_milp_solver`: 알려진 소규모 인스턴스에서 최적해·제약 준수 검증.
  - `test_config_renderer`: allocation → JSON이 `cluster_config` 스키마에 부합(예제 템플릿과 diff).
  - `test_sim_evaluator_mock`: 사전 준비한 mock CSV로 파싱·집계(TTFT/TPOT/ITL-p99) 정확성.
  - `test_objective`: 제약 위반 탈락·파레토 지배 판정.
- **통합 테스트** (시뮬 필요, CI에선 optional 태그):
  - `test_end_to_end_small`: 단일 노드·소수 요청으로 M4 파이프라인 1회 완주.
- **회귀**: 예제 스펙의 산출 config를 골든 파일로 저장, 변경 시 비교.

---

## 8. (선택적) 시뮬레이터 확장이 필요한 경우

기본은 비침습이지만, 다음이 필요하면 최소 수정을 검토한다(별도 PR·upstream 반영 고려):
- **전력/에너지 출력의 기계 파싱 편의 (✅ 정합성 검증에서 필요성 확인됨)**: `output/*.csv`에는 전력/에너지 컬럼이 없고 현재 표준출력 로그로만 나온다. `toks_per_wh` 목적함수를 견고하게 계산하려면 이 값을 `--output` CSV 컬럼(또는 별도 요약 CSV)으로 추가하는 소규모 패치가 사실상 필수다. → `inference_serving/`의 출력 경로에 패치. 패치를 미룰 경우 표준출력 파싱으로 임시 대응하되 취약성을 감수한다.
- **배치 스크립트 편의**: 여러 config를 순차 실행하는 헬퍼는 `planner/` 안에서 해결(시뮬 코드 수정 불필요).

> 수정 시 `README.md`의 "Adding a New Model & Hardware"에서 안내하는 `memory_model.py`·`trace_generator.py` 규약을 준수한다.

---

## 9. 의존성 (`requirements-planner.txt`)

```
ortools>=9.10          # CP-SAT MILP 솔버
networkx>=3.2          # 토폴로지 그래프
pydantic>=2.6          # 스펙 검증
pyyaml>=6.0            # 스펙 파싱
pandas>=2.2            # CSV 파싱·집계
# 선택
pymoo>=0.6            # (향후 #3 진화탐색 결합 시)
```

> 시뮬레이터 본체 의존성과 분리하여, 플래너만 별도 설치 가능하게 한다.

---

## 10. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| `cluster_config` 스키마가 버전마다 변경 | 렌더러 파손 | 예제 JSON을 **템플릿으로 로드**해 필드 채움(하드코딩 금지) + 골든 테스트 |
| `npu_num`/`npu_group`의 TP·DP·PP 해석이 문서(CLAUDE.md)와 `config_builder.py` 코드에서 상충 소지 | 렌더 결과 오배치 | M1에서 **손수 만든 배치를 실제 실행**해 토폴로지 직렬화 결과를 대조 검증(설계에 내장); PP는 확인 전까지 미사용 |
| MILP 선형 프록시가 동적 효과를 못 잡음 | 후보 순위 부정확 | Stage 2 시뮬 검증으로 최종 재랭킹(설계에 내장) |
| 시뮬 1회 ~수 분 → K개 평가 비용 | 탐색 느림 | Top-K를 작게(≤8), 결과 캐시, 병렬 실행 |
| NPU/TPU 프로파일 희소 | 정확도 저하 | 프로파일 존재 여부를 `spec_schema`에서 사전 검증, 없으면 명확한 에러 |
| ns3 백엔드 미성숙 | 네트워크 정밀도 | 기본은 `analytical` 백엔드, ns3는 옵션 |
| upstream 병합 충돌 | 유지보수 부담 | 신규 코드는 `planner/`에 격리, 본체 수정 최소화 |

---

## 11. 완료 정의 (Definition of Done)

- [ ] `python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml` 실행 시, SLO를 만족하는 `best_cluster_config.json`과 `pareto.csv`·`report.md`가 생성된다.
- [ ] 산출된 `best_cluster_config.json`을 `main.py`에 그대로 넣어 재현 실행이 성공한다.
- [ ] 단위 테스트(시뮬 불필요) 전부 통과, 통합 테스트 1건 통과.
- [ ] `planner/README.md`에 설치·실행·스펙 작성법이 문서화된다.
- [ ] 본체(`main.py`, `inference_serving/` 등) 수정 없이 동작한다(또는 §8 패치가 명시적 PR로 분리된다).

---

## 12. 향후 확장 (본 계획 범위 밖)

- **#4 대리모델 결합**: Stage 2 시뮬 결과를 학습 데이터로 ML surrogate를 훈련해 후보 평가를 가속.
- **#2/#3 정제 단계**: MILP Top-K 위에서 베이지안(Ax/BoTorch) 또는 NSGA-II(pymoo)로 연속 노브 미세조정.
- **실측 피드백 루프**: 실제 클러스터 telemetry로 프로파일·목적함수를 재보정(sim-to-real).
