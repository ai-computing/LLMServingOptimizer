# PLAN_webapp_dse_detail.md — DSE 웹 도구 단계별 세부 작업 계획

> [PLAN_webapp_dse.md](./PLAN_webapp_dse.md)의 각 Phase를 **실행 가능한 단위 작업**으로 분해. 기존 `webapp/` 코드베이스를 분석한 결과 상당 부분이 이미 구현되어 있어 **재사용 vs 신규 개발**을 명시.

---

## 0. 사전 분석 — 기존 `webapp/`이 이미 제공하는 것

PLAN_webapp_dse.md가 작성된 이후 `webapp/`에 많은 기능이 추가됐습니다. 새 `webapp_dse/`는 이 위에 얹는 게 효율적입니다.

### 이미 구현된 컴포넌트 (재사용 대상)

| 영역 | 기존 위치 | DSE에서 재사용 가능 여부 |
|---|---|---|
| **하드웨어 카탈로그** | `webapp/hardware_catalog.py` (`build_catalog`, `list_hardware`, `get_tp_options`, `_tp_dir_is_complete`) | 그대로 재사용 |
| **Cluster config I/O** | `webapp/cluster_io.py` (`list_configs`, `load_config`, `save_config`, `delete_config`) | 그대로 재사용 |
| **Cluster JSON builder** | `webapp/cluster_builder.py` (`build_cluster_json`, `InstanceSpec`, `ConfigSpec`, `power_template` 지원) | 그대로 재사용 |
| **Config enumerator** | `webapp/enumerate.py` (`enumerate_configs`, `_topology_valid`, P/D 필터, heterogeneous filter) | **확장** 필요 (현재 multi-group은 고정 group 수; DSE는 resource pool에서 group 수 자체를 탐색) |
| **Sweep runner** | `webapp/runner.py` (`run_sweep`, `_run_one_config`, `MAX_CONCURRENT` semaphore, PID_TAG isolation, cleanup) | 그대로 재사용 |
| **Log/metric parser** | `webapp/parser.py` (TTFT/TPOT/ITL/throughput/power 추출) | 그대로 재사용 |
| **Plotly 차트 생성** | `webapp/plots.py` (bar, pareto frontier, CDF, line, `assign_config_colors`) | 그대로 재사용 + Top-N 차트 추가 |
| **결과 페이지** | `webapp/templates/results.html` | 일부 재사용 (Config Legend, Pareto, breakdown) |
| **SSE 진행률 / WebSocket** | `webapp/runner.py` `_broadcast`, `subscribe_events`, `app.py` `/api/sweeps/{id}/events` (SSE) | 그대로 재사용 |
| **Power 시뮬레이션** | `webapp/parser.py` Wh 추출, `cluster_builder.py` `power_template` 주입 | 그대로 재사용 |

### 신규 개발이 필요한 영역

| 영역 | 이유 |
|---|---|
| **Resource pool → combination generator** | 기존 enumerate는 `instance_groups` 고정 입력. DSE는 hardware count 자체가 탐색 차원 |
| **SLO 필터링 + Pareto** | Pareto frontier는 plots.py에 있지만 **시각화용**. DSE는 ranking에 활용 |
| **Weighted scoring + Top-N** | 신규 |
| **가중치 재조정 즉시 재랭킹 (재시뮬 없이)** | 신규. 결과 캐싱 + 클라이언트사이드 재계산 |
| **Spec YAML/JSON 입력 형식** | 신규. 기존은 scenario form (instance_groups 직접 입력) |
| **Job DB (SQLite + job 상태 관리)** | 신규. 기존은 sweep_dir/status.json 기반 (job DB로 더 풍부한 메타 관리 필요) |
| **결과 캐시 (spec hash → 결과 dir)** | 신규 |
| **DSE 입력 페이지 UI** | 신규. resource pool + objectives + weights 입력 |

### 아키텍처 결정 사항 — **(B-1) 채택**

PLAN_webapp_dse.md의 "webapp_dse/ 별도 디렉토리" 제안은 두 가지로 해석 가능했고, 통합 방식 (B) 안에서도 다시 navigation 패턴 3가지가 있었음. 다음과 같이 결정:

**최종 채택: (B-1) 통합 + 헤더 nav bar**
- 기존 `webapp/`에 DSE 라우트/페이지/core를 추가
- `webapp/templates/base.html`에 **상단 nav bar** 1개 추가 — 모든 페이지에 표시
- nav 항목: `[Sweep] [DSE]` (현재 활성 페이지는 강조)
- 신규 패키지: `webapp/dse/` 서브디렉토리 (core / server routes / templates / static)

**결정 근거**:
- 인증/세션/카탈로그 API 중복 방지 (단일 FastAPI 앱)
- 같은 hardware 카탈로그·plot 인프라 공유
- 사용자가 두 워크플로우 (Sweep ↔ DSE) 사이를 **한 클릭으로** 이동 가능
- 새 진입점 발견 가능성 ↑ (URL을 모르는 사용자도 nav를 보고 DSE 인지)

**기존 페이지 영향**:
- 콘텐츠 / 동작 / 기존 단축키: **변경 없음**
- 시각적 변화: 모든 페이지 상단에 작은 nav bar (높이 ~40px)가 추가됨
- 기존 page-header는 그대로 — nav bar는 그 위에 위치
- 모바일에서는 햄버거 메뉴로 collapse (Phase 4.0에서 처리)

**탈락한 옵션**:
- (A) 완전 분리: 신규 FastAPI app + DB 인프라 ~1주일 추가, 카탈로그/플롯 중복 발생
- (B-2) URL only: 발견 가능성 낮음, DSE는 핵심 가치인데 nav 없으면 사용자가 못 찾음
- (B-3) 푸터/사이드바: 다른 페이지와 일관성 떨어짐

본 상세 계획은 (B-1) 전제로 작성됨. (A)로 회귀하려면 §3, §4 전체 재작업 필요.

---

## Phase 0 — 사전 조사 및 분석

### 0.1 기존 webapp 구조 문서화

**목적**: DSE 개발자가 webapp/이 무엇을 어떻게 제공하는지 빠르게 파악.

**산출**: `docs/dse/00_existing_webapp_notes.md`

**작업 항목**:
- [x] `webapp/` 모듈별 1줄 요약표
  - app.py — FastAPI 진입점, REST + SSE
  - cluster_builder.py — InstanceSpec → cluster JSON
  - cluster_io.py — `cluster_config/*.json` CRUD
  - enumerate.py — scenario → ConfigSpec 리스트
  - runner.py — Semaphore-bound subprocess sweep
  - parser.py — log/CSV → metrics dict
  - plots.py — Plotly JSON figure
  - hardware_catalog.py — perf_models/ 디렉토리 스캔
- [x] 데이터 흐름도 (scenario → enumerate → cluster_json → main.py → log → parse → metrics → plots)
- [x] SSE 이벤트 스키마 (run_one_config 진행 단계별로 broadcast하는 dict 모양)
- [x] sweep_dir 디렉토리 레이아웃 (`configs/`, `runs/`, `status.json`, `scenario.json`, `metrics.json`)
- [x] 재사용 가능한 함수 목록 + signature
- [x] **회피해야 할 부분**: ASTRA-Sim heterogeneous deadlock 패턴, PID_TAG 컨벤션, MAX_CONCURRENT 동시성 제약

**예상 공수**: 0.5일 (코드는 이미 친숙하므로 빠름) — ✅ **완료** (`docs/dse/00_existing_webapp_notes.md`)

### 0.2 cluster_config 스키마 역공학

**산출**: `docs/dse/01_cluster_config_schema.md`, `docs/dse/cluster_config_schema.json` (JSON Schema 형식)

**작업 항목**:
- [x] `inference_serving/config_builder.py:build_cluster_config` 검증 로직 추출
  - required_keys 목록 (`["model_name", "hardware", "npu_mem", "npu_num", "npu_group", "pd_type"]`)
  - power block 필드 매트릭스 (`npu_keys`, `cpu_keys`, `dram_keys`, ...)
  - placement / pim_config / cxl_mem 선택 필드
- [x] 모든 `cluster_config/*.json` 예시를 표로 정리 (어떤 필드 조합이 실전에 쓰이는지)
- [x] JSON Schema 자동 생성 (`jsonschema.Draft7Validator` 검증 가능 형식)
- [x] 의존성 표: `power_modeling=True` ↔ 모든 노드에 `power` 블록 필수 등
- [x] **알려진 제약**:
  - `npu_group <= npu_num`, `npu_num % npu_group == 0`
  - prefill instance는 ASTRA-Sim에서 `npu_num *= 2` 됨 (sender NPU)
  - 모든 인스턴스가 같은 system.json을 공유 → heterogeneous P/D 정확도 한계

**예상 공수**: 1일 — ✅ **완료** (`docs/dse/01_cluster_config_schema.md` + `cluster_config_schema.json`, 13/13 예시 통과)

### 0.3 main.py I/O 명세

**산출**: `docs/dse/02_main_io_spec.md`

**작업 항목**:
- [x] CLI flag 카테고리 (input/output/feature/scheduling/debug)
- [x] `--fp`, `--block-size`, `--num-req`, `--log-interval`, `--enable-attn-offloading`, `--enable-prefix-caching` 등 의미
- [x] CSV 출력 컬럼: `instance_id, request_id, model, input, output, arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL` (ns 단위, ITL은 JSON list)
- [x] stdout 메트릭 표 (정규식 패턴 포함 — `parser.py:PATTERNS` 참조)
- [x] power 출력 포맷 + 단위 변환 (kJ → Wh)
- [x] 종료 코드 / 에러 패턴

**예상 공수**: 0.5일 — ✅ **완료** (`docs/dse/02_main_io_spec.md`)

### 0.4 하드웨어 / 모델 카탈로그

**산출**: `docs/dse/03_catalog.yaml`

**작업 항목**:
- [x] `llm_profile/perf_models/` 디렉토리 스캔 → (hardware, model, tp) 매트릭스
- [x] 각 하드웨어별 메타데이터:
  - `mem_size_gb` (HBM/GDDR 용량)
  - `mem_bw_gbs` (HBM bandwidth)
  - `peak_flops_fp16_tflops`
  - `tdp_w` (max power)
  - `idle_power_w` (실측 or vendor)
  - `unit_cost_usd` (DSE의 cost 계산용, 선택)
- [x] 모델별 메타데이터:
  - `params_b` (8B, 70B 등)
  - `weight_size_fp16_gb` (메모리 사전 필터링용)
  - `num_layers`, `hidden_size`
- [ ] `_tp_dir_is_complete` 확장: 추가로 power profile 존재 여부 컬럼 (Phase 1에서 필요 시 추가 — 현재 catalog로 충분)

**예상 공수**: 0.5일 — ✅ **완료** (`docs/dse/03_catalog.yaml`, 5 hw / 4 model, build_catalog()와 0 mismatch)

### Phase 0 합계: **2.5일** — ✅ **완료** (4 문서, schema 13/13 통과, catalog 0 mismatch)

---

## Phase 1 — 백엔드 코어 엔진

> 모든 코드는 `webapp/dse/core/` (또는 `webapp_dse/core/`) 하위. 웹 의존 없이 CLI로 검증 가능.

### 1.1 Combination Generator (`webapp/dse/core/generator.py`)

**목적**: `ResourcePool` + `Constraints` → 시뮬레이션 가능한 `CandidateConfig` 리스트.

**입력 스키마** (Pydantic):
```python
class HwAllocation(BaseModel):
    hw: str            # "H100", "RNGD"
    min: int = 0
    max: int

class ResourcePool(BaseModel):
    items: list[HwAllocation]
    total_max_npus: int | None = None  # 전체 NPU 합 상한

class ModelSpec(BaseModel):
    name: str           # HuggingFace ID
    fp: int = 16        # 8 / 16 / 32

class FeatureFlags(BaseModel):
    allow_pd_disagg: bool = True
    allow_pim: bool = False
    allow_cxl: bool = False
    prefix_caching: bool = False
    attn_offloading: bool = False
    sub_batch_interleaving: bool = False

class SearchConfig(BaseModel):
    max_combinations: int = 64
    sampling_strategy: Literal["all", "random", "grid"] = "all"
    random_seed: int = 0
```

**작업 항목**:
- [ ] `webapp/dse/core/schemas.py` — Pydantic 모델 정의
- [ ] `generate_hw_counts(pool: ResourcePool, top_n: int) -> list[dict[hw, int]]` — cartesian product
- [ ] **사전 필터 1**: 모델 weight 메모리 합 > pool 합산 메모리 → reject
  - 8B FP16 = 16GB, 70B FP16 = 140GB. tp 분산 고려한 per-NPU 메모리 체크
- [ ] **사전 필터 2**: peak FLOPS × num_npus < workload throughput floor → reject (선택)
- [ ] **사전 필터 3**: 카탈로그에 모델 perf profile 없는 (hw, model) 페어 → reject (기존 `_tp_dir_is_complete` 활용)
- [ ] **gemerated instance_groups 변환**: 각 hw 카운트 조합을 `enumerate_configs` 입력 형식인 `instance_groups`로 변환
  - 예: `{H100: 2, RNGD: 4}` → `[{hw:"H100", count:1}×2, {hw:"RNGD", count:1}×4]` 또는 `[{hw:"H100", count:2}, {hw:"RNGD", count:4}]`
  - 두 형태 trade-off: 전자는 enumerate가 instance별 P/D 역할 자유 분배, 후자는 group별 분배
- [ ] `enumerate_configs(scenario, catalog)` 호출하여 각 hw 조합에서 valid parallelism configs 생성
- [ ] **샘플링**: `max_combinations` 초과 시 random sampling 또는 grid-based stratified sampling
- [ ] 출력: `list[CandidateConfig]` (각 candidate = `ConfigSpec` + hw 분배 메타)

**테스트** (`tests/dse/test_generator.py`):
- [ ] 단일 hardware pool에서 enumerate 결과와 동일
- [ ] 메모리 필터: 70B 모델 + H100 1개 → 메모리 부족으로 reject 확인
- [ ] 조합 폭발 방지: max_combinations=10일 때 출력 ≤ 10
- [ ] 카탈로그에 없는 (hw, model) → 자동 제외

**예상 공수**: 2일

### 1.2 Cluster Config Builder (`webapp/dse/core/config_builder.py`)

기존 `webapp/cluster_builder.py:build_cluster_json`을 호출하는 얇은 래퍼.

**작업 항목**:
- [ ] `build_dse_cluster_json(candidate, link_bw, link_latency, power_template)` — 기존 build_cluster_json wrapping
- [ ] Power template 자동 lookup: `docs/dse/03_catalog.yaml`의 `tdp_w`, `idle_power_w` 등에서 자동 생성 (사용자 수동 입력 불필요)
- [ ] cpu_mem / link_bw / link_latency 합리적 default
- [ ] dry-run 검증: 생성된 JSON을 `inference_serving/config_builder.build_cluster_config()`에 통과시켜 schema valid 확인
- [ ] 임시 디렉토리 출력: `output/dse_jobs/<job_id>/configs/<cand_id>.json`

**테스트**:
- [ ] 알려진 (hw, model, count) 조합에 대해 출력이 `inference_serving/config_builder`를 통과

**예상 공수**: 0.5일

### 1.3 Simulation Runner (`webapp/dse/core/runner.py`)

기존 `webapp/runner.py:run_sweep`을 재사용. DSE는 후보 리스트 → 병렬 실행 → 결과 수집.

**작업 항목**:
- [ ] `run_dse_job(job_id, candidates, workload, output_dir)` — 신규 entry point
- [ ] 내부적으로 `run_sweep` 호출 (이미 MAX_CONCURRENT semaphore + cleanup 적용됨)
- [ ] candidate별 결과를 `SimulationResult` dataclass로 변환:
  ```python
  @dataclass
  class SimulationResult:
      candidate_id: str
      metrics: dict   # parser.parse_run() 결과
      cluster_config_path: Path
      raw_csv_path: Path
      log_path: Path
      state: Literal["done", "failed", "timeout", "cancelled"]
      elapsed_s: float
      error: str | None = None
  ```
- [ ] Progress callback (Phase 3 SSE 연동) — 이미 `_broadcast` 인프라 있음

**테스트**:
- [ ] 작은 후보 셋 (3개) 병렬 실행 시 결과 3개 수집
- [ ] 실패 후보가 있어도 다른 후보 진행에 영향 없음

**예상 공수**: 0.5일 (대부분 재사용)

### 1.4 Result Analyzer / Ranker (`webapp/dse/core/ranker.py`)

**목적**: SimulationResult 리스트 → SLO 필터 + Pareto + Top-N.

**작업 항목**:

#### 1.4.1 SLO 필터
- [ ] `filter_slo(results, constraints) -> list[SimulationResult]`
- [ ] 적용 기준 (Constraints):
  ```python
  class Constraints(BaseModel):
      ttft_p99_ms: float | None = None
      tpot_p99_ms: float | None = None
      itl_p99_ms: float | None = None
      throughput_min_tok_s: float | None = None
      power_max_w: float | None = None
      energy_max_wh: float | None = None
  ```
- [ ] 각 조건 위반 시 `meets_slo=False` 라벨링 (제거하지는 않음, Pareto 계산에서 제외)

#### 1.4.2 Pareto frontier
- [ ] `pareto_frontier(results, objectives)` — 다차원 Pareto
- [ ] 기존 `plots.py:_pareto_frontier`는 2D — `n-dimensional` 확장 필요
- [ ] objectives 예: `[("ttft_p99_ms", "min"), ("throughput", "max"), ("total_energy_wh", "min")]`
- [ ] 알고리즘: O(N²) naive dominance check (N ≤ 1000이면 충분)

#### 1.4.3 Weighted scoring
- [ ] `compute_score(result, weights, norms)` — normalize 후 가중합
- [ ] Normalization: min-max scaling over candidate set
- [ ] 방향: latency/power는 minimize (1 - normalized), throughput은 maximize (normalized 그대로)
- [ ] `weights: dict[str, float]` 정규화하여 합=1로 보정

#### 1.4.4 Top-N + diversity
- [ ] `top_n(results, n=5, diversity_key="hw_signature")` — 단순 정렬 후 동일 hw signature 중복 시 다음 순위로
- [ ] hw_signature = sorted tuple of (hw, count) — 같은 하드웨어 조합은 1번만 노출

#### 1.4.5 Cost model (선택)
- [ ] `compute_cost(result, unit_costs)` — `Σ count × unit_cost`

**테스트**:
- [ ] SLO 위반 후보는 Pareto에 포함되지 않음
- [ ] 단일 차원 Pareto는 단순 정렬 결과
- [ ] 가중치 [1.0, 0.0]은 단일 차원 정렬과 동치
- [ ] diversity_key 적용 시 hw 중복 제거

**예상 공수**: 1.5일

### 1.5 CLI 진입점 (`webapp/dse/cli.py`)

**작업 항목**:
- [ ] `python -m webapp.dse.cli explore --spec spec.yaml --out results/`
- [ ] YAML/JSON spec 로드 → ResourcePool, Constraints, SearchConfig
- [ ] generator → config_builder → runner → ranker 파이프라인
- [ ] 출력: `results/top_n.json`, `results/all_candidates.json`, `results/pareto.json`
- [ ] verbose 모드: 단계별 progress bar (tqdm)
- [ ] argparse 또는 typer

**테스트**:
- [ ] `examples/dse/spec_llama8b_small.yaml`로 end-to-end (작은 pool, 5개 candidate)

**예상 공수**: 1일

### Phase 1 합계: **5.5일** — ✅ **완료** (CLI 동작: `python -m webapp.dse.cli explore`, 6/6 candidates 생성)

---

## Phase 2 — 검증 및 안정화

### 2.1 회귀 테스트

**목적**: DSE 도구 경유 vs `main.py` 단독 실행 결과 동일.

**작업 항목**:
- [ ] 알려진 cluster_config 2개 선정 (`single_node_single_instance.json`, `single_node_pd_instance.json`)
- [ ] 두 실행 경로의 메트릭 (`total_token_tp`, `mean_ttft_ms`, `mean_tpot_ms`, `mean_itl_ms`, `total_energy_wh`) 차이 ≤ 0.1%
- [ ] PID_TAG 격리로 인한 결과 동일성 (동일 PID에서 단독 실행 vs DSE pipeline)

### 2.2 조합 폭발 방지 검증

- [ ] resource pool `{H100: 8, A6000: 16, RNGD: 4}` 전체 cartesian = 4×9×17×5 = 3060
- [ ] `max_combinations=64`로 자르면 후보 ≤ 64
- [ ] random vs grid sampling 둘 다 검증

### 2.3 병렬 실행 race 방지 재확인

- [ ] PID_TAG isolation은 이미 적용됨. DSE pipeline에서도 동일하게 동작 확인
- [ ] cleanup 호출 확인 (sweep 종료 후 `astra-sim/inputs/{trace,workload}/pid*` 0개)

### 2.4 단위 테스트 인프라

- [ ] `tests/dse/conftest.py` — pytest fixture (가짜 catalog, 가짜 results)
- [ ] `tests/dse/test_generator.py`, `test_builder.py`, `test_ranker.py`
- [ ] CI 통합 (있다면)

### Phase 2 합계: **1.5일** — ✅ **완료** (38/38 unit tests passing, E2E smoke ✓)

---

## Phase 3 — 웹 백엔드

기존 `webapp/app.py` FastAPI 앱에 DSE 라우트 추가.

### 3.1 API 엔드포인트

**작업 항목**:
- [ ] `POST /api/dse/jobs` — body는 spec (Pydantic validation)
  ```python
  {
    "resource_pool": {...},
    "model": {...},
    "workload": {...},
    "constraints": {...},
    "features": {...},
    "search": {...}
  }
  ```
  - 응답: `{"job_id": "20260601-093012-abc12", "estimated_candidates": 32}`
- [ ] `GET /api/dse/jobs/{job_id}` — 상태 + progress %
- [ ] `GET /api/dse/jobs/{job_id}/results?sort=score&top=5` — 결과 + Top-N
- [ ] `GET /api/dse/jobs/{job_id}/results/{candidate_id}` — 개별 후보 상세
- [ ] `GET /api/dse/jobs/{job_id}/download.zip` — 결과 일괄
- [ ] `DELETE /api/dse/jobs/{job_id}` — 작업 취소/삭제
- [ ] **재랭킹**: `POST /api/dse/jobs/{job_id}/rerank` — body: 새 weights, 응답: 새 Top-N (시뮬레이션 재실행 없음, 결과 캐시에서 즉시 계산)
- [ ] `GET /api/dse/catalog` — hardware × model 카탈로그 (기존 `/api/hardware` 재사용 + 메타데이터 보강)

### 3.2 SSE / WebSocket

- [ ] 기존 `webapp/app.py:/api/sweeps/{id}/events` SSE 패턴 재사용
- [ ] 신규 `/api/dse/jobs/{job_id}/events` — 후보 단위 진행 + 새 Pareto 포인트 push
- [ ] 이벤트 종류: `candidate_started`, `candidate_done`, `pareto_updated`, `job_done`

### 3.3 Job DB

**스키마**:
```sql
CREATE TABLE dse_jobs (
  id TEXT PRIMARY KEY,
  spec_json TEXT NOT NULL,
  spec_hash TEXT NOT NULL,    -- cache key
  state TEXT NOT NULL,         -- queued/running/done/failed/cancelled
  progress_done INT DEFAULT 0,
  progress_total INT DEFAULT 0,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  output_dir TEXT NOT NULL
);

CREATE TABLE dse_candidates (
  job_id TEXT NOT NULL,
  cand_id TEXT NOT NULL,
  config_json TEXT NOT NULL,
  result_json TEXT,
  score REAL,
  on_pareto INT DEFAULT 0,
  meets_slo INT DEFAULT 0,
  PRIMARY KEY (job_id, cand_id),
  FOREIGN KEY (job_id) REFERENCES dse_jobs(id)
);

CREATE INDEX idx_spec_hash ON dse_jobs(spec_hash);
```

**작업 항목**:
- [ ] `webapp/dse/server/db.py` — SQLAlchemy/SQLModel 또는 sqlite3 thin wrapper
- [ ] 결과 캐시: `spec_hash` 일치 + state='done'이면 즉시 기존 결과 반환
- [ ] 마이그레이션: alembic은 과한 수준. 단일 `CREATE TABLE IF NOT EXISTS`로 부트스트랩

### 3.4 BackgroundTasks

- [ ] FastAPI BackgroundTasks로 `run_dse_job` 비동기 실행
- [ ] 동시 DSE job 수 상한 (env var or config): `DSE_MAX_CONCURRENT_JOBS=2`
- [ ] job마다 sub-semaphore로 candidate 동시 실행 (이미 `MAX_CONCURRENT` 적용됨)

### Phase 3 합계: **3일** — ✅ **완료** (10 DSE 라우트, /api/dse/*, SSE 재사용)

---

## Phase 4 — 웹 프론트엔드

기존 `webapp/templates/` + `webapp/static/` 스택 (Jinja + vanilla JS + Plotly) 유지.

### 4.0 헤더 Nav Bar 추가 ★ (B-1)

**목적**: 사용자가 모든 페이지에서 한 클릭으로 Sweep / DSE 전환 가능.

**변경 파일**:
- [ ] `webapp/templates/base.html` — `<body>` 시작 직후 `<nav>` 추가
- [ ] `webapp/static/app.css` — `.app-nav`, `.app-nav-link`, `.app-nav-link.active` 스타일 추가

**HTML 마크업** (예시):
```html
<nav class="app-nav">
    <a href="/" class="app-nav-brand">LLMServingSim</a>
    <div class="app-nav-links">
        <a href="/" class="app-nav-link {% if active_nav == 'sweep' %}active{% endif %}">Sweep</a>
        <a href="/dse/explore" class="app-nav-link {% if active_nav == 'dse' %}active{% endif %}">DSE</a>
        <a href="/sweeps" class="app-nav-link {% if active_nav == 'history' %}active{% endif %}">History</a>
    </div>
</nav>
```

**active_nav 변수 주입**:
- [ ] `webapp/app.py`의 각 render() 호출에 `active_nav` 추가
  - `page_index` → `active_nav="sweep"`
  - `page_progress` / `page_results` / `page_sweeps_list` → `active_nav="history"`
- [ ] DSE 라우트들은 `active_nav="dse"`
- [ ] 누락 시 default 빈 문자열 — 안전한 fallback

**CSS 디자인 기준**:
- [ ] 높이 ~44px, sticky top
- [ ] 다크/라이트 통일 (기존 `--surface`, `--border` 변수 재사용)
- [ ] active 링크는 underline + 강조 색상
- [ ] Hover 시 subtle background tint
- [ ] 모바일 (`@media (max-width: 720px)`): 햄버거 토글로 collapse (선택 — v2에서 보강해도 무방)

**기존 페이지 회귀 검증**:
- [ ] `/` (index): page-header 가독성 / 입력 폼 레이아웃 영향 없음
- [ ] `/sweep/{id}` (progress): progress-summary와 nav 간격 OK
- [ ] `/sweep/{id}/results`: Configuration Matrix 표 / 차트 영역 squeeze 없음
- [ ] `/sweeps` (history list): 페이지 헤더 중복 인상 없도록 디자인

**예상 공수**: 0.5일

### 4.1 입력 페이지 (`/dse/explore`)

**파일**:
- [ ] `webapp/templates/dse_explore.html`
- [ ] `webapp/static/dse_explore.js`

**섹션**:
- [ ] **자원 풀** — 행 추가/삭제 가능한 표 (hw select + min/max input)
  - Hardware select는 `/api/hardware`에서 자동 채움
- [ ] **모델/워크로드**
  - 모델 dropdown (선택된 hw들의 카탈로그 교집합만 표시)
  - 데이터셋 dropdown (`/api/datasets`) 또는 합성 트래픽 폼 (QPS, input len, output len)
- [ ] **요구사항**
  - SLO: TTFT p99 (ms), TPOT p99 (ms), ITL p99 (ms) — number input
  - Throughput floor (tok/s)
  - Power ceiling (W) or energy ceiling (Wh)
- [ ] **탐색 옵션**
  - Max combinations (slider 8–256)
  - Parallelism (max concurrent simulations, default = `MAX_CONCURRENT`)
  - Feature toggles: P/D disaggregation, prefix caching, attn offloading
  - Objective weights — 슬라이더 4개 (latency/throughput/power/cost), 합=1로 자동 normalize
- [ ] **검증 미리보기**
  - "예상 후보 수: N (필터 후 약 M)" 실시간 표시 — 클라이언트 사이드 카운트 또는 dry-run endpoint
- [ ] **Preset 저장/불러오기** — localStorage 기반 (서버 저장은 v2)
- [ ] **Start Exploration** 버튼 → POST /api/dse/jobs → redirect `/dse/jobs/{id}`

### 4.2 진행 모니터 페이지 (`/dse/jobs/{id}`)

**파일**:
- [ ] `webapp/templates/dse_progress.html`
- [ ] `webapp/static/dse_progress.js`

**UI 요소**:
- [ ] 전체 progress bar (N/M done, elapsed sec — 기존 sweep progress 패턴 재사용)
- [ ] 현재 실행 중인 후보 리스트 (label + elapsed)
- [ ] **실시간 산점도** (Plotly streaming):
  - X: TTFT p99, Y: throughput, color: power, size: hw_count
  - 새 후보 도착 시 마커 추가
  - Pareto frontier 실시간 갱신 (점선)
- [ ] Cancel 버튼

### 4.3 결과 대시보드 (`/dse/jobs/{id}/results`)

**파일**:
- [ ] `webapp/templates/dse_results.html`
- [ ] `webapp/static/dse_results.js`

**섹션**:
- [ ] **Top-N 테이블**
  - 순위, hw 구성 (예: `2×H100 + 4×A6000`), parallelism (TP/PP/DP), TTFT p99, TPOT p99, throughput, power, score
  - 클릭 시 상세 모달
  - 정렬 가능 (각 컬럼 헤더)
- [ ] **Pareto frontier plot** (2D + 축 선택 dropdown)
  - 축 옵션: ttft/tpot/itl/throughput/power/energy
  - Pareto-optimal 마커는 강조 색상
  - 비-Pareto 후보는 회색
- [ ] **레이더 차트**
  - Top-N 후보의 정규화 메트릭 (ttft, tpot, itl, throughput, power, score)
  - Plotly Scatterpolar
- [ ] **가중치 슬라이더** + "Re-rank" 버튼
  - 슬라이더 변경 시 즉시 `/api/dse/jobs/{id}/rerank` POST
  - Top-N 테이블 즉시 갱신 (재시뮬 없음)
- [ ] **상세 모달**
  - cluster_config JSON (syntax-highlighted)
  - raw CSV download link
  - 재현 CLI 명령어 (clipboard copy 버튼)
  - 단일 후보의 TTFT/ITL CDF 차트 (이미 plots.py 구현됨 — 재사용)
- [ ] **다운로드 섹션**
  - 전체 결과 zip
  - Top-N JSON
  - Pareto-only CSV

### 4.4 공통

- [ ] 다국어 (한/영) — 단순 dict 기반 i18n (`webapp/dse/i18n.py`)
- [ ] 색상 매핑 통일 — 기존 `assign_config_colors` 재사용
- [ ] 다크모드 (선택)

### Phase 4 합계: **5.5일** (4.0 nav bar 0.5일 + 4.1-4.4 5일) — ✅ **완료** (nav bar + 3 페이지 + 3 JS)

---

## Phase 5 — 통합, 문서화, 배포

### 5.1 문서

- [ ] `README.md` — DSE 섹션 추가 ("Design Space Exploration" 헤더)
- [ ] `webapp/dse/README.md` — 모듈 개요, 사용법, API 레퍼런스
- [ ] `docs/dse/USER_GUIDE.md` — 한국어 / 영어 사용자 가이드 (스크린샷 포함)
- [ ] `docs/dse/DEVELOPER_GUIDE.md` — 코드 구조, 확장 포인트, 새 hardware 추가 방법

### 5.2 Docker 통합

- [ ] 기존 `docker.sh`로 띄운 컨테이너에서 DSE 페이지 접근 가능
- [ ] `script/serve_webapp.sh`는 이미 uvicorn 띄우므로 `/dse/...` 라우트가 같은 서버에 마운트됨 (자동 — Phase 3 라우트 추가하면 됨)
- [ ] (선택) `docker compose` 파일에서 DSE job worker 별도 컨테이너

### 5.3 예시 spec 파일

- [ ] `examples/dse/spec_llama8b_a6000.yaml` — A6000 0~4 pool, Llama-3.1-8B, SLO 보수적
- [ ] `examples/dse/spec_llama70b_mixed.yaml` — H100 0~4 + A6000 0~8, Llama-3.1-70B
- [ ] `examples/dse/spec_pd_disagg.yaml` — P/D 분리 강제

### 5.4 E2E 데모 시나리오

PLAN_webapp_dse.md §4 Phase 5 명시 시나리오:

> "H100 1~4, A6000 0~8 범위에서 Llama-3.1-70B를 TTFT p99 ≤ 500ms, power ≤ 3kW로 서빙하는 최적 조합 5개"

**작업 항목**:
- [ ] `examples/dse/spec_demo_llama70b.yaml` 작성
- [ ] `script/demo_dse.sh` — spec 로드 → DSE job 시작 → 결과 페이지 URL 출력
- [ ] README에 데모 섹션 + 스크린샷 / GIF

### Phase 5 합계: **2일** — ✅ **완료** (webapp/dse/README.md, examples/dse/, script/demo_dse.sh)

---

## 6. 통합 마일스톤 및 검증 기준

| M | 목표 | 기준 | 예상 시점 |
|---|------|------|-----------|
| **M0** | Phase 0 완료 | 4개 문서 작성, 스키마 검증 (`jsonschema.Draft7Validator`로 모든 예시 config 통과) | Day 3 |
| **M1** | Phase 1 CLI 동작 | `python -m webapp.dse.cli explore --spec examples/dse/spec_llama8b_a6000.yaml` 실행 시 Top-5 JSON 출력 + 실제 시뮬레이션 호출 | Day 8 |
| **M2** | Phase 2 회귀 통과 | 동일 cluster_config 단독 vs DSE 경유 메트릭 동일 (≤ 0.1% 차이), 64-candidate sweep race-free 완료 | Day 10 |
| **M3** | Phase 3 백엔드 동작 | 8개 API + SSE 정상, 동시 DSE job 2개 race 없음, 재랭킹 endpoint 1초 이내 응답 | Day 13 |
| **M4** | Phase 4 프론트엔드 동작 | 입력 → 진행 → 결과 흐름 브라우저 완결, 가중치 슬라이더 즉시 재랭킹, 한/영 토글, **nav bar로 Sweep ↔ DSE 한 클릭 전환** | Day 18 |
| **M5** | Phase 5 데모 | README의 `script/demo_dse.sh` 한 줄로 e2e 시나리오 재현 | Day 20 |

**총 예상 공수**: **20 person-days = 약 4 weeks** (단일 개발자 풀타임 기준)

병렬화 가능성:
- Phase 1과 Phase 0.4 (카탈로그)는 동시 가능
- Phase 4는 Phase 3 API 윤곽만 확정되면 stub으로 진행 가능
- 2명이 분담하면 약 **2.5 weeks**

---

## 7. 리스크 매트릭스

| 리스크 | 영향 | 확률 | 완화 |
|---|---|---|---|
| ASTRA-Sim heterogeneous deadlock | High (DSE의 핵심 가치 훼손) | High | 이미 `enumerate.py` 필터로 차단. DSE generator도 이 필터를 거치므로 자동 회피 |
| 조합 폭발 | Medium (UX) | High | max_combinations cap + dry-run 카운트 표시 |
| 큰 모델 시뮬레이션 시간 | Medium | High | workload sub-sampling 옵션, timeout 자동 fail |
| 카탈로그 일부 hw에 perf profile 없음 | Low | Medium | `_tp_dir_is_complete` 자동 필터링 (이미 구현) |
| 결과 캐시 invalidation 버그 | Medium | Low | spec_hash는 deterministic (재현 가능). 캐시는 explicit invalidate API 제공 |
| webapp/과 webapp_dse/ 통합 시 라우트 충돌 | Low | Low | `/api/dse/...` prefix로 분리. Jinja 템플릿도 `dse_*.html` prefix |
| Nav bar 추가로 기존 페이지 layout shift | Low | Medium | Phase 4.0에서 모든 기존 페이지 회귀 체크리스트 (위 §4.0 참조). 높이 fixed 44px로 누적 height 변화 예측 가능 |
| SLO를 통과하는 후보가 없는 경우 | Medium (UX) | Medium | 명확한 "0 valid candidates" 메시지 + SLO 완화 추천 |
| 메모리 / 디스크 누적 | Medium | Medium | `_cleanup_pid_artifacts` 이미 적용 + job별 디렉토리 retention 정책 (기본 7일) |

---

## 8. 새 의존성 (있다면)

| 패키지 | 용도 | 필수성 |
|---|---|---|
| `pydantic >= 2.0` | spec 검증 | 필수 (기존 webapp이 이미 사용 중) |
| `pyyaml` | YAML spec 파싱 | 필수 |
| `tqdm` | CLI progress bar | 선택 |
| `sqlmodel` 또는 `sqlite3` | job DB | 필수 (sqlite3는 stdlib) |
| `hypothesis` | property-based testing | 선택 (테스트 견고성↑) |

기존 webapp에 이미 있는: FastAPI, uvicorn, Jinja2, Plotly.js — 추가 불필요.

---

## 9. 구체적 인터페이스 시그니처 (참고)

### 9.1 `webapp/dse/core/schemas.py`
```python
class CandidateConfig(BaseModel):
    candidate_id: str         # "c042"
    config_spec: ConfigSpec   # 기존 webapp.cluster_builder.ConfigSpec
    hw_distribution: dict[str, int]  # {"H100": 2, "A6000": 4}
    parallelism: dict[str, int]      # {"TP": 2, "PP": 1, "DP": 3}
    pd_layout: str                   # "1P+3D" or "—"

class JobSpec(BaseModel):
    resource_pool: ResourcePool
    model: ModelSpec
    workload: WorkloadSpec
    constraints: Constraints
    features: FeatureFlags
    search: SearchConfig
    objective_weights: dict[str, float]
    top_n: int = 5
```

### 9.2 `webapp/dse/core/generator.py`
```python
def generate_candidates(
    spec: JobSpec,
    catalog: dict[tuple[str, str], frozenset[int]],
    hw_meta: dict[str, dict],
) -> list[CandidateConfig]: ...
```

### 9.3 `webapp/dse/core/ranker.py`
```python
def rank_candidates(
    results: list[SimulationResult],
    constraints: Constraints,
    weights: dict[str, float],
    top_n: int = 5,
    diversity: bool = True,
) -> RankedResults:
    """
    Returns:
        RankedResults with: all_results, pareto_optimal, top_n, score_per_candidate
    """
```

---

## 10. 첫 1주 시작 시 권장 작업 순서

day 1: Phase 0.1, 0.3 (기존 코드 매핑 문서)
day 2: Phase 0.2 (스키마), 0.4 (카탈로그)
day 3: Phase 1.1 generator MVP (cartesian + 메모리 필터만)
day 4: Phase 1.2 config_builder wrap + 1.3 runner wrap
day 5: Phase 1.4 ranker (Pareto + scoring) + 1.5 CLI
day 6–7: Phase 1 end-to-end 테스트 + Phase 2 회귀

이 7일이 끝나면 **CLI로 완결된 DSE 도구**가 손에 들어옴. Phase 3~5는 그 위에 웹과 데모를 얹는 과정.

---

## 부록 A. 기존 webapp 함수 → DSE 매핑

| DSE 단계 | 호출하는 기존 함수 |
|---|---|
| generator → enumerate_configs | `webapp/enumerate.py:enumerate_configs(scenario, catalog)` |
| config_builder → cluster JSON | `webapp/cluster_builder.py:build_cluster_json(spec, cpu_mem, link_bw, link_latency, power_template)` |
| runner → 병렬 sweep | `webapp/runner.py:run_sweep(sweep_id, configs, scenario_json, sweep_dir, workload)` |
| ranker → 메트릭 파싱 | `webapp/parser.py:parse_run(log_path, csv_path)` |
| Pareto frontier | `webapp/plots.py:_pareto_frontier(points)` (확장 필요: 2D → ND) |
| 색상 매핑 | `webapp/plots.py:assign_config_colors(labels)` |
| 카탈로그 조회 | `webapp/hardware_catalog.py:build_catalog()`, `get_tp_options()` |
| 작업 정리 | `webapp/runner.py:_cleanup_pid_artifacts(pid)` |

이 매핑이 곧 "code reuse plan"입니다. 신규 코드는 DSE 특유 로직 (generator, ranker, Top-N UI) 에 집중.
