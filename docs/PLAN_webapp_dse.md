# PLAN.md — LLM Serving 최적 조합 탐색 웹 인터페이스

> LLMServingSim 위에서 동작하는 **Design Space Exploration (DSE) 웹 도구** 개발 계획.
> 사용자가 보유 가능한 컴퓨팅 자원 풀과 요구사항(SLO, throughput, power)을 입력하면, 시스템이 가능한 cluster 조합들을 자동 생성 → 시뮬레이션 → 결과 랭킹 → 최적 조합 Top‑N을 시각화한다.

---

## 0. 목차
1. 프로젝트 개요
2. 사전 조사 및 분석 (Phase 0)
3. 시스템 아키텍처
4. 작업 목록 (Phase 1 ~ Phase 5)
5. 주요 기능 명세
6. 데이터 모델 / API 스펙
7. 디렉토리 구조 (제안)
8. 기술 스택
9. 마일스톤 및 검증 기준
10. 리스크 및 향후 확장

---

## 1. 프로젝트 개요

### 1.1 목표
기존 `webapp/`이 **하나의 cluster config → 결과 시각화**만 지원하는 데 비해, 새 인터페이스는 **요구사항 → 다수의 cluster 조합 자동 생성 → 일괄 시뮬레이션 → 최적해 Top‑N 도출**을 수행한다.

### 1.2 입력
- **가용 자원 풀**: 하드웨어 타입별 최대 개수 (예: H100 ≤ 8, A6000 ≤ 16, TPU‑v6e‑1 ≤ 4)
- **모델 / 워크로드**: 대상 모델(Llama‑3.1‑8B/70B, Mixtral‑8x7B, Phi‑mini‑MoE), 데이터셋(jsonl) 또는 합성 트래픽 파라미터(QPS, input/output len 분포)
- **요구사항(Constraints + Objectives)**:
  - SLO: TTFT p99 ≤ X ms, TPOT p99 ≤ Y ms, ITL p99 ≤ Z ms
  - 처리량: throughput ≥ N tokens/s (또는 req/s)
  - 소비전력: total power ≤ P W (또는 energy ≤ E J/req)
  - (선택) 비용 상한, 메모리 상한
- **탐색 제어**: 최대 조합 수, 병렬도, P/D 분리 허용 여부, PIM/CXL 사용 여부, prefix caching 등 feature flags

### 1.3 출력
- 모든 시뮬레이션 결과의 trade‑off 시각화 (Pareto frontier)
- 사용자 정의 weight에 따른 **최적 조합 Top‑N** (기본 N=5)
- 각 조합의 상세 메트릭, 사용된 cluster_config JSON, 재현 명령어
- CSV / JSON 일괄 다운로드

### 1.4 핵심 가치
- "어떤 하드웨어를 몇 개, 어떻게 조합해야 SLO를 만족시키며 비용/전력을 최소화할 수 있는가"라는 **What‑if 질문에 정량적 답변**을 제공.
- 시뮬레이션 자체 로직은 LLMServingSim에 위임하고, 본 도구는 **DSE 오케스트레이션 계층**으로만 동작 (기존 코드 침습 최소화).

---

## 2. 사전 조사 및 분석 (Phase 0)

> Claude Code 첫 작업. 이 단계 산출물(요약 노트들)이 이후 모든 Phase의 입력이 된다.

### 2.1 기존 `webapp/` 코드 리딩
- [ ] 사용 중인 프레임워크(Flask / FastAPI / Streamlit 등) 식별
- [ ] 라우팅 구조, 정적/템플릿 위치, 시뮬레이션 호출 방식 파악
- [ ] 결과 시각화에 쓰인 차트 라이브러리 확인 (Plotly / Chart.js / Recharts 등)
- [ ] 재사용 가능한 컴포넌트(차트, config 뷰어 등) 리스트업
- 산출: `docs/dse/00_existing_webapp_notes.md`

### 2.2 `cluster_config/*.json` 스키마 역공학
- [ ] 모든 예시 config를 읽고 필드 분류 (node 토폴로지 / 인스턴스 레이아웃 / 메모리 계층 / 인터커넥트 / per‑layer 배치 / PIM)
- [ ] 필수 vs 선택 필드, 값 범위, 상호 의존성 정리 → 스키마 표 작성
- [ ] `inference_serving/config_builder.py` 동작 분석 (어떤 검증/변환을 수행하는지)
- 산출: `docs/dse/01_cluster_config_schema.md`, `cluster_config_schema.json` (JSON Schema 형식)

### 2.3 `main.py` I/O 명세
- [ ] CLI 플래그를 카테고리별로 정리 (자원/모델/스케줄링/feature/출력)
- [ ] 출력 CSV 컬럼 의미와 단위 정리 (TTFT, TPOT, ITL, throughput, power 등)
- [ ] stdout 로그에서 파싱 가능한 메트릭(throughput 시계열, memory, power) 식별
- 산출: `docs/dse/02_main_io_spec.md`

### 2.4 하드웨어 / 모델 카탈로그 작성
- [ ] `llm_profile/`에서 지원되는 (model, hardware) 페어와 프로파일링 파일 매핑
- [ ] 각 하드웨어의 메모리 용량, peak FLOPS, TDP 등 기준값 정리 (DSE 단계에서 사전 필터링에 사용)
- 산출: `docs/dse/03_catalog.yaml`

---

## 3. 시스템 아키텍처

```
┌──────────────────────────── Frontend (Browser) ────────────────────────────┐
│  [입력 폼]  [탐색 진행 모니터]  [결과 대시보드 / Pareto plot / Top-N 표]    │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │  HTTP / WebSocket(진행상황)
┌────────────────────────────────────▼───────────────────────────────────────┐
│                          DSE Web Backend (FastAPI 권장)                    │
│  - REST API: /jobs, /jobs/{id}, /jobs/{id}/results                         │
│  - WebSocket: /ws/jobs/{id}  (실시간 진행률)                                │
│  - Job DB (SQLite로 시작) + 결과 캐시                                       │
└──────┬──────────────────────────┬───────────────────────────────┬──────────┘
       │                          │                               │
       ▼                          ▼                               ▼
┌────────────────┐   ┌────────────────────────┐   ┌──────────────────────────┐
│ Combination    │   │ Simulation Orchestrator │   │ Result Analyzer / Ranker │
│ Generator      │──▶│ (parallel worker pool)  │──▶│ - SLO 필터               │
│ - 제약 기반     │   │ - cluster_config 생성   │   │ - Pareto frontier 계산   │
│ - 가지치기      │   │ - main.py 실행          │   │ - 가중합 스코어링        │
└────────────────┘   │ - 결과 파싱             │   │ - Top-N 선정             │
                     └───────────┬─────────────┘   └──────────────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ LLMServingSim    │
                       │ (main.py)        │
                       └──────────────────┘
```

**핵심 원칙**
- LLMServingSim 본체 코드는 **불수정 원칙** (수정이 필요하면 별도 PR로 분리)
- 새 코드는 모두 `webapp_dse/`(가칭) 하위에 격리
- 시뮬레이션 작업은 **subprocess** + 임시 디렉토리 모델로 격리 (병렬 안전)

---

## 4. 작업 목록

### Phase 1 — 백엔드 코어 엔진
실행 가능한 CLI 형태로 먼저 완성한 뒤, Phase 3에서 웹으로 래핑한다.

#### 1.1 Combination Generator (`webapp_dse/core/generator.py`)
- [ ] 입력 스키마 정의 (Pydantic `ResourcePool`, `Constraints`, `SearchConfig`)
- [ ] 기본 enumerator: 하드웨어 카운트 조합을 cartesian product로 생성
- [ ] **사전 필터(static pruning)**: 모델 weight 메모리 > 합산 메모리인 조합 즉시 제거
- [ ] 병렬화 차원(TP/PP/DP) 자동 후보 산출 함수
- [ ] P/D disaggregation 분배 후보 생성 (prefill 노드 수 vs decode 노드 수)
- [ ] 출력: `List[CandidateConfig]` (이 단계에서는 아직 JSON 파일 아님)

#### 1.2 Cluster Config Builder (`webapp_dse/core/config_builder.py`)
- [ ] `CandidateConfig` → `cluster_config/*.json` 변환기
- [ ] 템플릿 기반 (기존 예시 config를 base template으로 사용)
- [ ] 생성된 JSON이 LLMServingSim `config_builder.py` 검증을 통과하는지 dry‑run
- [ ] 임시 디렉토리(`/tmp/dse/<job_id>/configs/`)에 결과 저장

#### 1.3 Simulation Runner (`webapp_dse/core/runner.py`)
- [ ] 단일 시뮬레이션 실행 함수 (subprocess로 `python main.py ...` 호출)
- [ ] timeout, retry, stderr 캡처
- [ ] CSV 출력 파싱 → `SimulationResult` dataclass
- [ ] stdout 로그에서 throughput/power 시계열 파싱
- [ ] **병렬 실행기**: `concurrent.futures.ProcessPoolExecutor` 기반, 동시 실행 수 조절
- [ ] 진행 상황 콜백(callback) 인터페이스 (Phase 3 WebSocket 연동용)

#### 1.4 Result Analyzer / Ranker (`webapp_dse/core/ranker.py`)
- [ ] SLO 위반 필터 (TTFT/TPOT/ITL p99, throughput floor, power ceiling)
- [ ] **Pareto frontier 계산** (objective: latency↓, throughput↑, power↓, cost↓)
- [ ] 사용자 정의 가중합 스코어링: `score = w1·norm(latency) + w2·norm(power) - w3·norm(throughput) ...`
- [ ] 비용 모델(선택): 하드웨어별 단가 테이블 곱해 `total_cost` 산출
- [ ] Top‑N 선정 알고리즘 (단순 정렬 + 다양성 보장 옵션: 동일 하드웨어 구성 중복 억제)

#### 1.5 CLI 진입점 (`webapp_dse/cli.py`)
- [ ] `python -m webapp_dse.cli explore --spec spec.yaml --out results/` 명령
- [ ] 웹 없이도 동작 가능한 형태로 Phase 1을 마무리 → 단위 테스트 용이

### Phase 2 — 검증 및 안정화
- [ ] 알려진 cluster_config 1~2개로 회귀 테스트: 기존 main.py 결과와 본 도구를 통해 실행한 결과가 동일한지 확인
- [ ] generator의 조합 수 폭발 방지(상한 / sampling) 동작 검증
- [ ] runner의 병렬 실행에서 임시 디렉토리/파일 충돌 없음 확인
- [ ] 단위 테스트(`tests/dse/`): generator, builder, ranker

### Phase 3 — 웹 백엔드 (FastAPI 권장)
> 기존 `webapp/`이 Flask면 Flask로 통일해도 무방. Phase 0의 조사 결과로 결정.

#### 3.1 API 엔드포인트
- [ ] `POST /api/jobs` — 새 탐색 작업 생성 (입력: spec JSON, 반환: `job_id`)
- [ ] `GET  /api/jobs/{id}` — 상태 조회 (queued / running / done / failed, progress %)
- [ ] `GET  /api/jobs/{id}/results` — 전체 결과 + Top‑N (필터/정렬 쿼리 파라미터 지원)
- [ ] `GET  /api/jobs/{id}/results/{cand_id}` — 개별 후보 상세 (raw CSV, config JSON, 명령어)
- [ ] `GET  /api/jobs/{id}/download` — zip 일괄 다운로드
- [ ] `DELETE /api/jobs/{id}` — 작업 취소/삭제
- [ ] `WS   /ws/jobs/{id}` — 실시간 진행률 / 완료된 후보별 메트릭 push
- [ ] 카탈로그 조회: `GET /api/catalog/hardware`, `/api/catalog/models`

#### 3.2 Job 관리
- [ ] 백그라운드 워커 (FastAPI BackgroundTasks 또는 Celery/RQ; SQLite로 시작하면 BackgroundTasks로 충분)
- [ ] 작업 상태 DB 스키마: `jobs(id, spec_json, status, progress, created_at, finished_at)`, `candidates(job_id, cand_id, config_json, result_json, score, on_pareto)`
- [ ] 결과 캐시: 동일 spec 재요청 시 즉시 반환 (해시 키)

#### 3.3 보안 / 운영 (최소 수준)
- [ ] 입력 검증 (Pydantic), 임의 경로 주입 방지
- [ ] 동시 실행 작업 수 / 작업당 후보 수 상한
- [ ] 로깅: 작업별 로그 파일 분리

### Phase 4 — 웹 프론트엔드
> 기존 webapp의 스택을 최대한 재사용 (예: Jinja + vanilla JS, 또는 React).

#### 4.1 입력 페이지 (`/explore`)
- [ ] **자원 풀 섹션**: 하드웨어 타입 select + 수량 input (행 추가/삭제)
- [ ] **모델 / 워크로드 섹션**: 모델 dropdown, 데이터셋 업로드 또는 합성 트래픽 파라미터 폼
- [ ] **요구사항 섹션**: SLO 임계값(TTFT/TPOT/ITL p99), throughput floor, power ceiling
- [ ] **탐색 옵션 섹션**: 최대 조합 수, 병렬도, P/D 분리 허용, PIM/CXL 토글, prefix caching, 목적함수 가중치(슬라이더)
- [ ] **검증 미리보기**: "현재 설정으로 약 N개 조합이 생성될 예정" 추정 표시
- [ ] 사전 정의된 시나리오(preset) 저장/불러오기

#### 4.2 진행 모니터 페이지 (`/jobs/{id}`)
- [ ] 전체 진행률 바 (완료/전체)
- [ ] 현재 실행 중인 후보 리스트 + ETA
- [ ] 실시간으로 채워지는 산점도 (latency vs throughput, 색=power)
- [ ] 도착하는 후보부터 즉시 Pareto frontier 갱신
- [ ] 취소 버튼

#### 4.3 결과 대시보드 (`/jobs/{id}/results`)
- [ ] **Top‑N 테이블**: 순위, 하드웨어 구성 요약, TTFT/TPOT p99, throughput, power, score
- [ ] **Pareto frontier plot** (2D + 축 선택 가능, 3D 옵션)
- [ ] **레이더 차트**: 각 후보의 정규화된 메트릭 비교
- [ ] **목적함수 가중치 재조정 슬라이더**: 재시뮬레이션 없이 즉시 재랭킹
- [ ] **상세 모달**: 후보 클릭 시 cluster_config JSON 미리보기, raw CSV 다운로드, 재현 CLI 명령어 복사 버튼

#### 4.4 공통
- [ ] 다국어 (한/영) 기본 토글
- [ ] 다크모드 토글 (선택)

### Phase 5 — 통합, 문서화, 배포
- [ ] `README.md` 섹션 추가 / `webapp_dse/README.md` 작성
- [ ] Docker 통합: 기존 `docker.sh` 흐름 안에서 새 웹 서버도 함께 기동 가능하도록
- [ ] 예시 spec 파일 (`examples/dse/*.yaml`) 제공
- [ ] e2e 시나리오 1개: "H100 1~4, A6000 0~8 범위에서 Llama‑3.1‑70B를 TTFT p99 ≤ 500ms, power ≤ 3kW로 서빙하는 최적 조합 5개"를 처음부터 끝까지 실행하는 데모 스크립트

---

## 5. 주요 기능 명세 (요약)

| # | 기능 | 비고 |
|---|------|------|
| F1 | 가용 자원 풀 입력 (하드웨어 × 수량) | Phase 1.1, 4.1 |
| F2 | 모델/워크로드 입력 | Phase 4.1 |
| F3 | SLO/throughput/power 제약 입력 | Phase 1.4, 4.1 |
| F4 | 조합 자동 생성 (cartesian + pruning) | Phase 1.1 |
| F5 | cluster_config JSON 자동 빌드 | Phase 1.2 |
| F6 | 다수 시뮬레이션 병렬 실행 | Phase 1.3 |
| F7 | 실시간 진행률 / 결과 스트리밍 | Phase 3.1, 4.2 |
| F8 | SLO 필터 + Pareto frontier | Phase 1.4 |
| F9 | 사용자 가중치 기반 Top‑N 랭킹 | Phase 1.4, 4.3 |
| F10 | 가중치 재조정 즉시 재랭킹 (시뮬레이션 재실행 없음) | Phase 4.3 |
| F11 | 결과 시각화 (Pareto / 레이더 / 시계열) | Phase 4.3 |
| F12 | cluster_config, CSV, CLI 명령어 export | Phase 3.1, 4.3 |
| F13 | 작업 저장 / 재현 / 비교 | Phase 3.2 |
| F14 | 결과 캐시 | Phase 3.2 |

---

## 6. 데이터 모델 / API 스펙 (초안)

### 6.1 입력 스펙 (YAML/JSON)
```yaml
resource_pool:
  - hw: H100
    min: 0
    max: 4
  - hw: A6000
    min: 0
    max: 8
model:
  name: meta-llama/Llama-3.1-70B
  fp: 16
workload:
  dataset: dataset/sharegpt_req100_rate10_llama.jsonl   # 또는 synthetic 블록
  num_req: 200
constraints:
  ttft_p99_ms: 500
  tpot_p99_ms: 80
  throughput_min_tok_s: 1500
  power_max_w: 3000
features:
  allow_pd_disagg: true
  allow_pim: false
  allow_cxl: false
  prefix_caching: true
search:
  max_combinations: 64
  parallelism: 4
  objective_weights:
    ttft: 0.3
    tpot: 0.2
    throughput: 0.3
    power: 0.2
  top_n: 5
```

### 6.2 출력 결과 (per candidate)
```json
{
  "candidate_id": "c042",
  "hardware": { "H100": 2, "A6000": 4 },
  "parallelism": { "TP": 2, "PP": 1, "DP": 3 },
  "pd_disagg": { "prefill_nodes": 2, "decode_nodes": 4 },
  "metrics": {
    "ttft_p99_ms": 412,
    "tpot_p99_ms": 64,
    "itl_p99_ms": 71,
    "throughput_tok_s": 1820,
    "power_w": 2640,
    "energy_per_req_j": 18.4
  },
  "meets_slo": true,
  "on_pareto": true,
  "score": 0.812,
  "cluster_config_path": "/jobs/<id>/configs/c042.json",
  "raw_csv_path":       "/jobs/<id>/results/c042.csv",
  "reproduce_cmd": "python main.py --cluster-config ... --dataset ..."
}
```

---

## 7. 디렉토리 구조 (제안)

```
LLMServingSim/
├── webapp/                       # (기존, 그대로 둠)
└── webapp_dse/                   # ★ 신규
    ├── core/
    │   ├── generator.py
    │   ├── config_builder.py
    │   ├── runner.py
    │   ├── ranker.py
    │   └── schemas.py            # Pydantic 모델
    ├── server/
    │   ├── app.py                # FastAPI 진입점
    │   ├── routes/
    │   │   ├── jobs.py
    │   │   ├── catalog.py
    │   │   └── ws.py
    │   ├── db.py                 # SQLite + SQLModel/SQLAlchemy
    │   └── workers.py
    ├── frontend/
    │   ├── templates/            # Jinja or built React
    │   └── static/
    ├── cli.py
    ├── examples/
    │   └── spec_llama70b.yaml
    ├── tests/
    │   ├── test_generator.py
    │   ├── test_builder.py
    │   ├── test_runner.py
    │   └── test_ranker.py
    └── README.md

docs/dse/
├── 00_existing_webapp_notes.md
├── 01_cluster_config_schema.md
├── 02_main_io_spec.md
└── 03_catalog.yaml
```

---

## 8. 기술 스택 (제안)
- **언어**: Python 3.10+
- **백엔드**: FastAPI + Uvicorn (Phase 0에서 기존 webapp이 Flask로 확인되면 Flask로 통일 검토)
- **DB**: SQLite + SQLModel (단순)
- **작업 큐**: FastAPI BackgroundTasks → 부하 증가 시 RQ/Celery로 이관
- **프론트엔드**: Jinja + HTMX + Plotly.js (의존성 최소) — 기존 webapp 스택에 맞추는 것이 1순위
- **테스트**: pytest, hypothesis(조합 생성 검증)
- **포맷터**: ruff + black

---

## 9. 마일스톤 및 검증 기준

| M | 목표 | 검증 기준 |
|---|------|----------|
| M0 | 사전 조사 완료 | `docs/dse/` 4개 문서 작성, cluster_config 스키마가 실제 예시들과 1:1 매칭 |
| M1 | CLI 만으로 end‑to‑end 동작 | `python -m webapp_dse.cli explore --spec examples/spec_llama70b.yaml` 실행 시 Top‑5 결과 JSON 출력, 시뮬레이션 호출이 실제로 일어남 |
| M2 | 회귀 테스트 통과 | 동일 cluster_config에 대해 기존 main.py 단독 실행 결과와 본 도구 경유 결과의 메트릭이 동일 |
| M3 | 웹 백엔드 동작 | API 6종 정상, WebSocket으로 진행률 push, 동시 2개 job 실행 시 충돌 없음 |
| M4 | 웹 프론트엔드 동작 | 입력 → 진행 모니터 → 결과 대시보드 흐름이 브라우저에서 완결, 가중치 슬라이더로 즉시 재랭킹 |
| M5 | 데모 시나리오 | §4 Phase 5의 e2e 시나리오가 README의 명령어 하나로 재현 가능 |

---

## 10. 리스크 및 향후 확장

### 10.1 리스크
- **조합 수 폭발**: cartesian product가 수천 개로 커질 수 있음 → 사전 필터 + 샘플링 + 사용자에게 상한 강제
- **시뮬레이션 시간**: 큰 모델/긴 워크로드는 1회당 수 분 이상 → 워크로드 sub‑sampling 옵션, 또는 빠른 surrogate 모델(추후)
- **메모리/디스크 사용량**: 후보당 CSV·trace 누적 → 작업 종료 시 정리 + 보관 정책
- **기존 cluster_config 스키마의 미문서화 필드**: Phase 0에서 충분히 시간을 확보. 불확실한 필드는 기본값으로 고정 후 점진 확장.

### 10.2 향후 확장
- **Surrogate model**: 일부 후보만 시뮬레이션하고 나머지는 ML로 예측 (탐색 가속)
- **비용 모델**: 클라우드 인스턴스 가격 / 자체 운영비 시나리오
- **다중 워크로드 동시 최적화**: 여러 트래픽 패턴에 대해 robust한 조합 선택
- **유전 알고리즘 / Bayesian Optimization** 기반 탐색기 (현재의 enumerator를 대체 가능한 모듈식 구조로 설계)
- **실시간 협업**: 동일 job을 여러 사용자가 공유하며 가중치 토론

---

## 부록 A. Claude Code 첫 작업 추천 순서

1. Phase 0의 §2.1 ~ §2.4를 차례대로 수행 → `docs/dse/` 4개 문서 작성
2. `webapp_dse/core/schemas.py` 작성 (입력 스펙 Pydantic 모델)
3. `webapp_dse/core/generator.py` MVP (cartesian + 메모리 필터만)
4. `webapp_dse/core/config_builder.py` MVP (1개 템플릿 기반)
5. `webapp_dse/core/runner.py` MVP (단일 실행 + CSV 파싱)
6. `examples/spec_llama70b.yaml` 작성 후 CLI로 end‑to‑end 1회 성공
7. 이후 Phase 2 → 3 → 4 순서로 진행
