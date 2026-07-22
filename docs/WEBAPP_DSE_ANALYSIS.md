# webapp/dse 워크플로우 분석

> 작성일: 2026-05-29  
> 분석 대상: `webapp/dse/` 전체 + 연관 모듈 (`webapp/parser.py`, `webapp/enumerate.py`, `webapp/hardware_catalog.py`, `docs/dse/03_catalog.yaml`)

---

## 1. 전체 구조 및 페이지 흐름

DSE(Design Space Exploration) 도구는 사용자가 보유한 하드웨어 풀과 모델을 지정하면 가능한 모든 클러스터 구성(Parallelism + P/D 분리 + 하드웨어 조합)을 자동으로 열거하고 시뮬레이션한 뒤 사용자가 지정한 목표 메트릭 기준으로 랭킹하는 시스템이다.

```
사용자                     프론트엔드                     백엔드
  │                           │                              │
  ├─ /dse ─────────────────► Explore 페이지                 │
  │   리소스풀/모델/워크로드    (dse_explore.html + .js)      │
  │   설정, Dry-run 미리보기                                  │
  │                           │                              │
  ├─ Estimate count ────────► POST /api/dse/dry-run ────────► dry_run_detail()
  │                           ◄── 후보 목록 + 개수 ──────────
  │                           │                              │
  ├─ Start Exploration ─────► POST /api/dse/jobs ───────────► _execute_job() (BackgroundTask)
  │   ◄── job_id ────────────                                │
  │                           │                              │ 백그라운드:
  ├─ /dse/jobs/{id} ───────► Progress 페이지                 │  generate_candidates()
  │   실시간 SSE 업데이트       (dse_progress.html + .js)     │  run_dse_job()
  │   4개 scatter 차트                                        │    └─ run_sweep() (×retry)
  │   Candidates 테이블                                       │  rank_candidates()
  │                           │                              │
  ├─ Open Results ──────────► Results 페이지                 │
  │   Top-N 테이블             (dse_results.html + .js)      │
  │   Pareto 차트                                             │
  │   Radar 차트               │                              │
  │   Re-rank 슬라이더          └─ POST /api/dse/jobs/{id}/rerank
```

---

## 2. 디렉토리 구조

```
webapp/dse/
├── __init__.py
├── core/
│   ├── schemas.py          # 모든 Pydantic 모델 (입력/출력)
│   ├── generator.py        # ResourcePool → CandidateConfig 목록
│   ├── stage1_filters.py   # 시뮬레이션 전 분석적 사전 필터
│   ├── config_builder.py   # CandidateConfig → 클러스터 JSON
│   ├── runner.py           # 시뮬레이션 실행 + Retry 로직
│   └── ranker.py           # SLO 필터 + Pareto + 가중치 스코어링
└── server/
    └── routes.py           # FastAPI 라우트 (/api/dse/...)

webapp/
├── enumerate.py            # TP/PP/DP/P/D 조합 열거 (ConfigSpec 반환)
├── hardware_catalog.py     # llm_profile/ 디렉토리 스캔 → 카탈로그
├── parser.py               # 시뮬레이터 stdout 로그 + CSV → 메트릭 dict
├── runner.py               # 범용 sweep 실행기 (SSE, 상태 관리)
└── cluster_builder.py      # ConfigSpec → 클러스터 JSON dict

docs/dse/
└── 03_catalog.yaml         # HW 메타데이터 + 모델 파라미터 수동 관리

webapp/static/
├── dse_explore.js          # Explore 페이지 JS
├── dse_progress.js         # Progress 페이지 JS (SSE 소비자)
└── dse_results.js          # Results 페이지 JS

webapp/templates/
├── dse_explore.html
├── dse_progress.html
└── dse_results.html
```

---

## 3. 데이터 모델 (`core/schemas.py`)

DSE 전체 흐름의 입력과 출력을 Pydantic으로 정의한다.

### 3-1. 입력 모델 (`JobSpec`)

```
JobSpec
├── ResourcePool
│   ├── items: list[HwAllocation]   # 하드웨어별 min/max 수량
│   └── total_max_npus: int | None  # 전체 NPU 수 상한 (optional)
├── ModelSpec
│   ├── name: str                   # HuggingFace 모델 ID
│   └── fp: Literal[8, 16, 32]      # 정밀도
├── WorkloadSpec
│   ├── dataset: str | None         # 데이터셋 경로
│   ├── num_req: int                 # 요청 수
│   └── timeout_s: int | None       # 후보당 타임아웃 (초)
├── Constraints                     # SLO 조건 (None = 무제한)
│   ├── ttft_p99_ms / tpot_p99_ms / itl_p99_ms
│   ├── throughput_min_tok_s
│   ├── power_max_w / energy_max_wh
│   └── tokwh_min                   # 최소 에너지 효율 (tok/Wh)
├── FeatureFlags
│   ├── allow_pd_disagg             # P/D 분리 허용
│   ├── prefix_caching
│   └── attn_offloading
├── SearchConfig
│   ├── max_combinations: int       # 시뮬레이션 상한 (기본 20)
│   ├── sampling_strategy: "random"|"grid"
│   └── random_seed: int
├── ObjectiveWeights                # 가중치 (합산 후 자동 정규화)
│   ├── ttft / tpot / throughput / power / tokwh
└── top_n: int                      # 최종 상위 N개 선발
```

**`ObjectiveWeights` 자동 정규화**: `model_validator(mode="after")`에서 5개 가중치 합이 1이 되도록 나눈다. 모두 0이면 `ValueError` 발생.

### 3-2. 내부 중간 모델

| 모델 | 역할 |
|------|------|
| `CandidateConfig` | 열거된 후보 1개: ConfigSpec + hw_distribution + parallelism + pd_layout + label |
| `SimulationResult` | 시뮬레이션 결과 1개: state + metrics dict + ranker가 채운 meets_slo / on_pareto / score |
| `RankedResults` | `rank_candidates()` 출력: 전체 결과 + pareto_indices + top_n_indices + weights_used |

---

## 4. 카탈로그 시스템

DSE는 두 가지 카탈로그를 병행 사용한다.

### 4-1. 성능 프로파일 카탈로그 (`hardware_catalog.py`)

`llm_profile/perf_models/<hardware>/<vendor>/<model>/tp<N>/` 디렉토리 트리를 스캔한다.

```python
build_catalog() → dict[tuple[str, str], frozenset[int]]
# 키: (hardware, model)
# 값: 완전한 성능 데이터가 있는 TP 값 집합
```

`_tp_dir_is_complete(tp_dir)` 조건:
- `layers.csv` 존재
- `predictions/attn_prefill_prediction_dict.pkl` + `attn_decode_prediction_dict.pkl` 존재, 또는
- `predictions/attn_prefill_predictions.csv` + `attn_decode_predictions.csv` 존재

→ 예: RTX3090/Llama-3.1-8B는 디렉토리가 있어도 predictions가 없으면 catalog에서 제외됨.

이 카탈로그는 `enumerate_configs()`에서 특정 (hw, model, tp) 조합이 유효한지 판단하는 데 사용된다.

### 4-2. 하드웨어/모델 메타데이터 카탈로그 (`docs/dse/03_catalog.yaml`)

수동으로 관리하는 YAML. DSE 사전 필터와 Power template 생성에 사용된다.

**하드웨어 항목 (현재 정의된 5종)**:

| 항목 | A6000 | H100 | RNGD | RTX3090 | TPU-v6e-1 |
|------|-------|------|------|---------|-----------|
| mem_size_gb | 40 | 80 | 40 | 24 | 16 |
| mem_bw_gbs | 768 | 3350 | 1500 | 936 | 1640 |
| tdp_w | 300 | 700 | 150 | 350 | 190 |
| idle_power_w | 25 | 70 | 37.44 | 30 | 40 |

**모델 항목**: `params_b`, `weight_size_fp{8,16}_gb`, `num_key_value_heads`, `num_hidden_layers` 등 Stage 1 필터에 필요한 값들.

---

## 5. Candidate 생성 파이프라인

### 5-1. 전체 흐름

```
JobSpec
  │
  ├─ _enumerate_hw_counts()          # 하드웨어별 [min, max] 카르테시안 곱
  │    → list[dict[hw, count]]       # 총 합 > 0, total_max_npus 존중
  │
  ├─ _coarse_memory_prune()          # 빠른 사전 제거: 전체 HBM < 모델 가중치
  │    (집계 방식 — 샤드 단위 검사 전 단계)
  │
  ├─ _hw_counts_to_instance_groups() # hw_count 딕셔너리 → instance_group 리스트
  │
  ├─ enumerate_configs(scenario, catalog)  # TP/PP/DP/P/D 열거 (enumerate.py)
  │    → list[ConfigSpec]
  │
  ├─ _config_spec_to_candidate()     # ConfigSpec + hw_counts → CandidateConfig
  │
  ├─ label dedup                     # 동일 label이 여러 hw_count에서 나올 경우 첫 번째만 유지
  │
  ├─ apply_stage1_filters()          # 분석적 사전 필터 4종 (아래 §5-3 참조)
  │
  ├─ exclude_labels 제거             # retry 모드: 이미 시뮬레이션된 후보 제외
  │
  └─ _sample()                       # cap 초과 시 random/grid 샘플링
       → list[CandidateConfig]       # 최종 시뮬레이션 대상
```

### 5-2. enumerate.py 동작 상세

`enumerate_configs(scenario, catalog)` 는 단일 그룹(homogeneous)과 다중 그룹(heterogeneous) 두 경로로 나뉜다.

#### 단일 그룹 (homogeneous)

`_enum_single_group_combined()`: 주어진 budget(npu_count) 안에서
- TP: catalog에 실제 프로파일 있는 tp 값들 순회
- PP: 1부터 budget//tp까지
- DP: 1부터 budget//(tp×pp)까지, budget % (dp×tp×pp) == 0 조건 만족

`_enum_single_group_pd()`: P/D 분리 모드
- 알려진 제약: decode npu_num > 1 → ASTRA-Sim topology crash → 제외
- 알려진 제약: prefill npu_group > 1 (PP>1) → topology 불일치 → 제외
- topology_valid() 검사: 전체 NPU를 eff_num_inst로 나눌 수 없으면 deadlock 발생

A6000×4 기준 기대 출력 (11개):
- combined 9개: `tp1_pp1_dp1`, `tp2_pp1_dp1`, `tp1_pp2_dp1`, `tp2_pp2_dp1`, `tp1_pp4_dp1`, `tp1_pp1_dp2`, `tp2_pp1_dp2`, `tp1_pp2_dp2`, `tp1_pp1_dp4`
- P/D 2개: `pd_1p1d_tp1`, `pd_1p2d_tp1`

#### 다중 그룹 (heterogeneous)

- 그룹별 역할(combined/prefill/decode)의 카르테시안 곱을 시도
- heterogeneous combined-mode(서로 다른 hw가 모두 combined) → 제외  
  (ASTRA-Sim 균일 그리드 토폴로지가 두 compute rate를 표현 불가 → deadlock)
- 각 조합에 대해 `_topology_valid()` 검사 통과 필요
- 인스턴스 정렬: prefill(0) → combined(1) → decode(2) 순 — ASTRA-Sim NPU ID 배분 규칙

### 5-3. Stage 1 분석적 사전 필터 (`stage1_filters.py`)

dedup 이후, sampling 이전에 실행된다. 물리적으로 불가능한 후보를 시뮬레이션 전에 제거한다.

#### 필터 1: `filter_memory`
```
weight_shard_GB = total_weight_GB / (tp × pp)
limit_GB = min(HBM per NPU across all hw) × 0.85   # 15% 안전 여유
조건: weight_shard_GB ≤ limit_GB
```
집계 방식의 `_coarse_memory_prune`과 달리 TP/PP 샤딩을 고려한 per-NPU 검사.

#### 필터 2: `filter_kv_heads`
```
조건: num_key_value_heads % tp == 0
```
GQA(Grouped Query Attention) 구조에서 KV 헤드 수가 TP로 나뉘어야 한다.

#### 필터 3: `filter_roofline_decode`
```
TPOT_lb_ms = (weight_shard_GB / mem_bw_GB/s) × 1000
조건: TPOT_lb_ms ≤ tpot_slo_ms
```
decode는 배치 크기에 관계없이 매 토큰 생성마다 weight 전체를 HBM에서 스트리밍한다(memory-bandwidth bound). 이 하한값이 SLO보다 크면 어떤 최적화로도 SLO를 충족할 수 없다.

#### 필터 4: `filter_power`
```
total_TDP_W = sum(count × tdp_w for hw, count in hw_counts)
조건: total_TDP_W ≤ power_max_w
```

#### 필터 5: `filter_tokwh_roofline`
```
tok/s_ub   = mem_bw_GB/s / weight_shard_GB     # 메모리 BW bound 최대 처리량
tok/Wh_ub  = tok/s_ub × 3600 / TDP_W
조건: tok/Wh_ub ≥ tokwh_min
```
주의: TP를 늘리면 tok/s_ub는 2배가 되지만 TDP도 2배가 되어 tok/Wh_ub는 불변. PP를 늘리면 shard_GB가 줄어 tok/s_ub 증가 → tok/Wh_ub 개선.

---

## 6. Cluster JSON 빌드 (`core/config_builder.py`)

`write_candidate_cluster_json(candidate, output_dir, hw_meta)`:
1. `build_power_template_from_catalog()` → 03_catalog.yaml의 idle/standby/active_power 값으로 power 블록 구성
2. `webapp.cluster_builder.build_cluster_json(config_spec, ...)` → 실제 cluster JSON dict 생성
3. `output_dir/{label}.json`으로 저장

**Power template 구조**:
```json
{
  "base_node_power": 60,
  "npu": {"A6000": {"idle_power": 25, "standby_power": 115, "active_power": 300, "standby_duration": 18}},
  "cpu": {"idle_power": 10, "active_power": 200, "util": 0.15},
  "dram": {"dimm_size": 32, "idle_power": 2.0, "energy_per_bit": 6.0},
  ...
}
```

`dry_run_validate()`: `inference_serving.config_builder.build_cluster_config()` dry-run 호출로 ASTRA-Sim YAML 생성을 검증 (main.py처럼 cwd를 astra-sim/로 변경). 현재 runner에서 미사용이나 미래 검증 단계 확장 용도.

---

## 7. 시뮬레이션 실행 (`core/runner.py`)

### 7-1. 기본 실행 흐름

`run_dse_job(job_id, candidates, spec, job_dir)`:

1. **Power template 생성**: 모든 후보에 나타나는 hw의 union으로 단일 power template 구성
2. **workload dict 구성**: dataset 경로, num_req, timeout_s 포함
3. **`webapp.runner.run_sweep()` 호출**: 
   - asyncio.Semaphore(MAX_CONCURRENT)로 병렬 실행 제어
   - 각 ConfigSpec에 대해 main.py를 subprocess로 실행
   - PID_TAG를 통한 임시 파일 격리 (경합 없는 trace/workload 파일 경로)
   - 완료 시 stdout log + CSV 파싱 → status.json에 metrics 저장
   - SSE 이벤트 broadcast

### 7-2. Retry 메커니즘

최대 `_MAX_RETRY_ROUNDS = 3` 라운드 반복:

```
round 1~3:
  ├─ status.json에서 failed/cancelled 후보 수(N) 집계
  ├─ N == 0 → 종료
  ├─ generate_candidates(exclude_labels=tried_labels, override_max=N) 호출
  │    → 아직 시도되지 않은 후보 공간에서 N개 신규 후보 생성
  ├─ 신규 후보를 candidates.json에 추가 (Progress UI가 새 행으로 표시)
  └─ run_sweep(merge=True) → status.json에 병합
```

모든 retry 라운드 완료 후 `finalize_sweep()` 한 번 호출 → SSE 스트림 종료 신호.

### 7-3. 결과 수집 (`_collect_results`)

- `status.json`의 `configs[label]` 항목에서 state, elapsed_s, metrics 읽기
- metrics가 비어 있으면(구 버전 status.json) `parse_run(log, csv)` 재파싱
- 각 후보를 `SimulationResult` 객체로 변환

---

## 8. 결과 파싱 (`webapp/parser.py`)

### 8-1. `parse_log(log_path)`

simulator stdout에서 정규식으로 메트릭 추출. 주요 키:

| 키 | 내용 |
|----|------|
| `total_token_tp` | 전체 토큰 처리량 (tok/s) |
| `p99_ttft_ms` | P99 TTFT (ms) |
| `p99_tpot_ms` | P99 TPOT (ms) |
| `p99_itl_ms` | P99 ITL (ms) |
| `total_energy_wh` | 총 에너지 (Wh, kJ에서 변환) |
| `avg_power_w` | 평균 전력 (total_energy_wh × 3600 / total_latency_s, 파생값) |
| `tok_per_wh` | 에너지 효율 (total_token_tp × latency / total_energy_wh) |

### 8-2. `parse_csv(csv_path)`

per-request CSV(TTFT/TPOT/ITL 열)에서 P99 통계 계산. 단위는 ns → ms 변환.

| 키 | 내용 |
|----|------|
| `ttft_p99_ms` | P99 TTFT (CSV 기반) |
| `tpot_p99_ms` | P99 TPOT (CSV 기반) |
| `itl_p99_ms` | P99 ITL (CSV 기반) |

### 8-3. 키 이름 이중화 (`parse_run`)

log와 CSV는 동일한 지표를 다른 키 이름으로 저장한다:

| 지표 | log 키 | CSV 키 |
|------|--------|--------|
| TTFT P99 | `p99_ttft_ms` | `ttft_p99_ms` |
| TPOT P99 | `p99_tpot_ms` | `tpot_p99_ms` |
| ITL P99 | `p99_itl_ms` | `itl_p99_ms` |

`parse_run()` 는 두 키를 모두 채워 어느 소스에서 왔든 동일하게 접근할 수 있도록 alias를 추가한다. (ranker.py는 `p99_*` 형식, Progress UI는 `*_p99_ms` 형식 사용)

---

## 9. 랭킹 파이프라인 (`core/ranker.py`)

`rank_candidates()` = `filter_slo` → `pareto_frontier` → `compute_scores` → `top_n` 순서로 실행.

### 9-1. SLO 필터 (`filter_slo`)

`SimulationResult.meets_slo` 를 in-place로 설정한다. state != "done"이면 자동으로 False.
- Constraints에 None이 아닌 필드만 검사 (None = 무제한)
- 제거하지 않고 플래그만 설정 → Pareto 계산 대상에서 제외되지만 all_candidates에 남아 있음

### 9-2. Pareto Frontier

기본 axes: `(p99_ttft_ms, min)`, `(total_token_tp, max)`, `(total_energy_wh, min)`.

**missing metric 처리**: eligible 후보들 중 특정 메트릭이 하나도 없으면 해당 axis를 제외하고 계산. 단 한 후보라도 그 메트릭이 없으면 해당 후보를 Pareto 계산에서 제외.

**지배 조건 (`_dominates`)**: a가 b를 지배하려면 모든 axis에서 a ≥ b(같거나 좋고), 최소 하나의 axis에서 a > b(엄격히 좋음).

### 9-3. 가중치 스코어링 (`compute_scores`)

```python
_OBJECTIVES = {
    "ttft":       ("p99_ttft_ms",     "min"),
    "tpot":       ("p99_tpot_ms",     "min"),
    "throughput": ("total_token_tp",  "max"),
    "power":      ("total_energy_wh", "min"),
    "tokwh":      ("tok_per_wh",      "max"),
}
```

각 objective별 min-max 정규화 → [0,1] (1=best):
- min 방향: `(hi - v) / span`
- max 방향: `(v - lo) / span`
- span == 0 (모두 동일값): 1.0 부여

`score = Σ(weight_obj × normalized_obj)` — 이미 합산 정규화된 가중치를 적용.

어떤 후보에서 메트릭이 누락된 objective는 해당 후보의 normalized score = 0.0으로 처리.

### 9-4. Top-N 선발 (`top_n`)

점수 내림차순 정렬 후 diversity 옵션 활성화 시 동일 hw_distribution signature는 한 번만 카운트. 즉, A6000×4로 tp1_pp4_dp1와 tp2_pp2_dp1이 1위, 2위여도 같은 hw signature면 그 중 1개만 선발됨.

---

## 10. API 라우트 (`server/routes.py`)

### 10-1. 엔드포인트 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/dse/catalog` | 하드웨어 + 모델 + 가용성 정보 반환 |
| POST | `/api/dse/dry-run` | 후보 목록 미리보기 (시뮬레이션 없음) |
| POST | `/api/dse/jobs` | 새 DSE 작업 생성 + 백그라운드 실행 |
| GET | `/api/dse/jobs` | 작업 목록 조회 |
| GET | `/api/dse/jobs/{id}` | 특정 작업 상태 조회 |
| GET | `/api/dse/jobs/{id}/results` | 완료된 작업 결과 조회 |
| POST | `/api/dse/jobs/{id}/rerank` | 기존 결과로 가중치만 바꿔 재랭킹 |
| DELETE | `/api/dse/jobs/{id}` | 작업 취소 + soft-delete |
| GET | `/api/dse/jobs/{id}/events` | SSE 스트림 (실시간 진행 상황) |
| GET | `/api/dse/jobs/{id}/download.zip` | 전체 아티팩트 ZIP 다운로드 |

### 10-2. 작업 생성 흐름 (`POST /api/dse/jobs`)

1. spec_hash 계산 (weights, top_n 제외 — 재랭킹에만 영향)
2. 동일 hash의 완료된 작업 있으면 캐시 히트 → 즉시 반환
3. `job_dir` 생성, `spec.json` + `spec_hash.txt` 저장
4. `dry_run_detail()` 호출 → 예상 시뮬레이션 수 반환
5. `BackgroundTasks.add_task(_execute_job)` 등록 → HTTP 200 즉시 반환

### 10-3. SSE 이벤트 프로토콜 (`GET /api/dse/jobs/{id}/events`)

```
event: snapshot    → 연결 직후 status.json 전체 스냅샷 (기존 완료 상태 복원용)
data: {label, state, metrics, elapsed_s, ...}  → 개별 후보 완료/실패 알림
data: {"type": "heartbeat"}                     → 5초 idle 시 keepalive
data: {"sweep_state": "done"|"failed"}          → 스트림 종료 신호
```

- 한 연결에 snapshot 이벤트 1회, 이후는 모두 `data:` 이벤트
- 종료 신호 수신 시 클라이언트에서 `EventSource.close()` 호출

### 10-4. 파일시스템 레이아웃

```
output/dse_jobs/<job_id>/
├── spec.json          # 원본 JobSpec
├── spec_hash.txt      # 캐시 키
├── status.json        # webapp.runner가 관리하는 실시간 상태
│                      #  {state, configs: {label: {state, metrics, ...}}}
├── candidates.json    # 열거된 후보 목록 (slim — config_spec 제외)
├── all_candidates.json  # 랭킹 후 전체 SimulationResult 목록
├── top_n.json           # 상위 N 결과
├── pareto.json          # Pareto-optimal 결과
└── runs/
    ├── <label>.log    # 시뮬레이터 stdout
    └── <label>.csv    # per-request 지연 시간 CSV
```

---

## 11. 프론트엔드 페이지별 동작

### 11-1. Explore 페이지 (`dse_explore.html` + `dse_explore.js`)

**초기화 시**:
1. `GET /api/dse/catalog` → 하드웨어 드롭다운, 모델 드롭다운 채우기
2. `GET /api/datasets` → 데이터셋 드롭다운 채우기
3. 하드웨어 행 1개 자동 추가 (`addHwRow`)

**Estimate count (Dry-run)**:
- `collectSpec()` → JobSpec 수집
- `POST /api/dse/dry-run` → 응답에서 후보 목록 테이블 렌더링
  ```
  후보: candidate label | 하드웨어 | TP | PP | DP | P/D Layout | 시뮬레이션 여부(✓/—)
  ```
- 샘플링 대상이 아닌 후보는 opacity 0.45로 흐리게 표시

**가중치 수집 (`collectSpec`)**:
- 각 성능 지표(TTFT/TPOT/Throughput/Tok-Wh/Energy)에 체크박스 + Low/Med/High 라디오
- 체크 안 됨 → weight 0 (랭킹 제외)
- Low=1, Medium=3, High=9 (상대적 비율; ObjectiveWeights에서 자동 정규화)
- 모두 체크 안 된 경우 → 모두 1로 fallback

### 11-2. Progress 페이지 (`dse_progress.html` + `dse_progress.js`)

**초기화 시**:
1. `GET /api/dse/jobs/{id}/results` → `candidates_meta` 로드 (hw/parallelism 정보)
2. `GET /api/dse/jobs/{id}` → status.json의 기존 상태로 테이블 초기화
3. Plotly 4개 차트 초기화
4. `EventSource` 연결

**SSE 처리 (`applyUpdate(ev)`)**:
- 새 label이면 테이블에 행 추가
- state/metrics/elapsed 업데이트
- state == "done" + 메트릭 있으면 `livePoints[label]` 업데이트 → `redrawPlot()`
- 진행 바 업데이트

**retry 후보 처리**:
- retry 라운드에서 새로 생긴 label이 candidatesMeta에 없을 때 300ms 디바운스 후 재조회

**차트 4종** (모두 Throughput을 Y축으로 공유):
1. TTFT p99 vs Throughput
2. Energy (Wh) vs Throughput
3. TPOT p99 vs Throughput
4. Tokens/Wh vs Throughput

**클릭/호버 연동**: `wirePlot()` 로 모든 차트에 동일한 핸들러 등록 → 어느 차트에서 클릭해도 같은 label의 테이블 행이 하이라이트됨.

**테이블 정렬**: 컬럼 헤더 클릭 시 asc/desc 토글. 결측값("—")은 항상 하단으로.

### 11-3. Results 페이지 (`dse_results.html` + `dse_results.js`)

**초기화 시**:
- `GET /api/dse/jobs/{id}/results` → all_candidates, top_n, pareto, candidates_meta 로드
- Top-N 테이블 렌더링
- Pareto scatter plot 그리기 (X/Y 축 선택 가능)
- Top-N Radar chart 그리기

**Re-rank**:
- 슬라이더 5개 (TTFT/TPOT/Throughput/Energy/Tok-Wh) 0-100 조정
- `POST /api/dse/jobs/{id}/rerank` → 새 weights + top_n 전송
- 응답으로 받은 새 top_n, pareto로 UI 갱신 (시뮬레이션 재실행 없음)

**Pareto 차트 축 옵션**: `p99_ttft_ms`, `p99_tpot_ms`, `total_energy_wh` (X), `total_token_tp`, `total_energy_wh` (Y)

---

## 12. 주요 데이터 흐름 요약

```
사용자 입력 (JobSpec)
    │
    ▼
ResourcePool × hw_counts 카르테시안 곱
    │
    ▼ (_coarse_memory_prune — 빠른 제거)
enumerate_configs(scenario, catalog)
    │  ├─ 단일 HW: combined + P/D 모드
    │  └─ 다중 HW: 역할 조합 × 레이아웃 카르테시안 곱
    ▼
CandidateConfig 목록 (label dedup 후)
    │
    ▼ (apply_stage1_filters — 분석적 검증)
    │  ① memory shard 적합성 (85% HBM 기준)
    │  ② GQA kv_heads % tp == 0
    │  ③ TPOT roofline 하한 vs SLO
    │  ④ 총 TDP vs power 예산
    │  ⑤ tok/Wh roofline 상한 vs 최소 효율
    │
    ▼ (_sample — max_combinations 초과 시)
시뮬레이션 대상 최종 후보 목록
    │
    ▼ (run_dse_job)
    │  ├─ write_candidate_cluster_json (cluster JSON 파일)
    │  ├─ run_sweep (ASTRA-Sim subprocess × N 병렬)
    │  │   └─ SSE broadcast → Progress UI 실시간 업데이트
    │  └─ retry_loop (실패 후보 교체, 최대 3회)
    │
    ▼ (_collect_results)
SimulationResult 목록 (metrics: p99_ttft_ms, total_token_tp, total_energy_wh, tok_per_wh, ...)
    │
    ▼ (rank_candidates)
    │  ├─ filter_slo (meets_slo 플래그)
    │  ├─ pareto_frontier (on_pareto 플래그)
    │  ├─ compute_scores (min-max 정규화 + 가중 합산)
    │  └─ top_n (score 내림차순, hw diversity)
    │
    ▼
RankedResults → all_candidates.json / top_n.json / pareto.json
    │
    ▼
Results 페이지 (Top-N 테이블, Pareto 차트, Radar 차트, Re-rank)
```

---

## 13. 알려진 제약사항

| 제약 | 내용 | 위치 |
|------|------|------|
| decode npu_num > 1 금지 | ASTRA-Sim topology crash | `enumerate.py:166` |
| prefill PP > 1 금지 | topology 불일치로 deadlock | `enumerate.py:163` |
| heterogeneous combined 금지 | 다른 HW의 compute rate를 균일 그리드로 표현 불가 | `enumerate.py:418` |
| P/D TTFT 정의 차이 | CLAUDE.md: 첫 토큰 계산 완료 시점 (vLLM의 TTFT보다 낮음) | `CLAUDE.md` |
| 시간 단위 | 시뮬레이터 내부: ns (1GHz clock) | `CLAUDE.md` |
| RTX3090 | predictions 없어 catalog 제외 | `03_catalog.yaml` |
| Stage 1 tokwh 필터 | TP 증가 → tok/Wh_ub 불변 (처리량 ↑ & TDP ↑ 상쇄) | `stage1_filters.py` |
