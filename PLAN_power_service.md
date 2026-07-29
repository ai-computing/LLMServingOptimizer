# PLAN_power_service.md — 전력 최적 LLM 서빙 자원 추천 서비스

> **목표**: 이종 GPU/NPU 클러스터에서 사용자가 「모델 + SLO + 서비스 규모」를 제출하면,
> 가용 자원 중 **SLO·규모를 만족하는 최저 전력 자원 조합**을 도출하고 예약하는
> **다중 테넌트 상시 서비스**를 `LLMServingOptimizer` 위에 구현한다.
> **설계 근거 문서**: `LLM_서빙_전력최적화_서비스_설계서_v2.docx` (§ 번호는 그 문서를 가리킴)

---

## 0. 문서 정보

| 항목 | 내용 |
|---|---|
| 대상 저장소 | `ai-computing/LLMServingOptimizer` (main) |
| 신규 모듈 | `service/`, `sim_backends/measured/`, `profiles/power/` |
| 수정 모듈 | `planner/spec_schema.py`, `planner/milp_solver.py`, `planner/objective.py`, `sim_backends/__init__.py`, `webapp/` (라우터 추가만) |
| 금지 사항 | `backends/*` (submodule) 내부 수정 금지. 기존 planner/DSE의 공개 동작(기존 spec으로 실행) 회귀 금지 |
| 테스트 러너 | `pytest` — 마커 3종: `unit`(기본, 시뮬 불필요), `sim`(시뮬 백엔드 필요), `hw`(실기기 필요, CI 제외) |

**구현 순서는 M0→M10.** 각 마일스톤은 "구현 → 테스트 → DoD 체크" 순으로 완료해야
다음으로 넘어간다. 테스트가 실패한 채로 다음 마일스톤을 시작하지 않는다.

```
# 공통 테스트 명령 (마커 등록은 M0에서)
pytest -m unit                      # 매 커밋
pytest -m "unit or sim" planner/tests/ tests/ service/tests/   # 마일스톤 완료 시
```

---

## 1. 설계 원칙 (모든 마일스톤 공통)

1. **비침습**: 시뮬레이터 submodule은 블랙박스. 새 기능은 어댑터(`sim_backends/`)와
   상위 모듈에만 추가한다.
2. **SimBackend 인터페이스 유지**: 제3 백엔드 `measured`도 기존
   `get_backend(name) -> SimBackend` 계약(`run`, `parse_csv`, `parse_stdout`,
   `list_hardware`)을 그대로 구현한다. 호출측(planner/webapp)은 백엔드 이름 외에
   아무것도 몰라야 한다.
3. **관심사 분리**: 인스턴스 성능의 출처(측정 오라클 vs 시뮬)와 클러스터 합성
   (이벤트 시뮬)을 분리한다(§5.1).
4. **Stage-2가 최종 판정자**: Stage-1 프록시는 후보 생성용. SLO·전력의 최종 판정은
   Fidelity Router가 고른 백엔드의 Stage-2 결과로만 한다.
5. **infeasible은 1급 결과**: 실패 시 병목 제약과 완화 시나리오를 구조화해 반환한다.

---

## 2. 신규 디렉토리 구조

```
LLMServingOptimizer/
├── profiles/power/                  # M1: 전력 프로파일 (하드웨어별 YAML)
│   ├── schema.md
│   ├── A40.yaml  A5000.yaml  H100.yaml  RNGD.yaml ...
├── planner/
│   ├── power_profiles.py            # M1: 로더/검증 (3계층 폴백: measured>sim>const)
│   └── (spec_schema/milp_solver/objective 수정)   # M2
├── service/                         # 서비스 계층 (신규 최상위)
│   ├── __init__.py
│   ├── workload_synth.py            # M3: 규모→합성 jsonl
│   ├── presets.py                   # M3: chat/summarize/agentic 길이분포 프리셋
│   ├── inventory/                   # M8
│   │   ├── registry.py              #   cluster_registry.yaml 로더
│   │   ├── ledger.py                #   SQLite 할당 원장 + optimistic locking
│   │   └── views.py                 #   가용 서브토폴로지 → planner topology 직렬화
│   ├── fidelity_router.py           # M9: 후보→평가 백엔드 선택 정책 (§5.4)
│   ├── api/                         # M9: FastAPI 라우터 (webapp에 mount)
│   │   ├── routes.py
│   │   └── models.py                #   pydantic 요청/응답 스키마
│   └── tests/                       # 서비스 계층 pytest
├── sim_backends/measured/           # M5–M7: 제3 백엔드
│   ├── __init__.py                  #   MeasuredBackend(SimBackend)
│   ├── oracles/
│   │   ├── base.py                  #   Oracle 프로토콜 (아래 §5 참조)
│   │   ├── capacity_curve.py        #   공통 폴백 오라클
│   │   ├── npu_bucket.py            #   LENS식 버킷 오라클
│   │   └── gpu_iteration.py         #   (P2, 스텁만)
│   ├── event_sim.py                 #   이벤트 기반 클러스터 시뮬
│   ├── campaign/
│   │   ├── capacity_campaign.py     #   실측 캠페인 러너 (bench/ 재사용)
│   │   └── bucket_campaign.py       #   NPU 버킷 측정 러너
│   └── tests/
├── scripts/compare_backends.py      # M7: 3자 대조 리포트
└── examples/service/                # E2E 예제 spec
```

---

## 3. 마일스톤

### M0. 스캐폴딩 + 테스트 인프라

**구현**
- 위 디렉토리·빈 모듈 생성, `service/`와 `sim_backends/measured/`에 `tests/` 포함.
- `pyproject.toml`(또는 `pytest.ini`)에 마커 등록:
  `unit`(default), `sim`, `hw`. `addopts = -m unit`로 기본은 unit만 돌게.
- `requirements-service.txt`: fastapi, uvicorn, pydantic>=2, pyyaml, (기존 planner 의존 재사용).

**테스트 (DoD)**
- [ ] `pytest -m unit` 이 기존 스위트 포함 전부 green (신규 모듈은 import smoke test만).
- [ ] `python -c "import service, sim_backends.measured"` 성공.
- [ ] 기존 회귀 없음: `pytest planner/tests/ tests/` green.

---

### M1. 전력 프로파일 스키마 + 로더 (§4.2)

**구현**
- `profiles/power/<HW>.yaml` 스키마:
  ```yaml
  device: {name: A40, active_w: 300, idle_w: 60, mem_gb: 48}
  host_overhead: {base_w: 250, per_device_w: 15}
  measured:            # 선택
    - {model: "meta-llama/Llama-3.1-70B", tp: 4, load: 0.8, avg_w: 1180}
  meta: {source: "nvidia-smi|datasheet|estimate", measured_at: "2026-07-01", stack: "vllm-0.19"}
  ```
- `planner/power_profiles.py`:
  - `load_power_profile(hw: str) -> PowerProfile` (pydantic 검증)
  - `effective_power_w(hw, model, tp, load) -> tuple[float, str]`
    — 반환 (전력, 출처) with 3계층 폴백: measured 테이블(최근접 load 보간) →
    시뮬 상수 없음 시 `device.active_w` → 프로파일 파일 자체가 없으면
    기존 `milp_solver._HW_ACTIVE_POWER` 상수 + 경고 로그.
- `milp_solver`의 `_HW_ACTIVE_POWER` 참조를 `power_profiles` 경유로 교체
  (동작 동일해야 함 — 프로파일 파일이 없으면 기존 상수 그대로).

**테스트**
- `planner/tests/test_power_profiles.py` (unit):
  - [ ] 스키마 위반 YAML(음수 전력, 필수 필드 누락) → ValidationError.
  - [ ] measured 보간: load 0.5/1.0 두 점 → 0.75 질의 시 선형 보간값.
  - [ ] 폴백 체인: measured 없음 → active_w; 파일 없음 → 레거시 상수 + 경고.
  - [ ] **회귀**: 프로파일 파일이 하나도 없을 때 기존 `test_milp_solver.py` 전부 green
        (해가 바뀌지 않아야 함).

---

### M2. Planner power-min 모드 + 수요 제약 (§4.1) ★ 핵심

**구현**
- `planner/spec_schema.py`:
  - `VALID_OBJECTIVE_METRICS`에 `"power_w"` 추가 (direction: min만 허용).
  - `Requirements`에 `demand: {toks_per_s: float} | {req_per_s: float, preset|len_dist}` 추가
    (req_per_s는 M3의 환산기로 toks_per_s 변환).
- `planner/milp_solver.py`:
  - 모드 분기: spec 목적에 `power_w(min)`이 있으면 **power-min 모드** —
    `min Σ_t n_t·P(hw_t)·tp_t + Σ_h host_base_w(h)·used[h]`
    s.t. `Σ_t n_t·thr_proxy(t) ≥ demand_toks_s` (+ 기존 메모리/가용량/Max-Flow 제약).
  - `used[h] ∈ {0,1}` 호스트 활성화 변수: 해당 호스트 디바이스 사용 시 1
    (CP-SAT `AddMaxEquality` 또는 indicator 제약).
  - Top-K 생성: 기존 no-good cut 재사용. ε-스윕은 power-min 모드에서는
    `thr ≥ demand·(1+k·δ)` 스윕으로 여유용량 대안 생성(k=0..steps).
  - 전력 계수는 M1의 `effective_power_w` 사용.
- `planner/objective.py`:
  - `Metrics`에 `power_w`(시뮬 에너지/시간 or measured) 추가,
    `_metric_value`·score에 반영. power-min 모드의 최종 랭킹 키 = Stage-2 power_w 오름차순
    (SLO 통과자 한정).
- infeasible 리포트: Stage-1 UNSAT 시 어느 제약 계열(메모리/가용량/수요/링크)을
  완화하면 SAT이 되는지 이진 탐색으로 판정 → `InfeasibleReport` dataclass 반환.

**테스트**
- `planner/tests/test_power_min_solver.py` (unit — 시뮬 불필요, 솔버만):
  - [ ] **수동 최적해 일치**: 장난감 인벤토리(H100×1 700W thr6, A5000×4 230W thr0.8)
        에서 demand=1.6 → 최적해는 A5000×2(460W)이지 H100×1(700W)이 아님을 검증.
  - [ ] **호스트 전력 효과**: host_base_w=500 설정 시 위 최적해가 H100×1로 반전됨
        (A5000×2가 호스트 2대에 흩어진 시나리오) — 호스트 변수 동작 증명.
  - [ ] **수요 제약**: demand를 총 용량보다 크게 → UNSAT + InfeasibleReport에
        bottleneck="demand" 표시.
  - [ ] **여유용량 스윕**: steps=3 → 후보들의 총 thr_proxy가 단조 증가.
  - [ ] **회귀**: 기존 max-throughput/ε-스윕 spec은 동일 해 (골든 비교).
- `planner/tests/test_objective_power.py` (unit):
  - [ ] SLO 통과자 2명 중 power_w 낮은 쪽이 1위; SLO 위반자는 power 무관 탈락.

---

### M3. Workload Synthesizer (§4.3)

**구현**
- `service/presets.py`: chat/summarize/agentic 3종 — (입력 길이 분포, 출력 길이 분포,
  ShareGPT 유사 샘플링 파라미터).
- `service/workload_synth.py`:
  - `synthesize(scale: ScaleSpec, out_path) -> SynthResult`
    — ScaleSpec = {req_per_s, duration_s, preset|explicit 분포, seed}.
    출력: 백엔드 공용 jsonl(도착시각 포함) + `demand_toks_per_s` 환산값
    (E[in]+E[out] 기반) + 요약 통계.
  - 도착 과정: Poisson(기본) / uniform / pulse.
  - upstream `workloads/generators`가 있으면 재사용, 없어도 독립 동작(자체 샘플러).

**테스트**
- `service/tests/test_workload_synth.py` (unit):
  - [ ] 생성 jsonl의 실측 도착률이 목표 req_per_s ±5% (duration 충분히 길게, seed 고정).
  - [ ] `demand_toks_per_s` = Σ(in+out)/duration ±1%.
  - [ ] seed 고정 시 결정적(같은 파일 해시).
  - [ ] 스키마: 두 백엔드 어댑터의 데이터셋 파서(`sim_backends`)가 파싱 성공
        (legacy/upstream 각각의 필드 요구 충족).
- (sim) `service/tests/test_synth_sim_smoke.py`:
  - [ ] 합성 jsonl로 upstream 백엔드 소규모 실행(10 req)이 정상 종료·CSV 파싱 성공.

---

### M4. P0 통합 — CLI로 power-min E2E

**구현**
- `planner/cli.py`에 이미 있는 흐름 재사용. `examples/service/spec_powermin_hetero.yaml`
  작성: demand + power_w(min) + SLO + M3 합성 워크로드 참조.
- `report.py`에 전력 컬럼·백엔드·출처(power source) 표기 추가.

**테스트 (sim, E2E)**
- `planner/tests/test_e2e_powermin.py` (sim 마커):
  - [ ] 예제 spec 실행 → `best_cluster_config.json` 생성, 재투입 실행 성공.
  - [ ] **차별성 검증**: 같은 토폴로지에서 max-throughput spec과 power-min spec의
        최종 해가 다르고, power-min 해의 Stage-2 전력이 더 낮음을 assert.
  - [ ] SLO 위반 후보가 최종 추천에 없음.
- 수동 검증: `report.md`에 후보별 (전력, thr, SLO, 백엔드) 표가 사람이 읽게 나오는지.

**P0 완료 기준**: 위 전부 + `pytest -m "unit or sim"` green.

---

### M5. measured 백엔드 — 이벤트 시뮬 코어 + capacity curve 오라클 (§5.1–5.2)

**구현**
- `oracles/base.py` — Oracle 프로토콜:
  ```python
  class InstanceOracle(Protocol):
      hw: str; model: str; tp: int
      envelope: Envelope                 # 측정 범위 (배치, 길이, 동시성 상한)
      def step_latency(self, batch: BatchState) -> float: ...   # 또는
      def steady_state(self, concurrency: int) -> SteadyPoint   # capacity curve용
      def power_w(self, load: float) -> float
  ```
- `oracles/capacity_curve.py`: YAML 테이블
  `points: [{concurrency, thr_toks_s, ttft_ms, tpot_ms, itl_p99_ms, avg_w}, ...]`
  → 단조 보간. envelope 밖 질의는 `ExtrapolationWarning` + 최근접값.
- `event_sim.py`: 이벤트 기반 시뮬 (표준 라이브러리 heapq, 외부 의존 없음):
  - 입력: cluster allocation(인스턴스 목록+오라클 참조+라우팅 정책), 워크로드 jsonl.
  - 요청 도착 → 라우터(RR/가중치) → 인스턴스 큐 → 오라클로 처리시간 산출 →
    완료 시각·TTFT/TPOT 기록. P/D 분리 시 prefill 완료→KV 전송 지연(링크 bw)→decode 큐.
  - 전력: Σ(인스턴스 power_w(load) 적분) + idle 디바이스 idle_w + 활성 호스트 base_w.
  - 출력: 기존 CSV와 동일 스키마(어댑터 `parse_csv` 재사용 가능하게).
- `sim_backends/measured/__init__.py`: `MeasuredBackend(SimBackend)` —
  `run()`은 서브프로세스 대신 in-process로 event_sim 호출하되 인터페이스 동일.
  `sim_backends.get_backend("measured")` 등록.

**테스트**
- `sim_backends/measured/tests/test_event_sim.py` (unit — 합성 오라클로, 실측 불필요):
  - [ ] **보존 법칙**: 처리 완료 요청 수 = 투입 수; 요청별 TTFT ≤ latency.
  - [ ] **큐잉 정합성(해석해 대조)**: 단일 인스턴스, 고정 서비스율 μ, 도착률 λ<μ
        (M/D/1 근사) → 평균 대기시간이 이론값 ±15% 이내.
  - [ ] **포화 거동**: λ>μ → 대기열 길이 단조 증가, ITL-p99 폭발 (SLO 위반 감지됨).
  - [ ] **라우팅**: 빠른 인스턴스(2×μ)+느린 인스턴스(μ)에 RR vs 유량 가중(2:1) —
        가중 라우팅의 p99가 낮음을 assert (실험 D straggler 재현의 미니어처).
  - [ ] **전력 적분**: 2개 인스턴스 중 1개만 사용 시 총 에너지 =
        active분 + 미사용 idle분 + 호스트 base분 (손계산 대조).
  - [ ] envelope 밖 질의 → 경고 발생.
- `sim_backends/measured/tests/test_backend_contract.py` (unit):
  - [ ] `get_backend("measured")` 반환 객체가 SimBackend 계약 충족,
        `parse_csv` 출력 스키마가 upstream 어댑터와 동일 키.
  - [ ] planner `sim_evaluator`가 backend="measured"로 무수정 동작(mock 오라클 fixture).

---

### M6. 오라클 캠페인 + NPU 버킷 오라클 (§5.2)

**구현**
- `oracles/npu_bucket.py`: LENS 합성 공식 구현 —
  `T = TTFT_bucket(l_in) + Σ_j l_out_j · TBT_bucket(c_t)` (버킷 경계에서 세그먼트 분할),
  배치 확장(prefill 요청별 합산, decode는 배치 내 max KV 위치의 버킷).
  YAML: `buckets: [{size, ttft_ms, tbt_ms, avg_w}]`.
- `campaign/capacity_campaign.py`: (hw 마커) vLLM 서버 대상 동시성 스윕 → curve YAML 생성.
  upstream `bench/` 재사용. `--dry-run`은 명령만 출력.
- `campaign/bucket_campaign.py`: (hw 마커) NPU 대상 버킷당 2회 측정 프로토콜 러너 (스텁 허용:
  RNGD SDK 명령 템플릿 + 결과 파서만, 실측은 하드웨어 확보 시).

**테스트**
- `sim_backends/measured/tests/test_npu_bucket.py` (unit):
  - [ ] **LENS 논문 수식 재현**: 버킷 {4096: TTFT 100ms/TBT 10ms, 8192: 120/14},
        l_in=3000, l_out=2000 → decode가 4096 경계를 넘는 지점(t=1095)에서 세그먼트
        분할되어 T = 100 + 1096·10 + 904·14 (손계산) 일치.
  - [ ] 배치 decode: 배치 내 max KV가 지배함을 케이스로 검증.
  - [ ] 버킷 초과 입력(l_in > max bucket) → 명시적 InfeasibleError.
- `test_campaign.py` (unit): 캠페인 러너가 mock 측정치(주입된 가짜 bench 결과)로
  올바른 YAML을 생성; `--dry-run` 출력에 필수 인자 포함.
- (hw, CI 제외) A5000 또는 A40 1종 실측 캠페인 → curve YAML 커밋.

---

### M7. 3자 대조 리포트 (§7 P0.5 완료 기준)

**구현**
- `scripts/compare_backends.py`: 동일 (cluster config, 워크로드)에 대해
  measured / upstream sim / (있으면) 실측 vLLM 기록(fixture)을 실행·수집 →
  metric별 diff% 표 (md 출력). `validation/`의 기존 실측 결과를 fixture로 재사용.

**테스트**
- (unit) fixture 3종 주입 → diff 계산·표 생성 정확성.
- (sim) A5000 curve(M6 실측 또는 합성 fixture)로 measured vs upstream 실행이 완주.
- **P0.5 완료 기준(DoD)**:
  - [ ] 대조 리포트에서 measured 백엔드의 TTFT/TPOT 오차(실측 vLLM 대비)가
        upstream sim 오차보다 작거나 같음 (A5000 TP1 기준).
  - [ ] measured 평가 시간이 후보당 10초 미만.

---

### M8. Inventory Manager (§4.4)

**구현**
- `inventory/registry.py`: `cluster_registry.yaml` (노드/디바이스/링크/전력·오라클 참조,
  `docs/dse/03_catalog.yaml`의 availability 흡수) 로더 + 검증.
- `inventory/ledger.py`: SQLite. 테이블 `reservations(id, tenant, request_id,
  device_ids json, state, snapshot_ver, created_at)`. API:
  `snapshot() -> (ver, free_devices)`, `reserve(request_id, devices, snapshot_ver)`
  — ver 불일치 시 `ConflictError`, `release(request_id)`.
- `inventory/views.py`: free_devices → planner spec `topology` dict 직렬화
  (링크·노드 구조 보존, count만 차감).

**테스트** — `service/tests/test_inventory.py` (unit, tmp sqlite):
- [ ] reserve→release 사이클 후 free 수 복원.
- [ ] **optimistic locking**: 스냅샷 A로 두 예약 시도 → 첫 번째 성공, 두 번째 ConflictError.
- [ ] 동시성: ThreadPool 10개가 같은 디바이스 예약 경쟁 → 정확히 1개 성공 (sqlite 락).
- [ ] 부분 예약 후 views 직렬화가 planner `spec_schema` 검증 통과 + 남은 count 정확.
- [ ] 이중 release 멱등.

---

### M9. Service API + Fidelity Router (§4.5, §5.4)

**구현**
- `service/fidelity_router.py`: 후보(allocation)의 하드웨어 집합 → 백엔드 결정:
  ```
  NPU 포함 & 버킷 오라클 있음        → measured (강제)
  전 디바이스 오라클 있음            → measured
  오라클 없음 & GPU-only            → upstream
  오라클 없음 & RNGD 포함           → legacy (신뢰도 low 표기)
  spec.force_backend 있으면 override
  ```
  반환: (backend_name, confidence: high|medium|low, reason).
- `service/api/routes.py` (webapp에 `app.include_router`):
  `GET /api/cluster`, `GET /api/models`, `POST /api/serve-requests`,
  `GET /api/serve-requests/{id}` (SSE 진행), `POST .../confirm`, `POST .../release`.
  잡 실행은 DSE 잡 러너 패턴 재사용(백그라운드 태스크 + 산출물 디렉토리).
- confirm 흐름: 결과의 snapshot_ver로 `ledger.reserve` → ConflictError 시
  자동 재플래닝 1회 후 사용자에게 갱신안 제시.

**테스트** — `service/tests/test_api.py` (unit, FastAPI TestClient + planner mock):
- [ ] 정상 흐름: submit→(mock planner 즉시 완료)→result→confirm→cluster에 예약 반영
      →release→복원.
- [ ] 검증 실패: 카탈로그에 없는 모델 → 422 + 원인 메시지.
- [ ] infeasible: mock이 InfeasibleReport 반환 → 응답에 bottleneck·완화 시나리오 포함.
- [ ] **동시 confirm 충돌**: 두 클라이언트가 겹치는 자원 confirm → 한쪽 재플래닝 경로 진입.
- `service/tests/test_fidelity_router.py` (unit):
  - [ ] 표의 5개 규칙 각각 케이스; NPU+오라클 없음+legacy 프로파일 없음 → 명시적 에러.
- (sim) `test_api_e2e.py`: mock 없이 소형 spec으로 submit→result 완주 (수 분 허용).

**P1 완료 기준(DoD)**:
- [ ] 테넌트 A 예약 후 테넌트 B의 추천이 잔여 자원만 사용함을 E2E로 확인.
- [ ] NPU 포함 요청이 자동으로 measured 라우팅됨 (응답에 backend 표기).

---

### M10. 웹 UI + 문서화

**구현**
- webapp 템플릿에 서비스 탭: 요구사항 폼(모델 드롭다운=카탈로그, SLO, 규모),
  추천 카드(전력·성능·신뢰도·대안), 클러스터 현황판(가용/예약).
- `service/README.md`: 아키텍처 다이어그램, API 예제(curl), 오라클 캠페인 가이드,
  Fidelity Router 규칙표. 루트 README에 서비스 섹션 추가.

**테스트**
- [ ] Playwright 또는 수동 체크리스트: 폼 제출→진행 표시(SSE)→추천 카드→confirm 버튼
      →현황판 갱신.
- [ ] README의 모든 명령 복붙 실행 성공 (문서 스모크).

---

## 4. 테스트 전략 요약

| 계층 | 도구 | 시점 |
|---|---|---|
| 솔버·오라클·이벤트시뮬 수학 | pytest unit, 손계산·해석해(M/D/1) 대조 | 매 커밋 |
| 백엔드 계약 | contract test (세 백엔드 동일 스키마) | 매 커밋 |
| planner 회귀 | 골든 spec→해 비교 (기존 동작 불변) | 매 커밋 |
| E2E (sim) | 소형 spec 완주, 차별성 assert (M4/M7/M9) | 마일스톤 완료 시 |
| 실측 (hw) | 캠페인 + 3자 대조 (M6/M7) | 하드웨어 가용 시, CI 제외 |
| 동시성 | ThreadPool 경쟁, API 이중 confirm | M8/M9 |

**전역 회귀 가드**: 모든 마일스톤에서 `pytest planner/tests/ tests/ -m "unit"` green 유지.
기존 예제 spec(`example_hetero_8gpu.yaml`)의 산출 해는 골든 파일로 고정.

## 5. 리스크 메모 (구현 시 주의)

- legacy 백엔드 CLI는 **backend 루트 기준 상대 경로만** 허용 — 신규 경로 전달 시 반드시
  어댑터의 경로 변환 경유 (CLAUDE.md 명시 quirk).
- `VALID_OBJECTIVE_METRICS` 확장 시 webapp planner_server의 spec 빌더도 함께 갱신할 것
  (숨은 소비자 grep: `toks_per_wh`).
- CP-SAT 정수화: 전력(W)·thr 프록시는 기존 `_FLOW_SCALE` 방식으로 스케일링. 부동소수점
  계수를 직접 넣지 말 것.
- OR-Tools `IntVar.Proto()` introspection 금지 (기존 세그폴트 이력 — REPORT §8.5).
- event_sim의 시간 단위는 ns로 통일 (기존 CSV semantics와 일치).
- 오라클 YAML에는 반드시 `meta.stack`(vLLM/SDK 버전)과 `envelope`를 기록 — Fidelity
  Router와 신선도 검사가 이 필드에 의존.
