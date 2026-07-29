# LLMServingOptimizer

LLM 서빙 클러스터 설계 최적화 도구 모음 — **webapp DSE**(Design Space Exploration UI),
**MILP/Max-Flow planner**, 검증 스위트를 최상위에 두고, 시뮬레이터 두 종을
`backends/` 하위 git submodule로 구동하는 워크스페이스.

```
LLMServingOptimizer/
├── backends/
│   ├── legacy/        # ai-computing/LLMServingSim fork (branch backend-legacy)
│   └── upstream/      # casys-kaist/LLMServingSim v1.1+ (pinned)
├── sim_backends/      # 백엔드 어댑터 (CLI/config/CSV semantics 차이 흡수)
├── webapp/            # FastAPI 웹 UI (단일 실행/sweep + dse/ 자동 탐색)
├── planner/           # MILP/Max-Flow 서빙 플래너 (spec.backend로 선택)
├── tests/             # DSE 서브시스템 pytest
├── cluster_config/    # legacy 포맷 구성 (+ upstream/ 하위: 신 포맷)
├── dataset/           # 워크로드 jsonl
├── examples/dse/      # DSE 탐색 spec 예시 (smoke/70B demo)
├── profiles/upstream/ # 자체 프로파일한 신포맷 프로파일 (예: A5000)
├── validation/        # vLLM 실측 대조 스크립트/리포트
├── docs/              # 검증 리포트, 설계 문서 (PLAN_*.md)
├── output/            # 시뮬레이션 결과 (dse_jobs/ 포함)
└── scripts/setup.sh   # 전체 환경 셋업 (submodule→빌드→venv→심링크)
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
./scripts/setup.sh
pip install -r requirements-planner.txt   # planner 사용 시 (ortools, networkx, pydantic, pandas)
```

## 백엔드 개요

| | legacy | upstream |
|---|---|---|
| 소스 | ai-computing fork (우리 코어 패치 포함) | casys-kaist v1.1.0+ |
| 실행 | `python main.py` | `.venv/bin/python -m serving` |
| 프로파일 | `llm_profile/perf_models/` (구 포맷, A100/A40/A5000/A6000/RNGD…) | `profiler/perf/` (vLLM layerwise, RTXPRO6000 + 자체 A5000) |
| 정확도(A5000 TP1 실측 대비) | TTFT 65% / TPOT 16% 오차 | **TTFT 19% / TPOT 8% 오차** |
| 특기 | collective-overhead 캘리브레이션, DSE fabric | skew-attention, chunked prefill 기본, DP+EP |

두 백엔드 모두 TP>1 collective 비용은 과소평가 경향(특히 PCIe 클러스터) —
`docs/A5000_VALIDATION_REPORT.md` 참고.

## 사용법

**직접 실행 (어댑터):**
```python
from sim_backends import get_backend, ScenarioSpec
b = get_backend("upstream")          # 또는 "legacy"
proc = b.run("cluster_config/upstream/a5000_tp1_validation.json",
             "output/run.csv",
             ScenarioSpec(dataset="dataset/sharegpt_req100_rate10_llama.jsonl",
                          num_reqs=100, dtype="bf16"))
rows = b.parse_csv("output/run.csv")     # output 칼럼 semantics 정규화됨
print(b.parse_stdout(proc.stdout))
```

**webapp:** `./scripts/serve_webapp.sh` → http://localhost:8000 — Workload 카드의
"Simulator backend" 드롭다운으로 선택.

**DSE (자동 설계공간 탐색):** 웹 UI는 http://localhost:8000/dse/explore, CLI는:
```bash
python -m webapp.dse.cli explore --spec examples/dse/spec_llama8b_smoke.yaml --job-name smoke
./scripts/demo_dse.sh    # 위 smoke spec 원라인 데모 + Top-N 출력
```
결과는 `output/dse_jobs/<timestamp>-<name>/` (top_n/pareto/all_candidates.json).
상세: [webapp/dse/README.md](webapp/dse/README.md)

**planner (이종 클러스터 자원 할당):**
```bash
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --validate-only  # spec 검증만
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --dry-run        # Stage-1만 (시뮬 없음)
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --out-dir planner_out/ --jobs 8
```
spec YAML에 `backend: upstream` 한 줄 추가로 백엔드 선택 (기본 legacy).
상세: [planner/README.md](planner/README.md)

**DP 파티션 시뮬레이션:** 루트의 `run_dp_partition.py`(DP=N을 N개 단일 인스턴스
시뮬의 max로 근사) / `run_dp_comparison.py`(실측 vLLM DP 스케일링과 대조).

**테스트:**
```bash
pytest -m unit                 # 전체 unit 스위트 (시뮬 빌드 불필요)
pytest -m "unit or sim"        # 마일스톤 DoD: 시뮬 E2E 포함
pytest tests/                  # DSE 서브시스템만
pytest planner/tests/          # planner만 (MILP solver, renderer, mock evaluator)
```

**전력 최적 서빙 추천 서비스 (`service/` + `sim_backends/measured/`):**
「모델 + SLO + 규모」 제출 → SLO·수요를 만족하는 **최저 전력** 자원 조합 추천
→ 확정 시 인벤토리 예약. planner의 power-min 모드(CP-SAT: min Σ전력 s.t.
처리량 ≥ demand)와 제3 평가 백엔드 `measured`(실측 오라클 + 이벤트 시뮬,
후보당 수 초·A40 TP1 평균 오차 16% vs upstream 80%)를 사용한다.

```bash
cp service/cluster_registry.example.yaml cluster_registry.yaml
./scripts/serve_webapp.sh                    # → http://localhost:8000/service
# CLI만으로 power-min 플래닝:
python -m planner.cli --spec examples/service/spec_powermin_hetero.yaml --out-dir planner_out/powermin
# 3자 대조(measured vs sim vs 실측): scripts/compare_backends.py
```

자세한 API/캠페인 가이드는 `service/README.md`, 계획서는 `PLAN_power_service.md` 참고.

## 주의사항 (검증된 퀴크)

- upstream chakra는 `protobuf==6.*`을 핀하지만 gencode가 7.35 → venv에서 protobuf 업그레이드 필요 (setup.sh가 처리)
- upstream venv에 **msgspec** 필수 (없으면 radix_tree AttributeError)
- 양쪽 모두 ASTRA-Sim 빌드 후 `AnalyticalAstra` 심링크 필요 (setup.sh가 처리)
- legacy CLI는 백엔드 루트 기준 **상대 경로만** 허용 (config_builder가 `../` 접합) — 어댑터가 자동 변환
- GPU 프로파일링/bench는 별도 vLLM 0.19.0 venv 필요 (setup.sh 말미 안내 참조)
