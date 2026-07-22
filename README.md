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
├── webapp/            # FastAPI DSE UI  (백엔드 선택 드롭다운)
├── planner/           # MILP/Max-Flow 서빙 플래너 (spec.backend로 선택)
├── tests/             # DSE 서브시스템 pytest
├── cluster_config/    # legacy 포맷 구성 (+ upstream/ 하위: 신 포맷)
├── dataset/           # 워크로드 jsonl
├── profiles/upstream/ # 자체 프로파일한 신포맷 프로파일 (예: A5000)
├── validation/        # vLLM 실측 대조 스크립트/리포트
└── scripts/setup.sh   # 전체 환경 셋업 (submodule→빌드→venv→심링크)
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
./scripts/setup.sh
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

**planner:** spec YAML에 `backend: upstream` 한 줄 추가 (기본 legacy).

**테스트:** `pytest tests/`

## 주의사항 (검증된 퀴크)

- upstream chakra는 `protobuf==6.*`을 핀하지만 gencode가 7.35 → venv에서 protobuf 업그레이드 필요 (setup.sh가 처리)
- upstream venv에 **msgspec** 필수 (없으면 radix_tree AttributeError)
- 양쪽 모두 ASTRA-Sim 빌드 후 `AnalyticalAstra` 심링크 필요 (setup.sh가 처리)
- legacy CLI는 백엔드 루트 기준 **상대 경로만** 허용 (config_builder가 `../` 접합) — 어댑터가 자동 변환
- GPU 프로파일링/bench는 별도 vLLM 0.19.0 venv 필요 (setup.sh 말미 안내 참조)
