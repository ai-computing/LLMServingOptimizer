# `webapp/dse/` — Design Space Exploration

LLMServingSim의 sweep 기능 위에 얹어진 **자동 탐색 도구**. 사용자의 자원 풀(예: H100 1~4, A6000 0~8) + SLO/Throughput/Power 제약을 받아, 가능한 모든 cluster 조합을 자동 생성 → 병렬 시뮬레이션 → Pareto frontier + 가중치 기반 Top-N 출력.

## 빠른 시작 — CLI

```bash
python -m webapp.dse.cli explore \
    --spec examples/dse/spec_llama8b_smoke.yaml \
    --job-name smoke
```

결과는 `output/dse_jobs/<timestamp>-<name>/` 아래:
- `all_candidates.json` — 모든 후보의 SimulationResult
- `top_n.json` — 가중치 기반 상위 N개
- `pareto.json` — Pareto-optimal 후보
- `configs/<label>.json` — 각 후보의 cluster JSON
- `runs/<label>.{log,csv}` — main.py 출력

## 빠른 시작 — Web UI

1. `bash script/serve_webapp.sh`로 웹 서버 띄움
2. 브라우저에서 `http://localhost:8000/dse/explore` 접속
3. Resource pool / Model / Workload / Constraints / Weights 입력
4. **Estimate count**로 후보 수 확인
5. **Start Exploration** → 진행 페이지로 이동
6. 완료 후 결과 페이지에서 Top-N, Pareto, Radar 차트 확인 + 가중치 슬라이더로 재랭킹

## 디렉토리 구조

```
webapp/dse/
├── core/
│   ├── schemas.py         # Pydantic 모델 (JobSpec, CandidateConfig, ...)
│   ├── generator.py       # ResourcePool → list[CandidateConfig]
│   ├── config_builder.py  # CandidateConfig → cluster JSON + power
│   ├── runner.py          # webapp.runner.run_sweep 래퍼
│   └── ranker.py          # SLO 필터 + Pareto + Top-N
├── server/
│   ├── routes.py          # /api/dse/* 라우트
│   └── __init__.py
├── cli.py                 # python -m webapp.dse.cli
└── README.md              # 이 파일
```

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET    | `/api/dse/catalog` | hw + 모델 카탈로그 |
| POST   | `/api/dse/dry-run` | 후보 수 예상 (no job 생성) |
| POST   | `/api/dse/jobs` | 신규 탐색 작업 시작 |
| GET    | `/api/dse/jobs` | 작업 목록 (history) |
| GET    | `/api/dse/jobs/{id}` | 상태 |
| GET    | `/api/dse/jobs/{id}/results` | 결과 + Top-N + Pareto |
| POST   | `/api/dse/jobs/{id}/rerank` | 가중치 변경하여 재랭킹 (재시뮬 없음) |
| DELETE | `/api/dse/jobs/{id}` | 취소/삭제 |
| GET    | `/api/dse/jobs/{id}/events` | SSE 진행 stream |
| GET    | `/api/dse/jobs/{id}/download.zip` | 결과 zip |

## Spec 파일 예시

`examples/dse/spec_llama8b_smoke.yaml` 참조. 최소 형식:

```yaml
resource_pool:
  items:
    - {hw: A6000, min: 1, max: 4}
    - {hw: RNGD,  min: 0, max: 2}
model:
  name: meta-llama/Llama-3.1-8B
  fp: 16
workload:
  dataset: dataset/sharegpt_req100_rate10_llama.jsonl
  num_req: 100
  timeout_s: 120
constraints:
  ttft_p99_ms: 500
  throughput_min_tok_s: 1500
weights:
  ttft: 0.3
  tpot: 0.2
  throughput: 0.3
  power: 0.2
search:
  max_combinations: 64
top_n: 5
```

## 알려진 제약 / 알아둘 점

- ASTRA-Sim heterogeneous P/D는 일부 패턴에서 deadlock — 이미 `webapp/enumerate.py`가 사전 차단하지만 일부는 timeout으로 fail
- 메모리 사전 필터는 model_weight ≤ aggregate_npu_mem만 확인 (KV cache + activations는 무시)
- 결과 캐시: 동일 spec hash가 있으면 즉시 기존 결과 반환 (시뮬레이션 재실행 안 함). hash는 `weights` 및 `top_n`을 제외해서 계산하므로, 같은 시뮬레이션 결과를 다른 가중치로 보고 싶을 때는 rerank API 사용
- 재랭킹: 결과 파일이 있어야 동작 (`all_candidates.json`이 있어야 함)

## 모듈 단위 테스트

```bash
python -m pytest tests/dse/ -v
```

38개 테스트 (generator + ranker + schemas) 통과 검증됨.

## PLAN 문서

전체 개발 plan은:
- `PLAN_webapp_dse.md` — 원본 계획 (What/Why)
- `PLAN_webapp_dse_detail.md` — Phase별 세부 작업 (How)
- `docs/dse/00..03` — 사전 조사 자료
