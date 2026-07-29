# service/ — 전력 최적 LLM 서빙 자원 추천 서비스

사용자가 「모델 + SLO + 서비스 규모」를 제출하면, 가용 자원 중 SLO·규모를
만족하는 **최저 전력 자원 조합**을 도출하고 예약까지 관리하는 다중 테넌트
서비스 계층. `PLAN_power_service.md`(M0–M10)와 설계서
`LLM_서빙_전력최적화_서비스_설계서_v2.docx`의 구현이다.

## 아키텍처

```
POST /api/serve-requests ──▶ Job (async thread)
  │ 1. Inventory snapshot(ver) ── ledger.sqlite (예약 원장, optimistic locking)
  │ 2. Workload Synthesizer ──── 규모(req/s, preset) → 합성 jsonl + demand toks/s
  │ 3. Fidelity Router ───────── 후보 하드웨어 → 평가 백엔드 선택
  │      NPU+오라클 → measured 강제 | 전체 오라클 커버 → measured
  │      GPU-only 미커버 → upstream | NPU 미커버 → legacy(low) | 없음 → 에러
  │ 4. Power-Min Planner ─────── Stage-1 CP-SAT: min Σ전력 s.t. thr ≥ demand
  │                              Stage-2: 선택된 백엔드로 SLO 검증·전력 산정
  └─▶ 추천(전력·성능·신뢰도·대안 Top-K) 또는 InfeasibleReport(병목·완화안)

confirm ──▶ ledger.reserve(snapshot_ver)  # 충돌 시 자동 재플래닝 1회
release ──▶ ledger.release               # 멱등
```

핵심 모듈: `workload_synth.py`(M3) · `presets.py` · `inventory/`(M8:
registry/ledger/views) · `fidelity_router.py`(M9) · `api/`(M9).
measured 백엔드 자체는 `sim_backends/measured/`(M5–M7).

## 시작하기

```bash
pip install -r requirements-planner.txt -r requirements-service.txt
cp service/cluster_registry.example.yaml cluster_registry.yaml   # 클러스터 정의
./scripts/serve_webapp.sh          # http://localhost:8000/service 탭 활성화
```

### API 예제 (curl)

```bash
# 클러스터 현황 (가용/예약 + 스냅샷 버전)
curl -s localhost:8000/api/cluster | jq .

# 서비스 가능 모델 목록 (measured 오라클 + legacy 프로파일 교차)
curl -s localhost:8000/api/models | jq .

# 요구사항 제출 → 잡 id
curl -s -X POST localhost:8000/api/serve-requests \
  -H 'Content-Type: application/json' -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "tenant": "teamA",
    "slo": {"tpot_ms": 200},
    "scale": {"req_per_s": 2, "preset": "chat", "duration_s": 30}
  }' | jq .

# 결과 조회 / SSE 진행 스트림
curl -s localhost:8000/api/serve-requests/<id> | jq .result
curl -N localhost:8000/api/serve-requests/<id>/events

# 확정(예약) / 반환
curl -s -X POST localhost:8000/api/serve-requests/<id>/confirm | jq .
curl -s -X POST localhost:8000/api/serve-requests/<id>/release | jq .
```

## 오라클 캠페인 (measured 백엔드용 실측)

인스턴스 단위로 한 번 측정하면 클러스터 조합은 이벤트 시뮬이 합성한다
(측정 비용은 하드웨어×모델×TP에 선형, 후보 수와 무관).

```bash
# GPU capacity curve: vLLM 서버 동시성 스윕 (dry-run으로 명령 확인)
python -c "from sim_backends.measured.campaign.capacity_campaign import *; \
  print(run_campaign(CampaignConfig(hw='A40', model='meta-llama/Llama-3.1-8B', tp=1), dry_run=True))"

# NPU 버킷 (LENS 프로토콜: 버킷당 2회 측정)
python -c "from sim_backends.measured.campaign.bucket_campaign import *; \
  print(run_campaign(BucketCampaignConfig(hw='RNGD', model='meta-llama/Llama-3.1-8B', tp=1), dry_run=True))"
```

산출 YAML은 `profiles/measured/<HW>/<model>/tp<N>.yaml`. `meta.stack`이
스택 버전 기록(신선도 검사 기준)이며, 측정 범위는 `envelope`로 기록되어
범위 밖 질의는 `ExtrapolationWarning`을 낸다.

## 정확도 (A40 TP1, ShareGPT 300req — docs/COMPARE_BACKENDS_A40_TP1.md)

| | measured | upstream sim |
|---|---|---|
| 실측 vLLM 대비 평균 \|diff%\| | **16.0%** | 80.2% |
| 후보당 평가 시간 | 수 초 | 수 분 |

## 테스트

```bash
pytest -m unit service/tests sim_backends/measured/tests   # 매 커밋
pytest -m "unit or sim" service/tests                      # E2E 포함
```
