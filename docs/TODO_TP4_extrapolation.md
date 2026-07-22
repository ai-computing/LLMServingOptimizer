# TODO: LLMServingSim TP=4 외삽 지원 구현

A5000 2장(TP=1, 2)만으로 TP=4 성능을 시뮬레이션하기 위한 작업 목록.
핵심 전략: **통신은 ASTRA-sim 해석 백엔드에 위임(이미 TP=4 가능), 연산(compute) latency만 합성**하여 `perf_models/.../tp4/` 번들을 생성·주입한다.

---

## Phase 0. 환경 준비 & 현황 파악

- [ ] LLMServingSim(casys-kaist) 저장소 클론 및 빌드, ASTRA-sim 서브모듈(v0.2.0) 초기화
- [ ] 대상 모델 결정 (예: Llama-3.x-8B 등) — **heads % 4 == 0** 제약 충족 여부 확인
- [ ] `perf_models/{hardware}/{model}/tp{tp_size}/` 디렉토리 구조 실측 확인
- [ ] `layers.csv` / `attention.csv` 의 **정확한 컬럼 헤더 및 단위** 기록 (보고서에서 verbatim 미확인 항목)
- [ ] `generate_trace()` / `synthsize_trace()` 에서 latency 룩업 + ALL-REDUCE 삽입 코드 경로 위치 파악
- [ ] scikit-learn attention 예측기(`--enable-attn-prediction`, `build_predictor.sh`) 학습/추론 코드 위치 파악

## Phase 1. Ground Truth 확보 (A5000 2장)

- [ ] TP=1 프로파일링 실행 → `tp1/` 번들 생성
- [ ] TP=2 프로파일링 실행 → `tp2/` 번들 생성
- [ ] TP=1, TP=2 시뮬레이션이 정상 동작하는지 baseline 검증
- [ ] (가능 시) 실제 vLLM TP=2 측정값 확보 → 후속 외삽 역검증용 ground truth로 보관

---

## Phase 2. 연산 latency 합성 모듈 구현

> 목표: 측정 없이 `tp4/layers.csv`, `tp4/attention.csv` 를 생성하는 합성기

### 2A. 형상 기반 회귀 외삽 (1순위)

- [ ] TP=1, 2 프로파일에서 `(연산 형상 features) → latency` 데이터셋 추출
  - features 예: op type, M/N/K (GEMM), heads_per_device(=H/tp), hidden, batch, seq_len, dtype
- [ ] 회귀 모델 학습 (Vidur 방식 참고: **random forest** 권장, 비선형 커널 특성 포착)
- [ ] TP=4의 샤딩된 형상(heads = H/4) 생성 로직 구현
- [ ] 학습된 모델로 TP=4 각 op latency 예측 → `tp4/layers.csv` 합성
- [ ] attention latency도 동일 방식으로 `tp4/attention.csv` 합성

### 2B. 루프라인 1차 근사 (2순위 / fallback)

- [ ] compute-bound op(QKV proj, FFN): `latency ≈ FLOPs / (peak_compute × eff)` 로 근사
- [ ] memory-bound op(MHA, LayerNorm): `latency ≈ bytes / (mem_BW × eff)` 로 근사
- [ ] TP=4에서 디바이스당 작업량 1/4 반영하여 TP=1 단일 GPU 프로파일 스케일
- [ ] A5000 스펙 상수화: FP32 27.8 TFLOPS / Tensor 222.2 TFLOPS / 24GB / **768 GB/s** / PCIe Gen4 / 230W
- [ ] 효율 인자(eff) TP=1,2 실측으로 캘리브레이션 (예: compute ~60%, mem ~70%)

### 2C. (선택) TP-cross 예측기 확장

- [ ] 기존 scikit-learn 예측기 입력 feature에 `tp_size`(또는 heads_per_device) 추가
- [ ] TP=4는 학습범위(1,2) 밖 **외삽**임을 명시 — 신뢰구간/경고 출력 추가

---

## Phase 3. 통신(ALL-REDUCE) 모델 — ASTRA-sim 위임

> ASTRA-sim 해석 백엔드는 이미 TP=4 통신 모델링 가능 (NCCL all-reduce 대비 평균 오차 ~5%)

- [ ] Graph Converter가 TP=4에서 레이어당 ALL-REDUCE 2회(attention 뒤 / MLP 뒤) 삽입하는지 확인
- [ ] ASTRA-sim 네트워크 토폴로지/대역폭 설정을 **실제 인터커넥트에 맞게** 구성
  - [ ] NVLink 노출 여부 확인 (미노출 시 **PCIe Gen4 강제** 반영)
  - [ ] ring all-reduce 비용식 `(p−1)(α + v/(w·p))` 파라미터(α 지연, w 대역폭) 세팅
- [ ] decode 구간 ALL-REDUCE는 **작은 메시지(수십 KB~1MB) → latency-bound** 임을 반영

---

## Phase 4. 통합 & 주입

- [ ] 합성된 `perf_models/A5000/{model}/tp4/` 번들을 시뮬레이터가 정상 로드하는지 확인
- [ ] `--npu_group 4` (또는 해당 TP 파라미터)로 TP=4 시뮬레이션 end-to-end 실행
- [ ] 합성기 CLI/스크립트화 (예: `synthesize_tp.py --src tp1,tp2 --target tp4 --method rf|roofline`)

---

## Phase 5. 검증 (Validation) — **결정 임계값 포함**

> 핵심 원칙: TP=4를 직접 검증할 수 없으므로 **TP=2로 역검증(back-validation)**

- [ ] 합성기를 **TP=2에 적용**(TP=1만으로 TP=2 예측) → 실측 `tp2/`와 비교
- [ ] 오차 평가:
  - [ ] **< 15%** → TP=4 외삽 신뢰, 진행
  - [ ] **≥ 15%** → 도구 기반 추정(Vidur / GenZ)으로 전환 (Phase 6)
- [ ] **decode TPOT는 별도 검증** (PCIe 환경 통신 오버헤드 편차 큼)
- [ ] prefill / decode 구간 분리하여 오차 리포트

---

## Phase 6. (조건부) 외부 도구 교차검증 / 대체

> Phase 5에서 오차 ≥15% 이거나 신뢰도 보강이 필요할 때

- [ ] **Vidur**: 단일 A5000 프로파일 → random forest로 TP=4 예측 (TP 확장에 본질적으로 적합, 오차 <9% 보고)
- [ ] **GenZ**: A5000 스펙만 입력해 TP=4 해석적 투영 (1차 추정, prefill/decode 오차 2.73%/1.85% 보고)
- [ ] **LLMCompass**: 디바이스 description으로 4-way TP 평가 (오차 10.4%/4.1% 보고)
- [ ] LLMServingSim 합성 결과 vs 외부 도구 결과 비교표 작성

---

## Phase 7. 마무리 & 향후

- [ ] 실제 A5000 4장 확보 시 측정값으로 `tp4/` 교체 후 재검증하는 절차 문서화
- [ ] 합성 방법/효율 인자/검증 결과를 README에 기록
- [ ] 외삽 한계·주의사항 명시 (TP=4는 학습범위 밖 외삽, NVLink/PCIe 가정, 효율 인자 환경 의존)

---

### 참고: 핵심 가정/주의
- 합성은 **외삽**(TP=1,2 → TP=4)이며 보간이 아님 → 항상 역검증 동반
- 통신은 측정 불필요(ASTRA-sim 해석), **연산 latency만 합성**하면 됨이 핵심 단서
- 클라우드/멀티카드에서 NVLink 미노출 → PCIe 강제 가능성 높음, 통신 가정 반드시 확인
