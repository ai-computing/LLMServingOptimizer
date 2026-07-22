# A5000 Llama-3.1-8B Level 3 Validation Plan

**목표**: NVIDIA RTX A5000 2장 환경에서 Llama-3.1-8B 모델에 대해  
시뮬레이터 예측 결과와 실제 vLLM 서빙 결과를 비교해 정확도를 정량화한다.

**대상 설정**: TP=1 (GPU 1장), TP=2 (GPU 2장 PCIe)  
**측정 지표**: TTFT p50/p99, TPOT p99, Throughput (tok/s), GPU 전력 (W)

---

## 환경 요약

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX A5000 × 2 |
| GPU 메모리 | 24 GB GDDR6 (각) |
| 메모리 대역폭 | 768 GB/s (각) |
| TDP | 230 W (각) |
| GPU 간 연결 | PCIe 4.0 x16 (NODE 토폴로지, NVLink 비활성) |
| 인터커넥트 대역폭 (TP=2) | ~32 GB/s 단방향 |
| 모델 | meta-llama/Llama-3.1-8B |
| vLLM 버전 | 0.22.0 |

---

## Phase 1: A5000 프로파일링

A5000은 기존 프로파일이 없으므로 새로 측정한다.  
`llm_profile/` 안에서 작업하며 결과는 `llm_profile/perf_models/A5000/meta-llama/Llama-3.1-8B/tp{n}/`에 저장된다.

### Step 1-1: Layer 프로파일링 (TP=1, TP=2)

```bash
cd llm_profile

# TP=1
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.layers.main \
  --hardware A5000 \
  --model "meta-llama/Llama-3.1-8B" \
  --num-layers 1 \
  --tp-size "1" \
  --warmup 10 \
  --repeat 30 \
  --max-len 10 \
  --device cuda

# TP=2
CUDA_VISIBLE_DEVICES=0,1 python3 -m profiler.layers.main \
  --hardware A5000 \
  --model "meta-llama/Llama-3.1-8B" \
  --num-layers 1 \
  --tp-size "2" \
  --warmup 10 \
  --repeat 30 \
  --max-len 10 \
  --device cuda
```

출력: `perf_models/A5000/meta-llama/Llama-3.1-8B/tp1/layers.csv`,  
       `perf_models/A5000/meta-llama/Llama-3.1-8B/tp2/layers.csv`

### Step 1-2: Attention 프로파일링 (TP=1, TP=2)

```bash
# TP=1
CUDA_VISIBLE_DEVICES=0 python3 -m profiler.attention.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware A5000 \
  --max-len 2048 \
  --tp-size "1" \
  --warmup 10 \
  --repeat 50 \
  --device cuda

# TP=2
CUDA_VISIBLE_DEVICES=0,1 python3 -m profiler.attention.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware A5000 \
  --max-len 2048 \
  --tp-size "2" \
  --warmup 10 \
  --repeat 50 \
  --device cuda
```

출력: `perf_models/A5000/meta-llama/Llama-3.1-8B/tp{n}/attention.csv`

### Step 1-3: 어텐션 예측기 빌드 (TP=1, TP=2)

```bash
python3 -m profiler.predictor.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware A5000 \
  --tp-size "1, 2" \
  --kv-granularity 64 \
  --chunk-granularity 32 \
  --max-len 2048 \
  --max-batch 256
```

출력: `perf_models/A5000/meta-llama/Llama-3.1-8B/tp{n}/predictions/*.pkl`

### Step 1-4: GPU 전력 프로파일링

시뮬레이터 power model의 `idle_power`, `active_power` 파라미터 측정.

```bash
# 터미널 A: 전력 로깅 (1초 간격)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000 > validation/a5000_power_log.txt

# 터미널 B: 부하 측정 (Layer 프로파일링 중 power 기록)
# → 프로파일링 완료 후 로그에서 idle/active 값 추출
```

측정 목표:
- `idle_power`: GPU 유휴 시 평균 전력 (W)
- `active_power`: 추론 부하 시 95th percentile 전력 (W)
- `standby_power`: vLLM 서버 대기 중 전력 (W)

---

## Phase 2: Cluster Config 생성

A5000 스펙 기반 클러스터 설정 파일 2개 생성.

### `cluster_config/a5000_1gpu_validation.json` (TP=1)

```json
{
    "num_nodes": 1,
    "link_bw": 32,
    "link_latency": 0,
    "nodes": [
        {
            "num_instances": 1,
            "cpu_mem": { "mem_size": 128, "mem_bw": 256, "mem_latency": 0 },
            "instances": [
                {
                    "model_name": "meta-llama/Llama-3.1-8B",
                    "hardware": "A5000",
                    "npu_mem": { "mem_size": 24, "mem_bw": 768, "mem_latency": 0 },
                    "npu_num": 1,
                    "npu_group": 1,
                    "pd_type": null
                }
            ],
            "power": {
                "base_node_power": 60,
                "npu": {
                    "A5000": {
                        "idle_power": "<측정값>",
                        "standby_power": "<측정값>",
                        "active_power": "<측정값>",
                        "standby_duration": 18
                    }
                },
                "cpu": { "idle_power": 10, "active_power": 150, "util": 0.15 },
                "dram": { "dimm_size": 32, "idle_power": 2.0, "energy_per_bit": 6.0 },
                "link": { "num_links": 1, "idle_power": 5, "energy_per_bit": 4.0 },
                "nic": { "num_nics": 1, "idle_power": 20 },
                "storage": { "num_devices": 2, "idle_power": 5 }
            }
        }
    ]
}
```

### `cluster_config/a5000_2gpu_tp2_validation.json` (TP=2)

위와 동일하되 `npu_num: 2`, `npu_group: 2`로 변경.  
`link_bw: 32` (PCIe 4.0 x16 단방향 대역폭, NVLink 없음).

---

## Phase 3: vLLM 실측 서빙

### 실험 조건

| 항목 | 값 |
|------|----|
| 모델 | meta-llama/Llama-3.1-8B |
| 데이터셋 | `dataset/sharegpt_req300_rate10_llama.jsonl` (300 req, 10 req/s) |
| 추가 실험 | `dataset/sharegpt_req100_rate10_llama.jsonl` (빠른 검증용) |
| dtype | float16 |
| max_model_len | 4096 (메모리 제약) |

### Step 3-1: TP=1 vLLM 실행

```bash
# 터미널 A: vLLM 서버
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000

# 터미널 B: 전력 로깅
nvidia-smi --query-gpu=timestamp,index,power.draw \
  --format=csv,noheader,nounits -lms 500 \
  -i 0 > validation/vllm_tp1_power.csv

# 터미널 C: 요청 전송 스크립트
python3 validation/send_requests.py \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --server http://localhost:8000 \
  --output validation/vllm_tp1_results.jsonl
```

### Step 3-2: TP=2 vLLM 실행

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --port 8000
```

전력 로깅은 GPU 0, 1 모두 기록.

### `validation/send_requests.py` 구현 내용

- JSONL 파일에서 `arrival_time_ns`, `input_toks`, `output_toks` 읽기
- `arrival_time_ns` 기반 실시간 전송 (Poisson 도착 재현)
- 요청별 TTFT, TPOT 측정 후 JSONL로 저장
- vLLM TTFT: 첫 토큰 스트리밍 수신 시각 기준

---

## Phase 4: 시뮬레이션 실행

동일 데이터셋으로 LLMServingSim 실행.

```bash
# TP=1
python3 main.py \
  --cluster-config cluster_config/a5000_1gpu_validation.json \
  --fp 16 --block-size 16 \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --output validation/sim_tp1_results.csv \
  --num-req 300 --log-interval 1.0

# TP=2
python3 main.py \
  --cluster-config cluster_config/a5000_2gpu_tp2_validation.json \
  --fp 16 --block-size 16 \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --output validation/sim_tp2_results.csv \
  --num-req 300 --log-interval 1.0
```

---

## Phase 5: 비교 분석

### `validation/compare.py` 구현 내용

**지표 계산:**

| 지표 | 계산 방법 |
|------|-----------|
| MAPE | `mean(|pred - real| / real) × 100` |
| Pearson r | `scipy.stats.pearsonr` |
| p50/p99 | 백분위수 비교 |

**TTFT 정의 차이 처리:**
- vLLM TTFT = 첫 토큰 네트워크 수신 시점 (HTTP 오버헤드 포함)
- 시뮬레이터 TTFT = 첫 토큰 계산 완료 시점
- 분석 시 두 값을 별도 컬럼으로 표시하고 MAPE 외 상대 순위(Spearman r)도 보고

**출력물:**
1. `validation/results_summary.csv` — 지표별 MAPE, p50/p99 실측 vs 예측
2. `validation/ttft_cdf.png` — TTFT CDF 비교 (TP=1, TP=2 각각)
3. `validation/tpot_cdf.png` — TPOT CDF 비교
4. `validation/power_timeline.png` — 전력 시계열 (실측 vs 시뮬)
5. `validation/scatter_ttft.png`, `scatter_tpot.png` — 요청별 산포도

### 합격 기준 (목표)

| 지표 | 목표 MAPE |
|------|-----------|
| TTFT p99 | ≤ 20% |
| TPOT p99 | ≤ 15% |
| Throughput | ≤ 10% |
| GPU 평균 전력 | ≤ 15% |

---

## 파일 구조 (완료 후)

```
validation/
  send_requests.py            # vLLM 요청 전송 스크립트
  compare.py                  # 비교 분석 스크립트
  run_vllm_tp1.sh             # TP=1 vLLM 실행 편의 스크립트
  run_vllm_tp2.sh             # TP=2 vLLM 실행 편의 스크립트
  run_sim.sh                  # 시뮬레이션 실행 스크립트
  vllm_tp1_results.jsonl      # vLLM TP=1 실측 결과
  vllm_tp2_results.jsonl      # vLLM TP=2 실측 결과
  vllm_tp1_power.csv          # vLLM TP=1 GPU 전력 로그
  vllm_tp2_power.csv          # vLLM TP=2 GPU 전력 로그
  sim_tp1_results.csv         # 시뮬레이터 TP=1 결과
  sim_tp2_results.csv         # 시뮬레이터 TP=2 결과
  results_summary.csv         # MAPE 요약 테이블
  *.png                       # 비교 그래프

llm_profile/perf_models/A5000/meta-llama/Llama-3.1-8B/
  tp1/layers.csv, attention.csv, predictions/
  tp2/layers.csv, attention.csv, predictions/

cluster_config/
  a5000_1gpu_validation.json
  a5000_2gpu_tp2_validation.json
```

---

## 실행 순서 체크리스트

- [ ] **Phase 1-1**: Layer 프로파일링 TP=1, TP=2
- [ ] **Phase 1-2**: Attention 프로파일링 TP=1, TP=2
- [ ] **Phase 1-3**: 어텐션 예측기 빌드
- [ ] **Phase 1-4**: GPU idle/active/standby 전력 측정
- [ ] **Phase 2**: Cluster config 생성 (전력값 채우기)
- [ ] **Phase 3-1**: vLLM TP=1 서빙 + 실측 데이터 수집
- [ ] **Phase 3-2**: vLLM TP=2 서빙 + 실측 데이터 수집
- [ ] **Phase 4**: 시뮬레이션 TP=1, TP=2 실행
- [ ] **Phase 5**: `compare.py` 실행 → 그래프 및 요약 생성

---

## 주의 사항

1. **메모리**: Llama-3.1-8B FP16 모델 가중치 ~16 GB. TP=1에서 24 GB 중 ~18 GB 사용 (KV 캐시 포함 시 tight). `max_model_len=4096`으로 제한.
2. **NVLink 없음**: TP=2 통신이 PCIe (~32 GB/s)를 통하므로 ALLREDUCE 레이턴시가 NVLink 환경보다 높음. 시뮬레이터 `link_bw=32`로 설정.
3. **vLLM TTFT 정의 차이**: CLAUDE.md 참고 — 시뮬레이터 TTFT는 계산 완료 기준이므로 vLLM보다 낮게 나올 수 있음. Spearman 순위 상관계수로 보완 보고.
4. **HuggingFace 토큰**: LLaMA 모델 다운로드 시 HF_TOKEN 환경변수 필요.
5. **vLLM 0.22.0**: 구버전으로 최신 API와 다를 수 있음. `--max-model-len` 파라미터 지원 여부 확인.
