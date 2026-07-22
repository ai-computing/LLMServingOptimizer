# 02 — `main.py` I/O 명세

> DSE의 `runner`가 subprocess로 `python main.py ...`를 호출. 입력 플래그와 출력 (CSV + stdout) 모두 명세화.

## 1. CLI 플래그 카테고리

`python main.py --help` 출력을 5개 카테고리로 분류:

### 1.1 입력 (Input)

| 플래그 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `--cluster-config` | str | `cluster_config/single_node_single_instance.json` | cluster JSON 경로 (`os.chdir("astra-sim")` 후 `../` 자동 부착) |
| `--dataset` | str | `None` | per-request jsonl. 없으면 에러 |
| `--num-req` | int | 100 | dataset에서 처음 N개만 사용 |
| `--fp` | int | 16 | floating point bit (8 / 16 / 32) |
| `--block-size` | int | 16 | KV cache block 단위 (tokens) |
| `--gen` | flag | True | `--gen` 명시하면 False가 됨 (initiation phase skip — 부정 의미. argparse `store_false`) |

### 1.2 스케줄링 (Scheduling)

| 플래그 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `--max-batch` | int | 0 (∞) | 최대 batch 크기. 0이면 무제한 |
| `--max-num-batched-tokens` | int | 2048 | 한 iteration의 max token 수 |
| `--request-routing-policy` | enum | `RR` | `RR` / `RAND` / `CUSTOM` |
| `--expert-routing-policy` | enum | `FAST` | MoE expert routing (`RR` / `RAND` / `FAST` / `CUSTOM`) |
| `--prioritize-prefill` | flag | False | prefill을 decode보다 우선 schedule |

### 1.3 Feature (Optional 시뮬레이션 기능)

| 플래그 | 의미 | cluster_config 의존 |
|---|---|---|
| `--enable-prefix-caching` | RadixAttention 기반 prefix cache | dataset의 `input_tok_ids` 필요 |
| `--enable-prefix-sharing` | 2nd-tier prefix cache pool | `--enable-prefix-caching` 전제 |
| `--prefix-storage` | `None` / `CPU` / `CXL` | CXL이면 cluster JSON에 `cxl_mem` 필요 |
| `--enable-local-offloading` | weight를 NPU local memory에 offload | cluster JSON placement에 `npu` 항목 |
| `--enable-attn-offloading` | attention을 PIM으로 offload | `cpu_mem.pim_config` 필요, dram power 자동 조정 |
| `--enable-sub-batch-interleaving` | sub-batch 단위 interleaving | — |
| `--enable-attn-prediction` | scikit-learn 모델로 real-time prediction | `predictions/*.csv` 필요 |

### 1.4 출력 (Output)

| 플래그 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `--output` | str | `None` | per-request CSV 경로 |
| `--log-interval` | float | 0.5 | stdout에 throughput 시계열 찍는 주기 (sec) |
| `--log-level` | enum | `WARNING` | `WARNING` / `INFO` / `DEBUG` |

### 1.5 백엔드

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--network-backend` | `analytical` | `analytical` (안정) / `ns3` (WIP — 사용 비권장) |

## 2. 출력 CSV (`--output`)

per-request 단위 행. 컬럼 (순서대로):

| 컬럼 | 단위 | 의미 |
|---|---|---|
| `instance id` | int | 인스턴스 인덱스 |
| `request id` | int | 요청 ID |
| `model` | str | 모델 이름 |
| `input` | int | 입력 토큰 수 |
| `output` | int | 출력 토큰 수 |
| `arrival` | ns | dataset의 `arrival_time_ns` |
| `end_time` | ns | request 완료 시뮬 시각 |
| `latency` | ns | end_time - arrival |
| `queuing_delay` | ns | scheduler 큐 대기 시간 |
| `TTFT` | ns | Time to First Token (계산 완료 기준; vLLM 정의와 다름) |
| `TPOT` | ns | Time per Output Token (mean) |
| `ITL` | JSON list[ns] | Inter-Token Latency 분포 (per-token gap) |

`webapp/parser.py:parse_csv`가 이 컬럼을 읽고 ms 단위로 통계 산출.

## 3. stdout 패턴 (parser.py PATTERNS)

`webapp/parser.py:17-32`에 정의된 정규식:

| 메트릭 | 정규식 | 단위 |
|---|---|---|
| `total_latency_s` | `Total latency \(s\):\s+([\d.]+)` | s |
| `req_throughput` | `Request throughput \(req/s\):\s+([\d.]+)` | req/s |
| `prompt_throughput` | `Average prompt throughput \(tok/s\):\s+([\d.]+)` | tok/s |
| `gen_throughput` | `Average generation throughput \(tok/s\):\s+([\d.]+)` | tok/s |
| `total_token_tp` | `Total token throughput \(tok/s\):\s+([\d.]+)` | tok/s |
| `total_requests` | `Total requests:\s+(\d+)` | count |
| `mean_ttft_ms` | `Mean TTFT \(ms\):\s+([\d.]+)` | ms |
| `median_ttft_ms` | `Median TTFT \(ms\):\s+([\d.]+)` | ms |
| `p99_ttft_ms` | `P99 TTFT \(ms\):\s+([\d.]+)` | ms |
| `mean_tpot_ms` | `Mean TPOT \(ms\):\s+([\d.]+)` | ms |
| `median_tpot_ms`, `p99_tpot_ms` | 동일 패턴 | ms |
| `mean_itl_ms`, `median_itl_ms`, `p99_itl_ms` | 동일 패턴 | ms |

성공 마커: `"Simulation results"` 문자열 존재 (`_SUCCESS_MARKER`).

per-second 시계열:
```
[1.0s] Avg prompt throughput: 661.0 tokens/s, Avg generation throughput: 159.0 tokens/s
        ├─Running Instance[0]: 9 reqs, Total # 1 NPUs, Each NPU Memory Usage 15430.51 MB (37.672 % Used)
```
- `NPU_UTIL_RE = r"NPU Memory Usage [\d.]+ MB \(([\d.]+) % Used\)"` — 마지막 값을 `npu_util_pct`로 추출

## 4. Power 출력 (cluster_config에 `power` 블록 있을 때만)

```
                             Power Modeling Results
--------------------------------------------------------------------------------
Total energy consumption (kJ):                                      0.85
--------------------------------------------------------------------------------
Node 0 total energy consumption (kJ):                               0.85
├─ Base Node energy consumption (J):                                130.69
├─ NPU energy consumption (J):                                      321.45
├─ CPU energy consumption (J):                                      306.58
├─ Memory energy consumption (J):                                   17.43
├─ Link energy consumption (J):                                     10.89
├─ NIC energy consumption (J):                                      43.56
└─ Storage energy consumption (J):                                  21.78
--------------------------------------------------------------------------------
Power per 1.0 sec (W): [389.73, 395.46]
```

`webapp/parser.py`가 추출하는 키 (단위: **Wh로 자동 변환**):
- `total_energy_wh = system_total_kJ × 1000 / 3600`
- `base_node_energy_wh`, `npu_energy_wh`, `cpu_energy_wh`, `dram_energy_wh`, `link_energy_wh`, `nic_energy_wh`, `storage_energy_wh` = `device_J / 3600`

> 로그 표기는 `Memory`이지만 metric key는 `dram_energy_wh` (utils.py의 `dram` 키와 일치).

## 5. 종료 코드 / 에러 패턴

| Exit | 의미 | 처리 |
|---|---|---|
| `0` | 정상. `parser.is_successful()` 의 마커도 같이 확인 | `state=done` |
| `1` | Python exception 발생 | `webapp/parser.py:extract_error_excerpt()`가 마지막 Python exception 라인 추출 |
| `-15` (SIGTERM) | timeout 후 webapp이 보낸 신호 | `webapp/runner.py`가 `state=failed`, `error=timeout after Xs` |
| `-9` (SIGKILL) | SIGTERM 10초 후 강제 종료 | 동일 |

자주 보이는 에러:
- `KeyError: ('embedding', N, 0, K)` — perf_db 누락 → `_get_perf_row` nearest-neighbor fallback이 보통 잡음
- `FileNotFoundError: ... attn_prefill_predictions.csv` — incomplete profile → `hardware_catalog._tp_dir_is_complete` 사전 차단
- `TypeError: formatter() missing/takes ... positional arguments` — PID 격리 안 됐을 때의 trace race condition. 이미 `PID_TAG`로 해결됨
- ASTRA-Sim hang (progress 0줄 + timeout) — heterogeneous P/D collective deadlock

## 6. main.py 작동 단계 (CLAUDE.md 발췌 + 확장)

1. **부팅**:
   - `os.chdir("astra-sim")` — 모든 ASTRA-Sim-facing 경로 이 디렉토리 기준
   - `inference_serving.config_builder.build_cluster_config()`로 cluster JSON 검증/변환
   - `astra-sim/inputs/{network/network.yml, system/system.json, memory/memory_expansion.json}` 생성/업데이트
   - PowerModel 인스턴스화 (cluster_config에 `power` 있을 때만)
2. **ASTRA-Sim subprocess 띄움**: `build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra`
3. **메인 루프** (per token / per batch):
   - ASTRA-Sim stdout 읽음 → 어느 (instance_id, batch_id)가 다음 trace 필요한지 확인
   - `scheduler.schedule()` → 다음 batch 결정
   - `generate_trace()` → `inputs/trace/{hw}/{model}/{PID_TAG}instance{N}_batch{M}.txt`
   - `generate_graph()` → Chakra converter 호출 → `inputs/workload/{...}/llm.*`
   - ASTRA-Sim stdin에 workload path 전달
   - ASTRA-Sim이 cycle 반환 → 다음 iteration
4. **종료**: 모든 request done → throughput summary 출력 → CSV 쓰기 → exit 0

## 7. DSE runner가 호출할 명령 예시

`webapp/runner.py:_run_one_config`가 생성하는 명령 (이미 검증됨):

```bash
python3 /path/to/LLMServingSim/main.py \
  --cluster-config output/dse_jobs/<job_id>/configs/<cand_id>.json \
  --fp 16 \
  --block-size 16 \
  --dataset dataset/sharegpt_req100_rate10_llama.jsonl \
  --output output/dse_jobs/<job_id>/runs/<cand_id>.csv \
  --num-req 100 \
  --log-interval 1.0 \
  --log-level WARNING
```

**환경 변수** (`webapp/config.py:SIM_ENV`):
- `LD_LIBRARY_PATH=/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:...` (libprotobuf-23 위치)
- `PATH=$HOME/.local/bin:...` (`python` symlink 필요)
- `cwd=REPO_ROOT` (main.py가 알아서 `os.chdir("astra-sim")`)

## 8. DSE 측 활용 가이드

| 필요한 정보 | 어디서 얻는가 |
|---|---|
| 시뮬레이션 성공/실패 여부 | `parser.is_successful()` + subprocess returncode |
| Throughput | `metrics["total_token_tp"]` (tok/s) |
| Latency p99 | `metrics["p99_ttft_ms"]`, `metrics["p99_tpot_ms"]`, `metrics["p99_itl_ms"]` |
| Power 합산 | `metrics["total_energy_wh"]` |
| Per-device 에너지 | `metrics["*_energy_wh"]` |
| 실패 원인 | `parser.extract_error_excerpt()` |
| 시뮬레이션 wall time | `runner._run_one_config`의 `elapsed = time.monotonic() - start` |
| Per-request CSV (CDF용) | `dataset/output.csv` (DSE가 zip으로 묶어 다운로드 제공) |

DSE의 ranker는 위 메트릭 dict만 받으면 SLO 필터링 / Pareto / weighted score 산출 가능.
