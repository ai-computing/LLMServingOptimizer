# 00 — 기존 `webapp/` 구조 분석

> DSE 도구가 위에 얹힐 기반. 코드를 새로 쓰기 전에 무엇이 이미 있는지 명확히.

## 1. 스택

| 영역 | 선택 |
|---|---|
| 백엔드 | **FastAPI** + Uvicorn (`--reload` 모드로 개발) |
| 템플릿 | Jinja2 (`webapp/templates/`) |
| 정적 자원 | vanilla JS + CSS (`webapp/static/app.js`, `app.css`) — 번들러/프레임워크 없음 |
| 차트 | Plotly.js (CDN 로드, 서버에서 `figure.to_json()`으로 데이터 전달) |
| 실시간 통신 | **SSE (Server-Sent Events)** — `text/event-stream` |
| 영속 저장소 | 파일시스템 (`status.json`, `metrics.json` per sweep). DB 없음 |
| 동시성 | `asyncio.Semaphore(MAX_CONCURRENT)` — subprocess 풀 |

> PLAN_webapp_dse.md가 Flask일 수도 있다고 추측했지만 **현재 코드는 FastAPI**입니다. DSE는 그대로 FastAPI에 라우트 추가.

## 2. 모듈별 요약

| 파일 | 역할 | 주요 export |
|---|---|---|
| `webapp/app.py` | FastAPI 진입점, 모든 라우트 + SSE 정의 | `app` |
| `webapp/config.py` | 상수 (`MAX_CONCURRENT`, `CONFIG_TIMEOUT_S`, `HW_DEFAULTS`, `SIM_ENV`, dir paths) | constants |
| `webapp/hardware_catalog.py` | `llm_profile/perf_models/` 디렉토리 스캔 → (hw, model) → frozenset(tp) | `build_catalog()`, `list_hardware()`, `list_models_for_hardware()`, `get_tp_options()` |
| `webapp/cluster_io.py` | `cluster_config/*.json` CRUD (path traversal 방어 포함) | `list_configs()`, `load_config()`, `save_config()`, `delete_config()`, `sanitize_filename()` |
| `webapp/cluster_builder.py` | `ConfigSpec` / `InstanceSpec` dataclass + `build_cluster_json` (per-config JSON 빌더) | `ConfigSpec`, `InstanceSpec`, `build_cluster_json()`, `validate_spec()` |
| `webapp/enumerate.py` | scenario (instance_groups + axes) → `list[ConfigSpec]`. 토폴로지 검증 + heterogeneous 필터 | `enumerate_configs()`, `_topology_valid()` |
| `webapp/runner.py` | sweep 실행기. subprocess 풀, status.json 갱신, SSE broadcast, PID cleanup | `run_sweep()`, `_run_one_config()`, `_cleanup_pid_artifacts()`, `cancel_sweep()`, `subscribe_events()` |
| `webapp/parser.py` | 시뮬레이터 log + per-request CSV → metrics dict (TTFT/TPOT/ITL/throughput/power-Wh) | `parse_log()`, `parse_csv()`, `parse_run()`, `is_successful()`, `extract_error_excerpt()` |
| `webapp/plots.py` | 결과 → Plotly figure JSON. config 별 stable 색상 매핑 | `bar_charts()`, `pareto_scatter()`, `axis_line_charts()`, `cdf_charts()`, `all_plots()`, `assign_config_colors()`, `CONFIG_PALETTE` (48색) |

## 3. 데이터 흐름

### 3.1 정상 sweep 흐름 (인덱스 → 진행 → 결과)

```
사용자 입력 (브라우저)
   │  /api/cluster-configs (load existing JSON)
   │  /api/hardware (catalog dropdown)
   ▼
[Cluster Config Builder]  +  [Scenario form (instance_groups, axes)]
   │
   ▼ POST /api/enumerate  →  enumerate.enumerate_configs(scenario, catalog)
   │                          ├─ _topology_valid (NPU divisibility check)
   │                          ├─ heterogeneous combined filter
   │                          └─ ConfigSpec list 반환 (label, instances, tp/pp/dp/pd_layout)
   ▼
[미리보기: N configs, M configs 표시]
   │
   ▼ POST /api/sweeps  →  runner.run_sweep()
   │                       ├─ output/web_sweeps/<sweep_id>/ 생성
   │                       ├─ scenario.json + initial status.json 쓰기
   │                       ├─ 각 ConfigSpec → cluster_builder.build_cluster_json() → configs/<label>.json
   │                       └─ asyncio.Semaphore(MAX_CONCURRENT) 아래 _run_one_config 동시 실행
   │
   ▼ 브라우저 → /sweep/<sweep_id>  (progress 페이지)
   │  SSE: /api/sweeps/<sweep_id>/events  (실시간 진행률)
   │
   ▼ _run_one_config (per config, in subprocess):
   │  1. subprocess: python3 main.py --cluster-config configs/<label>.json ... → runs/<label>.log + .csv
   │  2. asyncio.wait_for(proc.wait(), timeout=workload.timeout_s | CONFIG_TIMEOUT_S)
   │  3. timeout → SIGTERM → SIGKILL
   │  4. parser.parse_run(log, csv) → metrics dict
   │  5. status.json 갱신 + SSE broadcast
   │  6. finally: _cleanup_pid_artifacts(proc.pid) — PID-namespaced trace/workload 제거
   │
   ▼ sweep 완료
   │  status.state = "done" + finished_at stamp
   ▼
[Results 페이지]  /sweep/<sweep_id>/results
   ├─ parser.parse_run으로 재파싱 (ttft_values_ms, itl_values_ms 복구)
   ├─ plots.all_plots(results) → JSON
   ├─ config_colors = assign_config_colors(labels)
   └─ Configuration Matrix 표 + Pareto + CDF + Energy breakdown
```

### 3.2 데이터 흐름 의존성 그래프

```
hardware_catalog ←─── enumerate ───→ cluster_builder ───→ runner ───→ parser ───→ plots
                       │                                    │
                       │                                    └─→ status.json + SSE
                       └─→ cluster_io (load/save user JSONs)
```

## 4. SSE 이벤트 스키마

`/api/sweeps/{id}/events`가 `text/event-stream`으로 push하는 dict 모양:

### 4.1 Snapshot (초기 + 재연결 시)

```jsonc
event: snapshot
data: {
  "sweep_id": "20260601-123012-scenario-1",
  "created_at": "2026-06-01T12:30:12+00:00",
  "state": "running",
  "configs": {
    "tp1_pp1_dp1": {"state": "done", "elapsed_s": 17.3, "metrics": {...}},
    "tp1_pp1_dp2": {"state": "running", "elapsed_s": 4.1},
    ...
  }
}
```

### 4.2 Per-config 업데이트

```jsonc
event: message
data: {
  "label": "tp1_pp1_dp1",
  "state": "done" | "running" | "queued" | "failed" | "cancelled",
  "elapsed_s": 17.3,
  "metrics": {...},      // when state==done
  "error": "exit code 1: TypeError: ...",  // when failed
  "last_log_line": "..."  // optional
}
```

### 4.3 Sweep 상태 전환

```jsonc
event: message
data: {
  "sweep_state": "done" | "failed" | "cancelled",
  "finished_at": "2026-06-01T12:31:45+00:00"  // terminal 시 포함
}
```

### 4.4 Heartbeat

```jsonc
event: message
data: {"type": "heartbeat"}
```

5초마다 keepalive. 클라이언트는 무시.

## 5. `sweep_dir` 디렉토리 레이아웃

```
output/web_sweeps/<sweep_id>/
├── scenario.json              # 입력 (POST /api/sweeps body)
├── status.json                # 진행 상태 (atomic write)
├── metrics.json               # 종료 후 결과 dump (다운로드용)
├── configs/
│   └── <label>.json           # cluster_builder.build_cluster_json() 출력 — main.py 입력
└── runs/
    ├── <label>.log            # main.py stdout (ANSI 코드 포함)
    └── <label>.csv            # main.py --output, per-request 메트릭
```

`status.json` 스키마:
```jsonc
{
  "sweep_id": "...",
  "created_at": "...iso...",
  "finished_at": "...iso...",   // terminal state 시
  "state": "queued" | "running" | "done" | "failed" | "cancelled",
  "configs": {
    "<label>": {
      "state": "...",
      "elapsed_s": float,
      "returncode": int,           // failed 시
      "error": "string",            // failed/timeout 시
      "metrics": {                  // done 시
        "total_token_tp": float,
        "mean_ttft_ms": float,
        "p99_ttft_ms": float,
        ...
        "total_energy_wh": float,   // power block 있을 때만
        "npu_energy_wh": float,
        ...
      }
    }
  }
}
```

## 6. 재사용 가능한 함수 시그니처

DSE가 호출할 main API surface:

### 6.1 카탈로그
```python
from webapp.hardware_catalog import build_catalog, list_hardware, list_models_for_hardware, get_tp_options

catalog: dict[tuple[str, str], frozenset[int]] = build_catalog()
# 예: catalog[("RNGD", "meta-llama/Llama-3.1-8B")] == frozenset({1})

hw_list = list_hardware(catalog)
# 예: ["A6000", "H100", "RNGD", "TPU-v6e-1"]

models = list_models_for_hardware(catalog, "H100")
tps = get_tp_options(catalog, "H100", "meta-llama/Llama-3.1-70B")
```

### 6.2 Cluster JSON
```python
from webapp.cluster_builder import InstanceSpec, ConfigSpec, build_cluster_json

spec = ConfigSpec(
    label="my_cand",
    instances=[InstanceSpec("H100", "meta-llama/Llama-3.1-8B", npu_num=1, npu_group=1, pd_type=None)],
    tp=1, pp=1, dp=1, pd_layout="—",
)
cluster_json: dict = build_cluster_json(
    spec,
    cpu_mem={"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
    link_bw=112, link_latency=0,
    power_template=None,   # or full power dict
)
```

### 6.3 Enumerate
```python
from webapp.enumerate import enumerate_configs, _topology_valid

specs: list[ConfigSpec] = enumerate_configs(
    scenario={
        "instance_groups": [
            {"hardware": "H100", "model": "meta-llama/Llama-3.1-8B", "npu_count": 1, "pd_role": "auto"},
            {"hardware": "RNGD", "model": "meta-llama/Llama-3.1-8B", "npu_count": 1, "pd_role": "auto"},
        ],
        "axes": {"vary_tp": True, "vary_pp": True, "vary_dp": True, "include_pd": True},
    },
    catalog=build_catalog(),
)

# 토폴로지 유효성 직접 검사 (DSE generator의 사전 필터)
is_valid: bool = _topology_valid(instances)
```

### 6.4 Sweep
```python
from webapp.runner import run_sweep, subscribe_events

await run_sweep(
    sweep_id="...",
    configs=[ConfigSpec, ...],
    scenario_json={...},
    sweep_dir=Path("output/web_sweeps/..."),
    workload={
        "dataset": "dataset/...",
        "num_req": 100,
        "phase": "full",
        "power_template": None,   # or {...}
        "timeout_s": 120,
    },
)
```

### 6.5 Parse
```python
from webapp.parser import parse_run, is_successful, extract_error_excerpt

metrics: dict = parse_run(log_path, csv_path)
# metrics keys: total_token_tp, mean_ttft_ms, p99_ttft_ms, ..., total_energy_wh, npu_energy_wh, ...

ok: bool = is_successful(log_path)   # "Simulation results" 마커 검사
err: str = extract_error_excerpt(log_path)
```

### 6.6 Plot
```python
from webapp.plots import all_plots, assign_config_colors, _pareto_frontier

plot_json: dict[str, str] = all_plots(results)
# keys: total_token_tp, mean_ttft_ms, mean_tpot_ms, mean_itl_ms, total_energy_wh,
#       pareto, line_tp, line_pp, line_dp, cdf_ttft, cdf_itl

colors: dict[str, str] = assign_config_colors(labels)

# 현재 2D만 지원. DSE에서 ND Pareto 필요 → 확장
pareto_indices: list[int] = _pareto_frontier([(x, y), ...])
```

## 7. 회피해야 할 부분 / 알려진 함정

### 7.1 ASTRA-Sim heterogeneous P/D deadlock

**문제**: ASTRA-Sim의 collective routing이 다양한 hardware/role 조합에서 deadlock. `output/web_sweeps/20260526-124701-scenario-1`에서 12/20 configs가 120초 안에 progress 0줄로 timeout.

**부분 대응** (`enumerate.py`):
- 모든 instance가 combined-mode이면서 hardware가 섞인 경우 차단
- heterogeneous combined > prefill+decode 패턴 차단

**근본 해결 안 됨**: ASTRA-Sim 소스 수정 필요 (PLAN_webapp_dse_detail.md §10.1 D안, 추정 80–240시간).

**DSE 시사점**: generator가 이 필터를 거치면 deadlock 위험 후보가 자동 차단. 그래도 borderline 4개 정도는 새어나갈 수 있어 timeout으로 graceful fail.

### 7.2 PID-namespaced trace/workload 경로

**배경**: 동시 실행 main.py들이 `astra-sim/inputs/trace/` 공유. PID 격리 없으면 file race → `TypeError: formatter() takes 11 positional arguments but 13 were given` 같은 random 오류.

**현재 처리**:
- `inference_serving/utils.py:PID_TAG = f"pid{os.getpid()}_"` 모듈 레벨 정의
- `trace_generator.py:53, 1814`, `utils.py:get_workload`, `graph_generator.py:18`이 모두 `PID_TAG` 사용
- `webapp/runner.py:_cleanup_pid_artifacts(pid)`가 subprocess 종료 후 PID 디렉토리 정리

**DSE 시사점**: subprocess 띄울 때 같은 인프라 사용 → 추가 신경 쓸 거 없음.

### 7.3 `astra-sim/inputs/system/system.json` 단일 파일

**문제**: 모든 NPU가 단일 `local-mem-bw` 공유. heterogeneous P/D에서 mem_bw 정확도 한계.

**부분 대응** (`config_builder.py`):
- `fcntl.flock` 으로 concurrent 쓰기 race 방지
- decode 인스턴스의 `mem_bw`를 우선 적용 (decode가 memory-bound)

**DSE 시사점**: heterogeneous P/D의 energy/throughput 메트릭은 ±10% 정도 부정확할 수 있음. 사용자에게 명시.

### 7.4 prefill 인스턴스의 NPU doubling

**배경**: `config_builder.py`가 prefill instance의 `npu_num`을 internally 2배로 만들어 sender NPU 표현. 1 prefill 인스턴스 (npu=1) = 토폴로지상 2 NPU.

**DSE 시사점**:
- enumerate가 생성한 ConfigSpec의 `instances`는 사용자가 입력한 npu_num 그대로 보존
- 토폴로지 검증 (`_topology_valid`)에서만 doubling 반영
- 사용자 입력 "physical NPU" 카운트 = sum of npu_num across instances (이미 `webapp/app.py:_phys_npus`가 이렇게 계산)

### 7.5 `MAX_CONCURRENT = min(10, cpu_count()//2)`

**현재값**: 10 (서버 cpu_count=20). 시뮬레이션이 CPU-bound라 이 이상 동시 실행하면 CPU 경합으로 오히려 느려짐.

**DSE 시사점**: 64개 candidate 병렬 실행 시 10개씩 wave로 처리됨. 64 candidate × 평균 60초 / 10 = ~6분.

### 7.6 main.py의 `os.chdir("astra-sim")`

**문제**: 모든 ASTRA-Sim-facing 경로는 `astra-sim/` 기준 상대. 사용자 입력 절대 경로는 깨짐.

**현재 처리**: `runner.py`가 `--cluster-config`에 sweep_dir 내부 상대 경로 전달.

**DSE 시사점**: cluster JSON 파일을 sweep_dir 안에 두기만 하면 OK.

## 8. DSE가 채워야 할 빈자리

기존 webapp은 **scenario 입력이 명시적**입니다 (사용자가 instance_groups를 손으로 정의). DSE는 다음을 추가:

| 영역 | 기존 webapp | DSE 추가 |
|---|---|---|
| **입력** | instance_groups (hw, model, count, pd_role) | resource_pool (hw min/max), objectives, weights, SLO |
| **탐색** | enumerate (parallelism axes 내부) | resource_pool → instance_groups 자동 생성 + parallelism 탐색 |
| **랭킹** | 없음 (Configuration Matrix 표만) | SLO 필터 + N차원 Pareto + weighted score + Top-N + diversity |
| **재랭킹** | 결과 변경 시 재실행 필요 | 가중치 변경 시 즉시 재계산 (재시뮬 없음) |
| **저장** | sweep_dir per run | job DB (SQLite) + spec hash 캐시 |
| **다운로드** | metrics.json + zip | + Top-N JSON, Pareto-only CSV, reproduce CLI |

## 9. 작업 권장 순서

이 문서를 읽은 다음 DSE 작업자는:

1. `cluster_config/single_node_single_instance.json` 손으로 열어 §0.2 (다음 docs)로 이어가기
2. `python3 -c "from webapp.hardware_catalog import build_catalog; print(build_catalog())"` 실행해 카탈로그 직접 확인
3. `webapp/enumerate.py` 읽고 ConfigSpec 만들기 흐름 익히기 (DSE generator가 이걸 재호출)
4. 가장 가벼운 sweep 1개 실행 (`single_node_single_instance.json`, num_req=3) → `output/web_sweeps/<id>/`에서 실제 출력 확인

## 10. 참고 파일

| 목적 | 경로 |
|---|---|
| 프로젝트 전체 가이드 | `CLAUDE.md` |
| Phase 0의 다음 문서 | `docs/dse/01_cluster_config_schema.md` (TBD) |
| 전체 작업 plan | `PLAN_webapp_dse_detail.md` |
| 기존 webapp launcher | `script/serve_webapp.sh` |
