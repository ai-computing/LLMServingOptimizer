# PLAN — Stage 1 (분석 필터) + Stage 2 (llm_profile 예측기) 구현

> LLMServingSim 기반 DSE 도구의 5단계 파이프라인 중 **Stage 1**과 **Stage 2**를 구현한다.
> 두 단계 모두 **ML이 필요 없으며, 시뮬레이터를 호출하지 않는다.** 후보당 O(1)~O(10ms)로 후보 공간을 70~99% 압축하는 것이 목표.
> Claude Code가 이 문서만 보고 처음부터 끝까지 구현할 수 있도록 작성되었다.

---

## 0. 목차

1. 목표와 산출물 / 비목표
2. 사전 조건 (Phase 0이 끝났다고 가정하는 것들)
3. 디렉토리 구조와 파일 책임
4. 데이터 모델 (`schemas.py`)
5. 하드웨어·모델 카탈로그 (`catalog.yaml`)
6. **Stage 1 구현 상세** — 6개 폐쇄형 필터
7. **Stage 2 구현 상세** — llm_profile 테이블 예측기
8. 통합: 파이프라인 진입점과 CLI
9. 테스트 전략 (단위 + 통합 + 회귀)
10. Claude Code 권장 작업 순서 (체크리스트)
11. 막힐 만한 지점 가이드 (FAQ)
12. 부록 A: 알려진 모델 메타데이터
13. 부록 B: 알려진 하드웨어 메타데이터
14. 부록 C: 참고 자료

---

## 1. 목표와 산출물

### 1.1 목표

* **Stage 1**: 사용자 입력(자원 풀 + 제약)으로부터 enumerate된 후보 cluster 조합 리스트에 대해, 다음 5가지 폐쇄형 검사를 적용하여 **물리적으로 불가능하거나 SLO 하한을 위반하는 후보를 제거**한다.
  1. KV 캐시 + 모델 weights + activation 메모리 가능성
  2. TP / PP 나눗셈 조건
  3. Roofline 기반 TTFT / TPOT lower bound
  4. TP all-reduce 통신 lower bound
  5. 전력 / TDP 예산
* **Stage 2**: Stage 1을 통과한 후보들에 대해 `llm_profile/` 디렉토리의 CSV 테이블을 읽어 **시뮬레이터 호출 없이 TTFT_pred / TPOT_pred를 후보당 ~10ms로 추정**한다. 추정값으로 추가 정렬·가지치기를 수행한다.

### 1.2 산출물

* Python 패키지: `webapp_dse/core/{schemas.py, catalog.py, stage1_filters.py, stage2_predictor.py, pipeline.py}`
* CLI: `python -m webapp_dse.cli filter --spec spec.yaml --out filtered.jsonl`
* 테스트: `tests/dse/{test_stage1.py, test_stage2.py, test_pipeline.py}`
* 카탈로그: `webapp_dse/data/{hardware.yaml, models.yaml}`
* 보고서 JSON: 각 단계에서 몇 개가 들어와 몇 개가 통과했는지, 어떤 사유로 reject되었는지 통계

### 1.3 비목표 (다음 단계로 미룸)

* 시뮬레이터 실제 호출 (`runner.py`, Stage 5에 해당) → 별도 PR
* Stage 3 (대리모델), Stage 4 (NSGA-II 탐색) → 별도 PR
* 웹 UI, FastAPI 엔드포인트 → 별도 PR
* PIM / CXL / MoE 전용 로직 → v1.1에서 추가 (v1.0은 dense Llama 계열만)

---

## 2. 사전 조건

이 작업은 다음이 끝났다고 가정한다. **하나라도 빠지면 멈추고 사용자에게 알리거나 직접 채워라.**

### 2.1 Phase 0 산출물 4종

* [ ] `docs/dse/00_existing_webapp_notes.md` — 기존 `webapp/` 분석
* [ ] `docs/dse/01_cluster_config_schema.md` + `cluster_config_schema.json` — cluster_config JSON 스키마
* [ ] `docs/dse/02_main_io_spec.md` — `main.py` CLI 및 출력 명세
* [ ] `docs/dse/03_catalog.yaml` — 지원되는 (hardware, model) 페어와 프로파일 파일 매핑

**없으면**: 작업 중단하고 사용자에게 "Phase 0 산출물 X가 필요하다"고 명시. 부록 A·B의 값을 임시로 사용해도 되지만, **그것은 어디까지나 임시값**이며 실제 `llm_profile/`과 LLMServingSim 소스에서 확인된 값으로 대체해야 함을 README에 기록해야 한다.

### 2.2 환경

* Python 3.10+
* 의존성: `pydantic>=2.0`, `numpy`, `pyyaml`, `pandas`, `pytest`, `scipy` (Spearman 상관계수용)
* LLMServingSim 저장소가 같은 작업 트리에 있어야 함. `llm_profile/` 경로를 환경변수 `LLMSERVINGSIM_ROOT`로 받거나, `webapp_dse/config.py`에 기본 경로(`../`)를 두고 override 가능하게.

### 2.3 Stage 0 산출물 (입력)

Stage 1의 입력은 `List[CandidateConfig]`이다. 이는 Stage 0(generator.py, 다른 PR)이 만든다. **이번 작업에서는 generator.py가 없어도 동작하도록**, `examples/` 디렉토리에 손으로 만든 JSONL 후보 리스트를 두고 그것을 입력으로 받을 수 있어야 한다.

---

## 3. 디렉토리 구조와 파일 책임

```
webapp_dse/
├── __init__.py
├── config.py                 # 환경변수, 기본 경로
├── core/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic 모델: CandidateConfig, FilterResult, PredictionResult, Spec
│   ├── catalog.py            # hardware/model 카탈로그 로더
│   ├── stage1_filters.py     # 5가지 폐쇄형 필터 + 통합 함수
│   ├── stage2_predictor.py   # llm_profile CSV 로더 + TTFT/TPOT 예측
│   └── pipeline.py           # Stage 1 → Stage 2 파이프 + 통계 수집
├── data/
│   ├── hardware.yaml         # HW 카탈로그 (부록 B 기반)
│   └── models.yaml           # 모델 카탈로그 (부록 A 기반)
├── cli.py                    # `python -m webapp_dse.cli filter ...`
└── examples/
    ├── spec_llama70b.yaml          # 사용자 spec 예시
    └── candidates_sample.jsonl     # 손으로 만든 50개짜리 후보 리스트 (테스트용)

tests/dse/
├── __init__.py
├── conftest.py
├── test_schemas.py
├── test_catalog.py
├── test_stage1.py            # 각 필터의 단위 테스트
├── test_stage2.py            # 예측기 단위 테스트
└── test_pipeline.py          # end-to-end

docs/dse/
└── 04_stage1_stage2_design.md   # 이 PLAN의 요약본 (구현 완료 후 작성)
```

**디렉토리 생성 원칙**: 기존 `webapp/`이나 LLMServingSim 본체 코드는 절대 건드리지 않는다. 모든 신규 코드는 `webapp_dse/` 하위에만 둔다. `llm_profile/` CSV는 **읽기 전용**으로만 접근한다.

---

## 4. 데이터 모델 (`webapp_dse/core/schemas.py`)

모든 데이터는 Pydantic v2 모델로 표현한다. 다른 모듈은 이 파일의 모델만 import한다.

### 4.1 입력측

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class HardwareEntry(BaseModel):
    """카탈로그의 한 하드웨어."""
    name: str                     # "H100", "A6000", "TPU-v6e-1"
    hbm_bytes: int                # 80 * 1024**3 같은 형태
    hbm_bw_bytes_s: int           # HBM 대역폭, 바이트/초 (예: 3.35e12)
    peak_flops_bf16: int          # peak BF16 FLOPS, FLOP/초
    peak_flops_fp16: Optional[int] = None
    tdp_w: int                    # 단일 디바이스 TDP, 와트
    interconnect_bw_bytes_s: int  # 노드 내 GPU 간 대역폭 (NVLink/PCIe 등), 바이트/초

class ModelEntry(BaseModel):
    """카탈로그의 한 모델."""
    name: str                                     # "meta-llama/Llama-3.1-70B"
    num_params: int                               # 총 파라미터 수
    num_layers: int                               # L
    hidden_size: int                              # 4096 등
    num_attention_heads: int                      # H_q
    num_kv_heads: int                             # H_kv (GQA에서 < H_q)
    head_dim: int                                 # 일반적으로 hidden_size / num_attention_heads
    architecture: Literal["dense", "moe"] = "dense"
    num_experts: Optional[int] = None             # MoE 전용
    num_experts_per_token: Optional[int] = None   # MoE 전용

class CandidateConfig(BaseModel):
    """Stage 0 generator가 만드는 후보 1개. Stage 1의 입력."""
    candidate_id: str                             # "c00042"
    model_name: str                               # ModelEntry.name과 매칭
    hardware: dict[str, int]                      # {"H100": 4} 또는 {"A6000": 8, "H100": 2}
    parallelism: dict[str, int]                   # {"TP": 4, "PP": 1, "DP": 1}
    pd_disagg: Optional[dict[str, dict]] = None   # 이질/분리형: 부록 D 참조 (v1.0은 None만 지원)
    batch_size: int                               # max batch (--max-batch와 매칭, 0이면 unlimited)
    max_seq_len: int                              # 평균 또는 95th 시퀀스 길이 (KV cache 크기 산정용)
    avg_prompt_len: int                           # 평균 prompt 길이 (prefill FLOPs용)
    dtype: Literal["bf16", "fp16", "fp8", "int4"] = "bf16"
    features: dict[str, bool] = Field(default_factory=dict)   # prefix_caching, chunked_prefill 등

class SLOSpec(BaseModel):
    """사용자 입력의 SLO 부분."""
    ttft_p99_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    throughput_min_tok_s: Optional[float] = None
    power_max_w: Optional[float] = None

class Spec(BaseModel):
    """사용자 입력 전체 (examples/spec_*.yaml에서 로드)."""
    model: ModelEntry
    workload: dict                       # {dataset_path, num_req, qps, ...} — 이번 PR에서는 traversal만
    slo: SLOSpec
    safety_margin_mem: float = 0.85      # weights+KV+act ≤ safety_margin × HBM
    safety_margin_power: float = 1.00    # power_actual ≤ safety_margin × budget
    util_factor: float = 0.75            # GPU 평균 사용률 가정
```

### 4.2 출력측 (필터·예측 결과)

```python
class FilterReason(BaseModel):
    """필터에 의해 reject된 사유 한 건."""
    filter_name: Literal["memory", "divisibility", "roofline_decode", 
                          "roofline_prefill", "communication", "power"]
    detail: str                          # 사람이 읽을 수 있는 한 줄 (예: "weights+KV=420GB > 0.85×320GB=272GB")
    computed_value: float                # 계산된 좌변 값
    threshold: float                     # 임계값 (우변)

class Stage1Result(BaseModel):
    candidate_id: str
    passed: bool
    reasons: list[FilterReason] = Field(default_factory=list)   # 모든 reject 사유 (early-exit 모드에선 1개)
    # 통과한 후보의 분석 lower bound 값들 (Stage 2가 sanity check 용으로 재사용)
    decode_tpot_lb_ms: Optional[float] = None
    prefill_ttft_lb_ms: Optional[float] = None
    total_kv_bytes: Optional[int] = None
    total_power_w: Optional[float] = None

class Stage2Prediction(BaseModel):
    candidate_id: str
    ttft_pred_ms: float
    tpot_pred_ms: float
    # 합산에 쓴 op별 breakdown (디버깅용)
    breakdown: dict[str, float] = Field(default_factory=dict)   # {"dense": 12.3, "attention": 8.7, ...}
    # 사용한 프로파일 파일 경로 (재현성)
    profile_source: str

class PipelineReport(BaseModel):
    """Stage 1 + Stage 2를 모두 거친 후의 최종 보고서."""
    total_input: int
    stage1_passed: int
    stage2_passed: int                   # Stage 2는 reject가 아니라 정렬만 하므로 == stage1_passed
    stage1_rejections_by_reason: dict[str, int]
    survivors: list[dict]                # {candidate_id, ttft_pred, tpot_pred, rank}
    elapsed_s: dict[str, float]          # {"stage1": 0.12, "stage2": 4.5}
```

**구현 메모**: Pydantic v2의 `model_dump()` / `model_validate()`를 사용. 모든 시간 단위는 **ms로 통일**, 메모리는 **bytes로 통일**, 대역폭은 **bytes/s로 통일**. CSV 원본(`time_us`)에서 읽을 때 `× 1000`으로 ns→μs를 거치지 말고, μs→ms로 `/ 1000`만 한다. 단위 혼동이 가장 흔한 버그 원인이다.

---

## 5. 카탈로그 (`webapp_dse/data/{hardware,models}.yaml`)

부록 A·B의 값으로 시작한다. 이 두 파일은 사용자가 사후 편집할 수 있어야 하므로 코드와 분리.

```yaml
# hardware.yaml
H100:
  hbm_bytes: 85899345920        # 80 GiB
  hbm_bw_bytes_s: 3350000000000 # 3.35 TB/s (SXM5)
  peak_flops_bf16: 989000000000000   # 989 TFLOPS (Tensor Core, dense, no sparsity)
  tdp_w: 700
  interconnect_bw_bytes_s: 900000000000   # NVLink 900 GB/s aggregate
A6000:
  hbm_bytes: 51539607552        # 48 GiB
  hbm_bw_bytes_s: 768000000000  # 768 GB/s
  peak_flops_bf16: 154800000000000   # ~155 TFLOPS BF16
  tdp_w: 300
  interconnect_bw_bytes_s: 112500000000   # PCIe 4.0 x16 ~ 32 GB/s 양방향 ≈ 64 GB/s aggregate; 보수적 NVLink 없음
TPU-v6e-1:
  hbm_bytes: 34359738368        # 32 GiB
  hbm_bw_bytes_s: 1640000000000 # 1.64 TB/s
  peak_flops_bf16: 918000000000000   # 918 TFLOPS BF16
  tdp_w: 350
  interconnect_bw_bytes_s: 460000000000   # 460 GB/s ICI
```

```yaml
# models.yaml
"meta-llama/Llama-3.1-8B":
  num_params: 8030000000
  num_layers: 32
  hidden_size: 4096
  num_attention_heads: 32
  num_kv_heads: 8
  head_dim: 128
  architecture: dense
"meta-llama/Llama-3.1-70B":
  num_params: 70554000000
  num_layers: 80
  hidden_size: 8192
  num_attention_heads: 64
  num_kv_heads: 8
  head_dim: 128
  architecture: dense
"mistralai/Mixtral-8x7B-v0.1":
  num_params: 46700000000
  num_layers: 32
  hidden_size: 4096
  num_attention_heads: 32
  num_kv_heads: 8
  head_dim: 128
  architecture: moe
  num_experts: 8
  num_experts_per_token: 2
"microsoft/Phi-mini-MoE-instruct":
  # 부록 A 참조 (실제 값은 Phase 0에서 확정)
  num_params: 7600000000
  num_layers: 32
  hidden_size: 3072
  num_attention_heads: 32
  num_kv_heads: 8
  head_dim: 96
  architecture: moe
  num_experts: 16
  num_experts_per_token: 2
```

### catalog.py

```python
import yaml
from pathlib import Path
from .schemas import HardwareEntry, ModelEntry

def load_hardware_catalog(path: Path | None = None) -> dict[str, HardwareEntry]:
    p = path or Path(__file__).parent.parent / "data" / "hardware.yaml"
    raw = yaml.safe_load(p.read_text())
    return {name: HardwareEntry(name=name, **vals) for name, vals in raw.items()}

def load_model_catalog(path: Path | None = None) -> dict[str, ModelEntry]:
    p = path or Path(__file__).parent.parent / "data" / "models.yaml"
    raw = yaml.safe_load(p.read_text())
    return {name: ModelEntry(name=name, **vals) for name, vals in raw.items()}
```

**검증 요구**: `catalog.py` 임포트 시 즉시 두 파일을 로드하고 Pydantic이 schema 위반을 잡도록 한다. 잘못된 yaml은 import time에 죽어야 한다.

---

## 6. Stage 1 구현 상세 — `webapp_dse/core/stage1_filters.py`

### 6.1 설계 원칙

* **각 필터는 순수 함수**다. (no I/O, no global state). 입력: `CandidateConfig`, `ModelEntry`, `HardwareEntry` (또는 `HardwareEntry`의 dict), `Spec`. 출력: `Optional[FilterReason]` (None이면 통과).
* **모든 단위를 SI**로. 메모리는 bytes, 시간은 seconds 또는 milliseconds (반환할 때만 ms로). 카탈로그 값과 단위가 어긋나면 catalog.py가 잡는다.
* **early-exit 모드와 collect-all 모드**를 둘 다 지원. 기본은 collect-all (사용자 디버깅에 더 유용). spec에 `early_exit: true`가 있으면 첫 reject에서 멈춤.
* 디버그용 logging은 `logging.DEBUG` 레벨에서만. 통과/실패는 반드시 `FilterReason`의 `computed_value`/`threshold`로 정량적 기록.

### 6.2 6가지 필터 함수 시그너처

```python
import math
from typing import Optional
from .schemas import (CandidateConfig, ModelEntry, HardwareEntry, 
                       Spec, FilterReason, Stage1Result)

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int4": 0.5}

# ---------- 0. 헬퍼 ----------

def total_device_count(c: CandidateConfig) -> int:
    return sum(c.hardware.values())

def aggregate_hbm_bytes(c: CandidateConfig, hw_catalog: dict[str, HardwareEntry]) -> int:
    return sum(n * hw_catalog[t].hbm_bytes for t, n in c.hardware.items())

def aggregate_tdp_w(c: CandidateConfig, hw_catalog: dict[str, HardwareEntry]) -> int:
    return sum(n * hw_catalog[t].tdp_w for t, n in c.hardware.items())

def model_weight_bytes(m: ModelEntry, dtype: str) -> int:
    return int(m.num_params * DTYPE_BYTES[dtype])

def kv_cache_bytes_per_token(m: ModelEntry, dtype: str) -> int:
    """2 × L × H_kv × d_head × bytes  (단일 시퀀스, 단일 토큰)"""
    return int(2 * m.num_layers * m.num_kv_heads * m.head_dim * DTYPE_BYTES[dtype])

def activation_bytes_estimate(m: ModelEntry, batch: int, seq_len: int, dtype: str) -> int:
    """
    아주 보수적인 prefill activation 추정.
    Llama 계열은 layer당 hidden + intermediate(보통 4×hidden 또는 SwiGLU의 ~2.67×hidden)을 유지.
    여기서는 layer 1개분의 peak activation만 잡는다 (재계산/오프로딩으로 1 layer만 유지된다 가정).
    """
    intermediate_factor = 3   # SwiGLU 보수치
    bytes_per_elem = DTYPE_BYTES[dtype]
    return int(batch * seq_len * m.hidden_size * (1 + intermediate_factor) * bytes_per_elem)


# ---------- 1. 메모리 가능성 ----------

def check_memory(c: CandidateConfig, m: ModelEntry,
                 hw_catalog: dict[str, HardwareEntry], spec: Spec
                 ) -> Optional[FilterReason]:
    """
    디바이스당 (weights/(TP×PP)) + KV/TP + act/TP 가 디바이스 HBM 한도 이내인지.
    
    v1.0: hardware가 동질(단일 HW 타입) 또는 P/D 분리가 None인 경우만 처리.
    혼합 HW + 분리는 v1.1.
    """
    if len(c.hardware) > 1 and c.pd_disagg is None:
        # 혼합 HW를 그냥 합산해서 처리 — 보수적
        # (정확하게는 가장 작은 HBM의 디바이스가 병목이 되므로 거기에 맞춤)
        smallest_hbm = min(hw_catalog[t].hbm_bytes for t in c.hardware)
    else:
        only_hw = next(iter(c.hardware))
        smallest_hbm = hw_catalog[only_hw].hbm_bytes

    TP = c.parallelism["TP"]
    PP = c.parallelism["PP"]
    weights = model_weight_bytes(m, c.dtype)
    kv_total = kv_cache_bytes_per_token(m, c.dtype) * c.batch_size * c.max_seq_len if c.batch_size > 0 \
               else kv_cache_bytes_per_token(m, c.dtype) * c.max_seq_len  # batch=0 (무제한)이면 1개로 추정
    act_total = activation_bytes_estimate(m, max(c.batch_size, 1), c.max_seq_len, c.dtype)

    per_dev = weights / (TP * PP) + kv_total / TP + act_total / TP
    limit = spec.safety_margin_mem * smallest_hbm

    if per_dev > limit:
        return FilterReason(
            filter_name="memory",
            detail=(f"weights/(TP×PP)={weights/(TP*PP):.2e} + KV/TP={kv_total/TP:.2e} "
                    f"+ act/TP={act_total/TP:.2e} = {per_dev:.2e} B > "
                    f"safety×HBM={limit:.2e} B"),
            computed_value=per_dev,
            threshold=limit,
        )
    return None


# ---------- 2. TP/PP 나눗셈 ----------

def check_divisibility(c: CandidateConfig, m: ModelEntry,
                        hw_catalog: dict[str, HardwareEntry], spec: Spec
                        ) -> Optional[FilterReason]:
    TP, PP = c.parallelism["TP"], c.parallelism["PP"]
    DP = c.parallelism.get("DP", 1)
    replicas = c.parallelism.get("replicas", 1)

    # 1) heads
    if m.num_attention_heads % TP != 0:
        return FilterReason("divisibility",
            f"num_attention_heads={m.num_attention_heads} not divisible by TP={TP}",
            float(m.num_attention_heads % TP), 0.0)
    if m.num_kv_heads % TP != 0:
        return FilterReason("divisibility",
            f"num_kv_heads={m.num_kv_heads} not divisible by TP={TP} (GQA constraint)",
            float(m.num_kv_heads % TP), 0.0)
    # 2) layers
    if m.num_layers % PP != 0:
        return FilterReason("divisibility",
            f"num_layers={m.num_layers} not divisible by PP={PP}",
            float(m.num_layers % PP), 0.0)
    # 3) GPU 수 = TP×PP×DP×replicas
    total = TP * PP * DP * replicas
    n_dev = total_device_count(c)
    if total != n_dev:
        return FilterReason("divisibility",
            f"TP×PP×DP×replicas={total} != total devices={n_dev}",
            float(total), float(n_dev))
    return None


# ---------- 3a. Roofline: Decode TPOT lower bound ----------

def check_roofline_decode(c: CandidateConfig, m: ModelEntry,
                           hw_catalog: dict[str, HardwareEntry], spec: Spec
                           ) -> Optional[FilterReason]:
    if spec.slo.tpot_p99_ms is None:
        return None  # SLO 없으면 스킵

    TP = c.parallelism["TP"]
    PP = c.parallelism["PP"]
    # 단일 token decode는 weights 전체 + 현재까지의 KV cache를 한 번 fetch.
    weights_per_dev = model_weight_bytes(m, c.dtype) / (TP * PP)
    # 단순화: KV는 평균 max_seq_len/2까지 누적되어 있다고 가정 (decode 중간점)
    kv_per_dev = (kv_cache_bytes_per_token(m, c.dtype) * c.max_seq_len // 2) / TP

    # 가장 느린 HW가 병목 (혼합인 경우)
    min_hbm_bw = min(hw_catalog[t].hbm_bw_bytes_s for t in c.hardware)

    tpot_lb_s = (weights_per_dev + kv_per_dev) / min_hbm_bw
    tpot_lb_ms = tpot_lb_s * 1000

    if tpot_lb_ms > spec.slo.tpot_p99_ms:
        return FilterReason("roofline_decode",
            f"decode TPOT lower bound {tpot_lb_ms:.1f}ms > SLO {spec.slo.tpot_p99_ms:.1f}ms",
            tpot_lb_ms, spec.slo.tpot_p99_ms)
    return None


# ---------- 3b. Roofline: Prefill TTFT lower bound ----------

def check_roofline_prefill(c: CandidateConfig, m: ModelEntry,
                            hw_catalog: dict[str, HardwareEntry], spec: Spec
                            ) -> Optional[FilterReason]:
    if spec.slo.ttft_p99_ms is None:
        return None

    TP = c.parallelism["TP"]
    # prefill FLOPs ≈ 2 × N_params × T_prompt × batch  (FMA factor 2)
    batch = max(c.batch_size, 1)
    prefill_flops = 2 * m.num_params * c.avg_prompt_len * batch

    # 가장 느린 HW (FLOPS)가 병목
    min_peak_flops = min(hw_catalog[t].peak_flops_bf16 for t in c.hardware)

    ttft_lb_s = prefill_flops / (TP * min_peak_flops)
    ttft_lb_ms = ttft_lb_s * 1000

    if ttft_lb_ms > spec.slo.ttft_p99_ms:
        return FilterReason("roofline_prefill",
            f"prefill TTFT lower bound {ttft_lb_ms:.1f}ms > SLO {spec.slo.ttft_p99_ms:.1f}ms",
            ttft_lb_ms, spec.slo.ttft_p99_ms)
    return None


# ---------- 4. TP all-reduce 통신 ----------

def check_communication(c: CandidateConfig, m: ModelEntry,
                         hw_catalog: dict[str, HardwareEntry], spec: Spec
                         ) -> Optional[FilterReason]:
    if spec.slo.tpot_p99_ms is None:
        return None
    TP = c.parallelism["TP"]
    if TP == 1:
        return None  # all-reduce 없음

    batch = max(c.batch_size, 1)
    # Decode는 토큰 1개만 처리 → seqlen=1
    msg_bytes = batch * 1 * m.hidden_size * DTYPE_BYTES[c.dtype]
    # 가장 느린 interconnect가 병목
    min_link_bw = min(hw_catalog[t].interconnect_bw_bytes_s for t in c.hardware)

    # ring all-reduce: 2(TP-1)/TP × msg / bw
    one_allreduce_s = 2.0 * (TP - 1) / TP * msg_bytes / min_link_bw
    # decode 1 step당 layer마다 2번 (attention + FFN 뒤)
    total_comm_per_token_s = m.num_layers * 2 * one_allreduce_s
    total_comm_per_token_ms = total_comm_per_token_s * 1000

    if total_comm_per_token_ms > spec.slo.tpot_p99_ms:
        return FilterReason("communication",
            f"TP all-reduce per-token lower bound {total_comm_per_token_ms:.1f}ms > "
            f"TPOT SLO {spec.slo.tpot_p99_ms:.1f}ms (msg={msg_bytes}B, link_bw={min_link_bw:.2e}B/s)",
            total_comm_per_token_ms, spec.slo.tpot_p99_ms)
    return None


# ---------- 5. 전력 ----------

def check_power(c: CandidateConfig, m: ModelEntry,
                hw_catalog: dict[str, HardwareEntry], spec: Spec
                ) -> Optional[FilterReason]:
    if spec.slo.power_max_w is None:
        return None
    expected_w = sum(n * hw_catalog[t].tdp_w * spec.util_factor for t, n in c.hardware.items())
    limit = spec.slo.power_max_w * spec.safety_margin_power
    if expected_w > limit:
        return FilterReason("power",
            f"expected power {expected_w:.0f}W (util={spec.util_factor}) > budget {limit:.0f}W",
            expected_w, limit)
    return None
```

### 6.3 통합 함수

```python
ALL_FILTERS = [
    check_memory,
    check_divisibility,
    check_roofline_decode,
    check_roofline_prefill,
    check_communication,
    check_power,
]

def apply_stage1(c: CandidateConfig, m: ModelEntry,
                  hw_catalog: dict[str, HardwareEntry], spec: Spec,
                  early_exit: bool = False) -> Stage1Result:
    reasons: list[FilterReason] = []
    for f in ALL_FILTERS:
        r = f(c, m, hw_catalog, spec)
        if r is not None:
            reasons.append(r)
            if early_exit:
                break
    passed = len(reasons) == 0

    # 통과한 후보의 lower bound 값들도 함께 계산해서 반환 (Stage 2 sanity check용)
    lb_decode_ms = lb_prefill_ms = None
    kv_total = power_w = None
    if passed:
        # 이미 위 함수들이 계산한 적 있지만 통과한 후보에 대해 다시 계산해 채워줌
        # (성능이 문제되면 함수가 값+사유를 함께 반환하도록 리팩터)
        TP = c.parallelism["TP"]; PP = c.parallelism["PP"]
        weights_per_dev = model_weight_bytes(m, c.dtype) / (TP * PP)
        kv_total = kv_cache_bytes_per_token(m, c.dtype) * max(c.batch_size, 1) * c.max_seq_len
        kv_per_dev = (kv_total // 2) / TP
        min_hbm_bw = min(hw_catalog[t].hbm_bw_bytes_s for t in c.hardware)
        lb_decode_ms = (weights_per_dev + kv_per_dev) / min_hbm_bw * 1000

        batch = max(c.batch_size, 1)
        prefill_flops = 2 * m.num_params * c.avg_prompt_len * batch
        min_peak = min(hw_catalog[t].peak_flops_bf16 for t in c.hardware)
        lb_prefill_ms = prefill_flops / (TP * min_peak) * 1000

        power_w = sum(n * hw_catalog[t].tdp_w * spec.util_factor for t, n in c.hardware.items())

    return Stage1Result(
        candidate_id=c.candidate_id,
        passed=passed,
        reasons=reasons,
        decode_tpot_lb_ms=lb_decode_ms,
        prefill_ttft_lb_ms=lb_prefill_ms,
        total_kv_bytes=kv_total,
        total_power_w=power_w,
    )
```

### 6.4 Stage 1 단위 테스트 케이스 (`tests/dse/test_stage1.py`)

각 필터마다 **통과 케이스 1개 + 실패 케이스 1개**를 최소 작성. 추가로:

* **메모리**: Llama-3-70B + TP=1 + H100 단일 → reject (weights만 140GB, HBM 80GB). TP=4, H100×4 → 통과.
* **나눗셈**: Llama-3-70B (kv_heads=8) + TP=3 → reject. TP=8 → 통과.
* **Roofline decode**: H100 × TP=1 + Llama-3-70B + max_seq_len=128k + TPOT SLO 50ms. weights 140GB / HBM BW 3.35TB/s ≈ 42ms (통과 보더라인). max_seq_len 128k + batch 32면 KV가 큼 → reject 가능. **수치를 직접 계산해 fixtures에 둬라**.
* **Roofline prefill**: Llama-3-8B + 32k prompt + TTFT SLO 100ms + H100 TP=1. 2×8B×32k×1 / (1×989T) ≈ 520ms → reject. TP=8이면 ~65ms → 통과.
* **통신**: A6000 (no NVLink, PCIe 4.0 ~32GB/s) + TP=8 + Llama-3-70B + batch=128 + TPOT 100ms. → reject 예상. **계산해서 fixture로 확인**.
* **전력**: H100 × 16, TDP 700W, util 0.75 → 8400W. budget 5000W → reject.

테스트 예시:

```python
import pytest
from webapp_dse.core.schemas import CandidateConfig, SLOSpec, Spec
from webapp_dse.core.catalog import load_hardware_catalog, load_model_catalog
from webapp_dse.core.stage1_filters import apply_stage1

@pytest.fixture
def hw_catalog():
    return load_hardware_catalog()

@pytest.fixture
def model_catalog():
    return load_model_catalog()

@pytest.fixture
def llama70b(model_catalog):
    return model_catalog["meta-llama/Llama-3.1-70B"]

def test_memory_reject_single_h100(hw_catalog, llama70b):
    spec = Spec(model=llama70b, workload={}, slo=SLOSpec())
    c = CandidateConfig(
        candidate_id="t1",
        model_name="meta-llama/Llama-3.1-70B",
        hardware={"H100": 1},
        parallelism={"TP": 1, "PP": 1, "DP": 1, "replicas": 1},
        batch_size=1,
        max_seq_len=4096,
        avg_prompt_len=2048,
    )
    r = apply_stage1(c, llama70b, hw_catalog, spec)
    assert not r.passed
    assert any(reason.filter_name == "memory" for reason in r.reasons)

def test_memory_pass_h100_tp4(hw_catalog, llama70b):
    spec = Spec(model=llama70b, workload={}, slo=SLOSpec())
    c = CandidateConfig(
        candidate_id="t2",
        model_name="meta-llama/Llama-3.1-70B",
        hardware={"H100": 4},
        parallelism={"TP": 4, "PP": 1, "DP": 1, "replicas": 1},
        batch_size=8,
        max_seq_len=4096,
        avg_prompt_len=2048,
    )
    r = apply_stage1(c, llama70b, hw_catalog, spec)
    # 분기점: weights/4=35GB, KV는 작아서 통과해야 함
    assert all(reason.filter_name != "memory" for reason in r.reasons)
```

**숙제: 테스트를 작성하면서 "왜 이 수치인가"를 코멘트로 남겨라.** 미래의 자신이 임계값을 바꾸려 할 때 이유를 알 수 있도록.

### 6.5 회귀 테스트

* `examples/candidates_sample.jsonl`에 알려진 통과·실패 케이스 50개를 두고, Stage 1을 돌렸을 때 통과 개수가 기대치 ±5% 안인지 확인. **이 50개는 손으로 작성하고 각각에 대해 통과/실패 예상을 적어두는 게 좋다.**

---

## 7. Stage 2 구현 상세 — `webapp_dse/core/stage2_predictor.py`

### 7.1 llm_profile 디렉토리 구조

(2026.02 v1.0.0 기준; Phase 0에서 `docs/dse/03_catalog.yaml`에 실제 경로를 확정해두고, 본 모듈은 이것을 참조한다.)

```
llm_profile/perf/<hardware>/<model>/<variant>/
├── meta.yaml             # engine flags, sweep specs, skew_fit 요약
└── tp<N>/
    ├── dense.csv         # tokens → latency (token-level ops: linear, activation, RMSNorm 등)
    ├── per_sequence.csv  # n_seq → latency (per-sequence ops: tokenizer overhead, scheduling 등)
    ├── attention.csv     # (pc, kv_pre, n_dec, kv_dec) → latency
    ├── moe.csv           # (tokens, experts) → latency  (MoE 모델에만)
    ├── skew.csv          # raw heterogeneous-decode samples
    └── skew_fit.csv      # 버킷별 fit α 테이블
```

* 시간 단위는 **μs (마이크로초)**, 컬럼명은 `time_us`로 가정 (실제 컬럼명은 Phase 0 산출물에서 확인).
* `<variant>`는 실험 변형 ID (예: `vllm_baseline`, `chunked_prefill`). 기본값은 `default`.

### 7.2 디렉토리 스캔 + CSV 로더

```python
from pathlib import Path
import pandas as pd
from functools import lru_cache

class ProfileNotFoundError(FileNotFoundError):
    pass

@lru_cache(maxsize=None)
def _load_csv(path: Path) -> pd.DataFrame:
    """CSV 한 번 로드한 뒤 process 내에서 캐시. 후보가 많으면 같은 파일이 수천 번 조회됨."""
    if not path.exists():
        raise ProfileNotFoundError(f"profile CSV not found: {path}")
    df = pd.read_csv(path)
    return df

def _profile_dir(root: Path, hardware: str, model: str, variant: str = "default") -> Path:
    # 카탈로그 키와 디렉토리 이름이 다를 수 있으므로 매핑 테이블이 필요할 수 있음
    # Phase 0 산출물(03_catalog.yaml)에서 hw/model_name → 디렉토리명 매핑을 가져온다고 가정
    # 여기선 1:1 매핑으로 단순화
    model_dir = model.replace("/", "--")    # "meta-llama/Llama-3.1-70B" → "meta-llama--Llama-3.1-70B"
    return root / "perf" / hardware / model_dir / variant
```

**중요한 가정**: `<model>` 디렉토리명에 `/`가 들어가는 hugging face 형식의 이름은 `--` 등으로 escape됐을 수 있다. **실제 디렉토리명을 Phase 0에서 확인하고 매핑을 명시**. 모르겠으면 코드를 멈추고 사용자에게 묻는다.

### 7.3 합산기

#### 7.3.1 dense.csv

dense.csv는 token 수를 키로 한 token-level op의 합산 latency. **하나의 row가 layer 1개분이 아니라 모델 전체의 dense layer 묶음을 의미하는지, 아니면 layer당 값인지**는 Phase 0에서 반드시 확인.

* 만약 row가 모델 전체 dense의 합산 latency라면: `dense_latency_us = lookup(dense.csv, tokens=N)`.
* 만약 row가 op 1개 또는 layer 1개라면: `× num_layers` 또는 `× layer_op_count`.

가정 1(모델 전체 합산)로 우선 구현하고, Phase 0에서 가정 2가 맞다면 즉시 수정 가능하도록 분기를 둔다.

```python
def lookup_dense(df: pd.DataFrame, tokens: int) -> float:
    """가장 가까운 tokens 값의 latency(us)를 반환. interpolation은 선형."""
    df_sorted = df.sort_values("tokens")
    xs = df_sorted["tokens"].values
    ys = df_sorted["time_us"].values
    if tokens <= xs[0]: return float(ys[0])
    if tokens >= xs[-1]: return float(ys[-1])
    import numpy as np
    return float(np.interp(tokens, xs, ys))
```

#### 7.3.2 attention.csv

키가 4D: `(pc, kv_pre, n_dec, kv_dec)`.

* `pc`: prefill chunk tokens (현재 step에서 prefill되는 토큰 수)
* `kv_pre`: prefill 단계의 누적 KV 길이
* `n_dec`: decode 중인 시퀀스 수
* `kv_dec`: decode 시퀀스들의 평균 KV 길이

* **TTFT 예측 (prefill 한 step)**: `(pc=avg_prompt_len, kv_pre=0, n_dec=0, kv_dec=0)`로 lookup.
* **TPOT 예측 (decode 한 step)**: `(pc=0, kv_pre=0, n_dec=batch, kv_dec=max_seq_len/2)`로 lookup.

4D 보간은 까다로우므로 **nearest-neighbor lookup**으로 시작. 정확도가 부족하면 scipy.interpolate.RegularGridInterpolator로 업그레이드.

```python
def lookup_attention(df: pd.DataFrame, pc: int, kv_pre: int, 
                      n_dec: int, kv_dec: int) -> float:
    """nearest neighbor (Manhattan distance 정규화)."""
    import numpy as np
    keys = df[["pc", "kv_pre", "n_dec", "kv_dec"]].values
    # 정규화 (각 축의 max로 나누어 같은 스케일로)
    scale = keys.max(axis=0) + 1e-9
    target = np.array([pc, kv_pre, n_dec, kv_dec]) / scale
    norm_keys = keys / scale
    dists = np.abs(norm_keys - target).sum(axis=1)
    idx = dists.argmin()
    return float(df.iloc[idx]["time_us"])
```

#### 7.3.3 per_sequence.csv

decode step마다 `(n_seq = batch)` 기준 fixed overhead. 단순 lookup.

#### 7.3.4 MoE 모델 (모델의 `architecture == "moe"`인 경우)

* `moe.csv`에서 `(tokens, experts)`로 lookup해서 dense 대신 사용. v1.0에서는 dense 모델만 지원하고 MoE는 NotImplementedError를 던지자. Mixtral 등 MoE는 v1.1.

#### 7.3.5 PP / DP 보정

llm_profile은 보통 `tp<N>` 디렉토리만 있고 PP는 별도 차원이 없다. **단순화**: PP > 1일 때 prefill/decode latency를 `× PP`로 보정하지 않는다 (PP는 pipeline이라 throughput만 늘리고 single-request latency는 약간 증가하므로). Replicas/DP는 latency에 영향 없음. **이 가정을 README에 명시**.

### 7.4 TTFT_pred / TPOT_pred 함수

```python
from .schemas import CandidateConfig, ModelEntry, Stage2Prediction

def predict_ttft_tpot(c: CandidateConfig, m: ModelEntry, 
                       profile_root: Path) -> Stage2Prediction:
    if m.architecture == "moe":
        raise NotImplementedError("MoE 예측은 v1.1에서. 현재 후보는 dense만.")

    # 1. 사용할 HW와 TP 결정
    #    혼합 HW일 때는 가장 작은(보수적) HW를 기준으로 (정확하지 않으나 보수적 lower bound)
    only_hw = next(iter(c.hardware))   # v1.0은 단일 HW 가정. 혼합이면 v1.1.
    TP = c.parallelism["TP"]
    
    pdir = _profile_dir(profile_root, only_hw, m.name)
    tpdir = pdir / f"tp{TP}"
    if not tpdir.exists():
        raise ProfileNotFoundError(
            f"profile dir not found: {tpdir}. "
            f"가용한 TP 값: {sorted(p.name for p in pdir.glob('tp*'))}"
        )

    dense_df = _load_csv(tpdir / "dense.csv")
    attn_df = _load_csv(tpdir / "attention.csv")
    perseq_df = _load_csv(tpdir / "per_sequence.csv")

    batch = max(c.batch_size, 1)
    prompt_tokens = batch * c.avg_prompt_len
    
    # ---- TTFT: prefill 1 step ----
    ttft_dense_us = lookup_dense(dense_df, tokens=prompt_tokens)
    ttft_attn_us = lookup_attention(attn_df, pc=c.avg_prompt_len, 
                                      kv_pre=0, n_dec=0, kv_dec=0)
    ttft_perseq_us = lookup_per_sequence(perseq_df, n_seq=batch)
    ttft_pred_ms = (ttft_dense_us + ttft_attn_us + ttft_perseq_us) / 1000

    # ---- TPOT: decode 1 step (batch개 시퀀스 동시 decode) ----
    decode_tokens = batch  # 한 step에 batch개 시퀀스가 각각 토큰 1개씩 처리
    tpot_dense_us = lookup_dense(dense_df, tokens=decode_tokens)
    tpot_attn_us = lookup_attention(attn_df, pc=0, kv_pre=0,
                                     n_dec=batch, kv_dec=c.max_seq_len // 2)
    tpot_perseq_us = lookup_per_sequence(perseq_df, n_seq=batch)
    tpot_pred_ms = (tpot_dense_us + tpot_attn_us + tpot_perseq_us) / 1000

    return Stage2Prediction(
        candidate_id=c.candidate_id,
        ttft_pred_ms=ttft_pred_ms,
        tpot_pred_ms=tpot_pred_ms,
        breakdown={
            "ttft_dense_us": ttft_dense_us,
            "ttft_attn_us": ttft_attn_us,
            "ttft_perseq_us": ttft_perseq_us,
            "tpot_dense_us": tpot_dense_us,
            "tpot_attn_us": tpot_attn_us,
            "tpot_perseq_us": tpot_perseq_us,
        },
        profile_source=str(tpdir),
    )
```

### 7.5 Sanity check vs Stage 1 lower bound

Stage 1이 계산한 `decode_tpot_lb_ms`는 lower bound이므로 Stage 2의 `tpot_pred_ms` ≥ `decode_tpot_lb_ms`가 항상 성립해야 한다. **그렇지 않으면 코드 버그**.

```python
def sanity_check(p: Stage2Prediction, s1: Stage1Result) -> list[str]:
    warnings = []
    if s1.decode_tpot_lb_ms is not None and p.tpot_pred_ms < s1.decode_tpot_lb_ms * 0.9:
        warnings.append(
            f"TPOT pred {p.tpot_pred_ms:.2f}ms < roofline lb {s1.decode_tpot_lb_ms:.2f}ms "
            f"by >10% — 카탈로그 값(HBM BW)이나 dense.csv 단위를 확인할 것"
        )
    if s1.prefill_ttft_lb_ms is not None and p.ttft_pred_ms < s1.prefill_ttft_lb_ms * 0.9:
        warnings.append(
            f"TTFT pred {p.ttft_pred_ms:.2f}ms < roofline lb {s1.prefill_ttft_lb_ms:.2f}ms"
        )
    return warnings
```

### 7.6 Stage 2 검증 — Spearman 상관성

**검증이 가장 중요한 부분이다.** Stage 2 예측이 full sim의 순위를 잘 매기지 못하면 무용지물.

```python
# 별도 스크립트: scripts/validate_stage2.py
import json
from scipy.stats import spearmanr

def validate_against_full_sim(
    candidates: list[CandidateConfig],
    full_sim_results: dict[str, dict],   # candidate_id → {"ttft_p99_ms": ..., "tpot_p99_ms": ...}
    profile_root: Path,
    models: dict[str, ModelEntry],
) -> dict:
    pred_ttft, pred_tpot = [], []
    true_ttft, true_tpot = [], []
    for c in candidates:
        m = models[c.model_name]
        p = predict_ttft_tpot(c, m, profile_root)
        pred_ttft.append(p.ttft_pred_ms)
        pred_tpot.append(p.tpot_pred_ms)
        true_ttft.append(full_sim_results[c.candidate_id]["ttft_p99_ms"])
        true_tpot.append(full_sim_results[c.candidate_id]["tpot_p99_ms"])
    return {
        "spearman_ttft": float(spearmanr(pred_ttft, true_ttft).correlation),
        "spearman_tpot": float(spearmanr(pred_tpot, true_tpot).correlation),
        "n": len(candidates),
    }
```

**목표 Spearman**: ≥ 0.80 (발표자료에서 제시한 임계값). 30~50개 후보로 시작.

* Spearman이 < 0.80이면: (a) attention.csv lookup 키 차원(pc/kv_pre/n_dec/kv_dec)이 잘못 매칭되었는지, (b) PP 보정을 누락했는지, (c) per_sequence.csv가 실제로는 다른 의미인지 점검.
* Spearman이 < 0.60이면: 예측기 자체가 무용지물이므로 사용자에게 보고하고 Stage 2 사용을 중지.

### 7.7 Stage 2 단위 테스트 (`tests/dse/test_stage2.py`)

* CSV 파일이 없는 경우 `ProfileNotFoundError`가 정확히 발생.
* CSV가 있는 경우 (테스트 fixture로 작은 더미 CSV를 둠) `predict_ttft_tpot`이 양수 결과를 반환.
* `sanity_check`가 roofline lb 위반을 정확히 잡음.
* nearest-neighbor lookup이 키 값과 정확히 매칭되는 경우 (interpolation 없이) 그 값을 그대로 반환.

더미 CSV 예시 (`tests/dse/fixtures/perf/H100/meta-llama--Llama-3.1-70B/default/tp4/dense.csv`):

```csv
tokens,time_us
1,150.0
4,200.0
16,300.0
64,800.0
256,2400.0
1024,9000.0
4096,35000.0
```

(테스트 fixture는 가짜이지만 monotonic increasing이어야 코드 검증에 의미가 있다.)

---

## 8. 통합 파이프라인과 CLI

### 8.1 `pipeline.py`

```python
import time
import logging
from pathlib import Path
from .schemas import (CandidateConfig, Spec, PipelineReport, 
                       Stage1Result, Stage2Prediction)
from .catalog import load_hardware_catalog, load_model_catalog
from .stage1_filters import apply_stage1
from .stage2_predictor import predict_ttft_tpot, sanity_check
from collections import Counter

log = logging.getLogger(__name__)

def run_pipeline(
    candidates: list[CandidateConfig],
    spec: Spec,
    profile_root: Path,
    rank_by: str = "score",   # "ttft" | "tpot" | "score" (weighted)
    weights: dict[str, float] = None,
) -> PipelineReport:
    weights = weights or {"ttft": 0.5, "tpot": 0.5}
    hw = load_hardware_catalog()
    models = load_model_catalog()
    m = models[spec.model.name]

    t0 = time.time()
    # --- Stage 1 ---
    stage1_results: list[Stage1Result] = []
    rejection_counter: Counter[str] = Counter()
    for c in candidates:
        r = apply_stage1(c, m, hw, spec)
        stage1_results.append(r)
        if not r.passed:
            for reason in r.reasons:
                rejection_counter[reason.filter_name] += 1
    t1 = time.time()
    survivors_s1 = [c for c, r in zip(candidates, stage1_results) if r.passed]
    log.info(f"Stage 1: {len(survivors_s1)}/{len(candidates)} passed ({t1-t0:.2f}s)")

    # --- Stage 2 ---
    preds: list[Stage2Prediction] = []
    warnings_all = []
    for c in survivors_s1:
        s1 = next(r for r in stage1_results if r.candidate_id == c.candidate_id)
        try:
            p = predict_ttft_tpot(c, m, profile_root)
            warns = sanity_check(p, s1)
            if warns:
                warnings_all.extend([(c.candidate_id, w) for w in warns])
            preds.append(p)
        except (FileNotFoundError, NotImplementedError) as e:
            log.warning(f"Stage 2 skipped for {c.candidate_id}: {e}")
            # 예측 실패한 후보는 큰 값을 줘서 뒤로 밀어냄
            preds.append(Stage2Prediction(
                candidate_id=c.candidate_id, ttft_pred_ms=float("inf"),
                tpot_pred_ms=float("inf"), profile_source="N/A"))
    t2 = time.time()

    # --- 정렬 ---
    if rank_by == "ttft":
        preds.sort(key=lambda p: p.ttft_pred_ms)
    elif rank_by == "tpot":
        preds.sort(key=lambda p: p.tpot_pred_ms)
    else:  # weighted score (낮을수록 좋음)
        # 정규화는 max로 (모두 양수)
        max_ttft = max((p.ttft_pred_ms for p in preds if p.ttft_pred_ms < float("inf")), default=1.0)
        max_tpot = max((p.tpot_pred_ms for p in preds if p.tpot_pred_ms < float("inf")), default=1.0)
        preds.sort(key=lambda p: 
            weights["ttft"] * (p.ttft_pred_ms / max_ttft) +
            weights["tpot"] * (p.tpot_pred_ms / max_tpot))

    return PipelineReport(
        total_input=len(candidates),
        stage1_passed=len(survivors_s1),
        stage2_passed=len([p for p in preds if p.ttft_pred_ms < float("inf")]),
        stage1_rejections_by_reason=dict(rejection_counter),
        survivors=[
            {"candidate_id": p.candidate_id,
             "ttft_pred_ms": p.ttft_pred_ms,
             "tpot_pred_ms": p.tpot_pred_ms,
             "rank": i + 1}
            for i, p in enumerate(preds)
        ],
        elapsed_s={"stage1": t1 - t0, "stage2": t2 - t1},
    )
```

### 8.2 CLI (`webapp_dse/cli.py`)

```python
import click
import json
import yaml
from pathlib import Path
from .core.schemas import Spec, CandidateConfig
from .core.pipeline import run_pipeline

@click.group()
def cli(): pass

@cli.command()
@click.option("--spec", "spec_path", required=True, type=click.Path(exists=True))
@click.option("--candidates", "cand_path", required=True, type=click.Path(exists=True))
@click.option("--profile-root", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--top-n", type=int, default=None)
@click.option("--rank-by", type=click.Choice(["ttft", "tpot", "score"]), default="score")
def filter(spec_path, cand_path, profile_root, out, top_n, rank_by):
    """Stage 1 + Stage 2를 실행하여 후보를 필터링하고 정렬한다."""
    spec_dict = yaml.safe_load(Path(spec_path).read_text())
    spec = Spec.model_validate(spec_dict)
    candidates = [CandidateConfig.model_validate(json.loads(line))
                  for line in Path(cand_path).read_text().splitlines() if line.strip()]
    report = run_pipeline(candidates, spec, Path(profile_root), rank_by=rank_by)
    if top_n:
        report.survivors = report.survivors[:top_n]
    Path(out).write_text(report.model_dump_json(indent=2))
    click.echo(f"OK: in={report.total_input} → s1={report.stage1_passed} → s2={report.stage2_passed}, "
               f"saved to {out}")

if __name__ == "__main__":
    cli()
```

### 8.3 예시 사용

```bash
python -m webapp_dse.cli filter \
    --spec webapp_dse/examples/spec_llama70b.yaml \
    --candidates webapp_dse/examples/candidates_sample.jsonl \
    --profile-root ../llm_profile \
    --out filtered.jsonl \
    --top-n 20 \
    --rank-by score
```

기대 출력:

```
OK: in=50 → s1=12 → s2=12, saved to filtered.jsonl
```

---

## 9. 테스트 전략

### 9.1 단위 테스트 (각 함수 단위)

* `tests/dse/test_schemas.py`: Pydantic 모델 round-trip + 잘못된 입력에 ValidationError
* `tests/dse/test_catalog.py`: catalog yaml이 정상 로드, 잘못된 yaml에서 에러
* `tests/dse/test_stage1.py`: 6개 필터 각각 통과·실패 2개씩 (총 12개 이상)
* `tests/dse/test_stage2.py`: dummy CSV로 lookup 검증, sanity_check 검증

### 9.2 통합 테스트

* `tests/dse/test_pipeline.py`:
  * 50개 후보 + 알려진 spec → 통과율이 사전 계산한 값과 일치
  * Stage 1만 통과한 후보들의 lower bound가 실제로 Stage 2 예측보다 작음 (≤ 110% 허용)
  * profile_root가 없을 때 graceful degradation (Stage 1만 돌고 Stage 2는 skip, error 아님)

### 9.3 검증 스크립트 (테스트가 아닌 1회성 도구)

* `scripts/validate_stage2.py`: LLMServingSim full sim 30~50회를 돌려 Spearman 측정. CI에서 돌리지 않고, **사람이 1회 돌려서 README에 결과 기록**.

### 9.4 CI

* `pytest tests/dse/ -v` 가 통과해야 PR merge 가능.
* 카탈로그 yaml schema 검증을 CI에 추가.

---

## 10. Claude Code 권장 작업 순서

다음 순서로 진행. **각 단계가 끝날 때마다 테스트가 green이어야 다음으로 넘어간다.**

1. **준비**: `webapp_dse/` 디렉토리, `__init__.py`, `config.py`, `data/{hardware,models}.yaml` 생성. 부록 A·B의 값으로 yaml 채움.
2. **schemas.py**: 4장의 모든 Pydantic 모델 작성. `tests/dse/test_schemas.py`로 round-trip 검증.
3. **catalog.py**: 5장 그대로 작성. `tests/dse/test_catalog.py`로 검증.
4. **stage1_filters.py — 함수 6개**: 6.2의 시그너처대로 하나씩 구현. 각 함수 작성 직후 해당 단위 테스트(`test_stage1.py`)를 추가하고 즉시 실행.
5. **stage1_filters.py — `apply_stage1` 통합 함수**: 6.3. test_stage1.py에 collect-all과 early-exit 모드 둘 다 테스트 추가.
6. **examples/candidates_sample.jsonl**: 손으로 50개 후보 작성. 각각 예상 통과/실패와 사유를 주석 또는 별도 yaml에 기록. (이 50개가 회귀 fixture가 된다.)
7. **stage2_predictor.py — CSV 로더와 lookup 함수**: 7.2~7.3. dummy CSV로 테스트.
8. **stage2_predictor.py — `predict_ttft_tpot`**: 7.4. dummy profile 디렉토리로 통합 테스트.
9. **stage2_predictor.py — `sanity_check`**: 7.5.
10. **pipeline.py**: 8.1. `tests/dse/test_pipeline.py`에서 50개 후보로 end-to-end 검증.
11. **cli.py**: 8.2. `--help` 출력만 확인하고 실제 호출 1회 (`examples/`로).
12. **scripts/validate_stage2.py**: 7.6. **이건 사용자가 LLMServingSim 환경이 준비된 상태에서 직접 돌릴 것**. Claude Code는 스크립트만 작성.
13. **docs/dse/04_stage1_stage2_design.md**: 본 PLAN의 핵심을 요약한 설계 문서. 코드와 함께 PR.
14. **README 업데이트**: `webapp_dse/README.md`에 사용법, 알려진 제약, validate_stage2.py 실행 가이드.

### 10.1 의사결정이 필요한 지점

다음은 Claude Code가 혼자 결정하지 말고 사용자에게 물어야 하는 것들:

* `llm_profile/` 디렉토리의 실제 경로와 디렉토리 명명 규칙 (`meta-llama/Llama-3.1-70B`가 디렉토리에서 어떻게 표현되는지)
* dense.csv의 row가 "모델 전체 dense 합산"인지 "layer 1개"인지
* attention.csv의 컬럼명이 정확히 `pc, kv_pre, n_dec, kv_dec, time_us`인지
* PP > 1에 대한 latency 보정을 어떻게 할지 (현재는 무보정)
* Mixtral 같은 MoE를 v1.0에 포함할지 (현재는 NotImplementedError)

답을 모르겠으면 **합리적 가정을 하고 README에 명시한 뒤 진행**. 다만 위 다섯 가지 중 하나라도 가정이 틀리면 Stage 2 정확도가 무너지므로, 실제 데이터로 검증 단계(7.6)에서 반드시 점검할 것.

---

## 11. 막힐 만한 지점 가이드 (FAQ)

### Q1. `llm_profile/` 디렉토리에 내가 필요한 (HW, model, TP) 조합이 없다.

A. Stage 2 예측이 불가능. 두 가지 선택지:
1. 해당 후보를 Stage 2에서 skip하고 `ttft_pred=inf, tpot_pred=inf`로 후순위로 보낸다 (현재 구현).
2. 사용자가 LLMServingSim의 프로파일러를 돌려 누락된 (HW, model, TP)에 대한 프로파일을 생성해야 한다는 메시지를 명시적으로 띄운다.

운영상 (2)가 옳다. CLI 종료 메시지에 누락된 페어를 모아서 출력.

### Q2. 메모리 필터에서 혼합 HW(예: H100 2장 + A6000 4장) 후보는 어떻게 처리?

A. v1.0은 보수적으로 **가장 작은 HBM**을 기준으로 평가. 실제로는 모델 weights가 shard되는 방식에 따라 다르지만 그건 v1.1의 이질 sharding 모델에서 정확히 다룬다. README에 한계 명시.

### Q3. Roofline lower bound가 너무 보수적이라 모든 후보가 통과한다.

A. 정상이다. Roofline은 **필요조건이지 충분조건이 아니다**. 통과한 후보들은 Stage 2 또는 Stage 5(full sim)에서 더 정밀하게 평가된다. Stage 1의 목적은 명백히 불가능한 후보를 자르는 것.

### Q4. Stage 2의 Spearman이 0.6 정도로 낮다.

A. 디버깅 체크리스트:
1. attention.csv lookup 키 차원(pc/kv_pre/n_dec/kv_dec)이 실제 CSV 컬럼과 맞는지
2. `time_us` 컬럼이 정말 마이크로초인지 (혹시 나노초·밀리초?)
3. `_load_csv` 캐싱이 잘못된 파일을 반환하지 않는지 (path 비교 정상?)
4. 모델·HW 디렉토리 매핑이 1:1인지 (`/` → `--` 변환이 맞는지)
5. PP > 1 후보에 대해 보정이 누락되었는지
6. batch_size=0 (무제한)을 1로 대체한 것이 실제 시뮬 결과와 동떨어지는지

### Q5. 카탈로그 값(peak FLOPS, HBM BW)이 데이터시트와 다른데?

A. **데이터시트의 이론치보다 실측치를 우선**하라. 가능하면 LLMServingSim 자체가 가정하는 값(`inference_serving/memory_model.py` 등)에서 가져오는 것이 가장 안전. Phase 0의 `docs/dse/03_catalog.yaml`이 그 답을 가지고 있어야 한다. 없으면 일단 부록 B 값으로 진행하고, validate_stage2.py가 큰 오차를 보이면 의심 1순위.

### Q6. Llama-3-70B + 128k context에서 KV가 40GB 나온다는데 내 계산은 다르다.

A. 공식:
```
KV/token = 2 × L × H_kv × d_head × bytes
         = 2 × 80 × 8 × 128 × 2 (BF16)
         = 327,680 bytes/token ≈ 0.31 MB/token
```
128k 토큰 × 0.31 MB ≈ 40 GB. 이게 안 맞으면 단위(KB vs KiB, MB vs MiB)나 H_kv vs H_q(64 vs 8) 혼동 가능성 1순위.

### Q7. early_exit과 collect-all 모드 중 기본은?

A. **collect-all이 기본**. 디버깅 시 모든 reject 사유를 보는 게 압도적으로 유용. 만 개 이상 후보를 처리하느라 성능이 문제될 때만 early_exit으로 전환.

---

## 12. 부록 A: 알려진 모델 메타데이터

| 모델 | num_params | L (layers) | hidden | H_q | H_kv | d_head | arch |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 8.03B | 32 | 4096 | 32 | 8 | 128 | dense |
| Llama-3.1-70B | 70.55B | 80 | 8192 | 64 | 8 | 128 | dense |
| Mixtral-8x7B-v0.1 | 46.7B | 32 | 4096 | 32 | 8 | 128 | MoE (8 experts, top-2) |
| Phi-mini-MoE | ~7.6B | 32 | 3072 | 32 | 8 | 96 | MoE (16 experts, top-2) — 정확값 확인 필요 |

**출처**: huggingface model cards + LLMServingSim의 `model_config/`. **반드시 실제 model_config의 값으로 교차 검증**.

## 13. 부록 B: 알려진 하드웨어 메타데이터

| HW | HBM | HBM BW | BF16 TFLOPS | TDP | Interconnect |
|---|---|---|---|---|---|
| H100 SXM5 | 80 GiB | 3.35 TB/s | 989 (dense) | 700 W | NVLink 900 GB/s |
| H100 PCIe | 80 GiB | 2.04 TB/s | 756 | 350 W | PCIe 5.0 ~128 GB/s |
| A100 80GB SXM | 80 GiB | 2.04 TB/s | 312 | 400 W | NVLink 600 GB/s |
| A6000 | 48 GiB | 768 GB/s | ~155 | 300 W | PCIe 4.0 (no NVLink) |
| TPU-v6e-1 | 32 GiB | 1.64 TB/s | 918 | 350 W | ICI 460 GB/s |

**주의**:
* TFLOPS는 BF16 dense Tensor Core 기준 (sparsity 가속 제외).
* H100은 SXM5와 PCIe가 사양이 크게 다르다. 어느 쪽인지 명확히.
* 카탈로그 값은 LLMServingSim의 가정과 일치시키는 것이 최우선.

## 14. 부록 C: 참고 자료

* 발표자료 PPTX (이 PR의 컨텍스트): `LLM_DSE_발표자료.pptx` 슬라이드 13~16 (Stage 1), 17~19 (Stage 2)
* 한국어 보고서: `research_report_ko.md` §2.1, §2.6
* LLMServingSim repo: `https://github.com/ai-computing/LLMServingSim`
* LLMServingSim 공식 문서: `https://llmservingsim.ai/docs/`
* Roofline 논문: Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363)
* KV cache 공식 출처: Hyperstack engineering blog, Lyceum AI memory math

---

## 끝.

이 PLAN을 그대로 Claude Code에 전달하면 처음부터 끝까지 1~3일 안에 구현 가능하다. 만약 막히는 부분이 생기면 §11 FAQ를 먼저 확인하고, 그래도 안 풀리면 §10.1의 의사결정 지점을 사용자(인간)에게 묻는다.
