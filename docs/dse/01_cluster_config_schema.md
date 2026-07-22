# 01 — `cluster_config/*.json` 스키마 역공학

> `inference_serving/config_builder.py:build_cluster_config()`가 정의하는 모든 검증 규칙을 항목별로 정리. DSE 도구의 generator/builder는 **이 스키마를 어기지 않는** JSON을 생성해야 합니다.

## 1. Top-level 구조

```jsonc
{
  "num_nodes": 1,           // (required) int. nodes.length와 일치해야 함
  "link_bw": 112,           // (required) GB/s
  "link_latency": 0,        // (required) ns
  "nodes": [...],           // (required) length == num_nodes
  "cxl_mem": {...}          // (optional) CXL 메모리 풀
}
```

### 검증 규칙
- `len(nodes) == num_nodes` — 어기면 `ValueError`
- `link_bw`, `link_latency` 둘 다 명시 — None이면 `KeyError`

## 2. Node 객체

```jsonc
{
  "num_instances": 2,       // (required) instances.length와 일치
  "cpu_mem": {...},         // (required) Memory config — 아래 §3
  "instances": [...],       // (required)
  "power": {...}            // (conditional) 모든 노드에 있거나 모두 없거나
}
```

### 검증 규칙
- `len(instances) == num_instances` — 어기면 `ValueError`
- `power` 블록: **all-or-nothing**. 어느 노드 하나라도 빠지면 power modeling 전체 비활성화 (`config_builder.py:74-75`)

## 3. Memory config (cpu_mem / cxl_mem / npu_mem)

공통 `mem_required_keys`:

| 키 | 단위 | 의미 |
|---|---|---|
| `mem_size` | GB | 메모리 용량 |
| `mem_bw` | GB/s | 메모리 대역폭 |
| `mem_latency` | ns | 메모리 접근 latency |

추가 키:
- `cxl_mem`: `num_devices` (optional, default 1)
- `cpu_mem` w/ attn offloading: `pim_config` (str, name of `pim_config/<name>.ini`)

## 4. Instance 객체

```jsonc
{
  "model_name": "meta-llama/Llama-3.1-8B",  // (required) HF id
  "hardware": "A6000",                        // (required) catalog 내 키
  "npu_mem": {                                // (required) — §3 mem 키
    "mem_size": 40,
    "mem_bw": 768,
    "mem_latency": 0
  },
  "npu_num": 1,                               // (required) NPU 개수
  "npu_group": 1,                             // (required) PP stage 수
  "pd_type": null,                            // (required) null | "prefill" | "decode"
  "placement": {...},                         // (optional) 레이어 배치 — §6
  "pim_config": "..."                         // (optional) PIM 사용 시
}
```

### 검증 규칙
- 위 6개 키 모두 명시 — 어기면 `KeyError`
- `npu_group <= npu_num`, `npu_num % npu_group == 0` (`config_builder.py`)
- `pd_type` ∈ {`null`, `"prefill"`, `"decode"`}
- `hardware`가 catalog에 존재해야 함 (`llm_profile/perf_models/<hw>/<vendor>/<model>/tp{npu_num // npu_group}/` 디렉토리 + layers.csv + attention predictions)

### 파생 의미
- `tp = npu_num // npu_group` (tensor parallelism per stage)
- `pp = npu_group` (pipeline stages)
- prefill instance: ASTRA-Sim 내부에서 `npu_num *= 2` (sender NPU 추가). cluster JSON에는 사용자가 의도한 compute NPU 수만 적습니다.

## 5. Power block (선택. 있으면 모든 노드에 있어야)

```jsonc
"power": {
  "base_node_power": 60,         // (W)
  "npu": {
    "A6000": {                    // 노드 내 모든 instance.hardware에 대해 entry 필요
      "idle_power": 25,
      "standby_power": 115,
      "active_power": 300,
      "standby_duration": 18      // (s)
    }
  },
  "cpu":     { "idle_power": 10, "active_power": 200, "util": 0.15 },
  "dram":    { "dimm_size": 32, "idle_power": 2.0, "energy_per_bit": 6.0 },
  "link":    { "num_links": 1, "idle_power": 5, "energy_per_bit": 4.0 },
  "nic":     { "num_nics": 1, "idle_power": 20 },
  "storage": { "num_devices": 2, "idle_power": 5 }
}
```

### 검증 규칙 (`config_builder.py:84-125`)
- `base_node_power, npu, cpu, dram, link, nic, storage` 7개 키 모두 필요
- `npu` 안: 노드 내 모든 instance의 `hardware` 마다 `idle_power, standby_power, active_power, standby_duration` 4개 필요
- `cpu`: `idle_power, active_power, util`
- `dram`: `dimm_size, idle_power, energy_per_bit` (단, `enable_attn_offloading`이면 `energy_per_bit`만 필요 — PIM이 나머지를 덮어씀)
- `link`: `num_links, idle_power, energy_per_bit`
- `nic`: `num_nics, idle_power`
- `storage`: `num_devices, idle_power`

> `inference_serving/config_builder.py:163-167`에서 instance loop 안에 `power["npu"][hw]["num_npus"] += inst["npu_num"]`로 누적함. 사용자가 직접 채울 필요 없음 (자동 산정).

## 6. Placement (선택 — 메모리 계층 배치 제어)

```jsonc
"placement": {
  "default": {
    "weights":      "npu" | "cpu" | "cxl" | "cxl:0",  // lowercase. cxl는 :N 접미사로 device 선택 가능
    "kv_loc":       "npu" | "cpu" | "cxl",
    "kv_evict_loc": "npu" | "cpu" | "cxl"
  },
  "block": [   // (선택) 길이 = num_hidden_layers. block 단위 override
    {"weights": "cpu", "kv_loc": "npu"}, ...
  ],
  "layer": {   // (선택) 레이어 이름별 override
    "q_proj": {"weights": "cxl:0", "kv_loc": "npu", "kv_evict_loc": "cpu"}
  }
}
```

DSE 도구는 일단 **`placement` 미사용**이 기본. 필요 시 사용자 입력으로 enable.

## 7. CXL memory (선택)

```jsonc
"cxl_mem": {
  "mem_size": 256,
  "mem_bw": 64,
  "mem_latency": 200,
  "num_devices": 1
}
```

## 8. PIM (선택 — `--enable-attn-offloading`과 함께 사용)

cpu_mem에 `pim_config` 필드 추가하면 자동 enable:
```jsonc
"cpu_mem": {
  "mem_size": 128,
  "pim_config": "AiM"        // → pim_config/AiM.ini 로드
}
```

`pim_config` 사용 시 cpu_mem의 `mem_bw, mem_latency`는 PIM ini 값으로 overwrite. 또한 dram power의 `dimm_size, idle_power`도 overwrite.

## 9. 예시 config 매트릭스

`cluster_config/*.json` 13개 파일을 핵심 차원으로 분류:

| 파일 | nodes | inst | pd_type | power | placement | cxl | pim |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `single_node_single_instance.json` | 1 | 1 | — | — | — | — | — |
| `single_node_single_instance_H100.json` | 1 | 1 | — | — | — | — | — |
| `single_node_multi_instance.json` | 1 | 2+ | — | — | — | — | — |
| `single_node_pd_instance.json` | 1 | 2+ | P/D | — | — | — | — |
| `dual_node_multi_instance.json` | 2 | 4 | — | — | — | — | — |
| `single_node_memory_instance.json` | 1 | 1 | — | — | ✓ | — | — |
| `single_node_cxl_instance.json` | 1 | 1 | — | — | ✓ | ✓ | — |
| `single_node_pim_instance.json` | 1 | 1 | — | — | — | — | ✓ |
| `single_node_power_instance.json` | 1 | 1 | — | ✓ | — | — | — |
| `single_node_rngd_power_instance.json` | 1 | 1 | — | ✓ | — | — | — |
| `single_node_moe_single_instance.json` | 1 | 1 | — | — | — | — | — |
| `single_node_moe_multi_instance.json` | 1 | 2+ | — | — | — | — | — |
| `single_node_moe_pd_instance.json` | 1 | 2+ | P/D | — | — | — | — |

**DSE generator의 시작점**: `single_node_single_instance.json` (가장 단순). `single_node_power_instance.json`은 power template 참조용.

## 10. 의존성 / 일관성 규칙 요약

| 조건 | 효과 |
|---|---|
| `power` 블록이 모든 node에 ⇔ power modeling 활성화 | 결과에 에너지 메트릭 포함 |
| `enable_attn_offloading=True` (CLI 플래그) | cpu_mem에 `pim_config` 필수, dram power 일부 override |
| `cxl_mem` top-level에 있음 | memory_config에 cxl_mem 항목 추가 |
| `pd_type` 중에 prefill 있음 | ASTRA-Sim 토폴로지에서 그 인스턴스 NPU 수 2배 |
| `npu_group > 1` | pipeline parallelism active. `_create_network_config`이 2D FullyConnected 토폴로지 emit |
| 노드 내 mixed hardware + 모든 instance.combined | ASTRA-Sim deadlock (현재 enumerate에서 차단) |

## 11. JSON Schema (Draft 7)

별도 파일: `docs/dse/cluster_config_schema.json` (자동 검증용)

```bash
# 모든 예시 config가 schema 통과하는지 한 번에 검증
python3 -c "
import json, jsonschema
schema = json.load(open('docs/dse/cluster_config_schema.json'))
import glob
for path in glob.glob('cluster_config/*.json'):
    try:
        jsonschema.Draft7Validator(schema).validate(json.load(open(path)))
        print('OK ', path)
    except jsonschema.ValidationError as e:
        print('FAIL', path, '—', e.message)
"
```

## 12. DSE generator에서의 시사점

1. **사전 필터**: hardware ∈ catalog, npu_group | npu_num
2. **자동 채울 필드**: power block의 `num_npus`는 `config_builder.py`가 알아서 계산 (생성기는 카탈로그 기반 `idle/standby/active/standby_duration`만 채움)
3. **금지 패턴**:
   - 단일 노드에 mixed hardware + 모두 `pd_type=null` (heterogeneous combined — enumerate가 이미 차단)
   - prefill PP > 1 in multi-group (silent NPU drop)
   - decode `npu_num > 1` (known ASTRA-Sim crash, enumerate가 차단)
4. **최소 spec**: nodes + cpu_mem(default) + instances(min 1) + link_bw + link_latency. power/placement/cxl/pim은 모두 optional
