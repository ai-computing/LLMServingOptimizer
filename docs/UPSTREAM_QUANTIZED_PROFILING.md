# 업스트림 프로파일러로 양자화 정밀도 프로파일 만들기 (A5000)

`backends/upstream`의 레이어 프로파일러로 fp8 / int8 / int4 프로파일을 뽑는
방법과, 왜 문서에 적힌 `--dtype fp8`이 동작하지 않는지 정리한다.

## 1. `--dtype fp8`은 쓸 수 없다

프로파일러 도움말은 `--dtype (bfloat16/float16/float32/fp8)`을 광고하지만,
이 값은 `vllm.LLM(dtype=...)`으로 그대로 전달된다(`profiler/core/engine.py`).
현재 venv의 vLLM은 dtype 리터럴에서 fp8을 받지 않는다:

```
Input should be 'auto', 'half', 'float16', 'bfloat16', 'float' or 'float32'
  [type=literal_error, input_value='fp8']
```

vLLM에서 fp8 가중치는 dtype이 아니라 **양자화 경로**(`--quantization`, 또는
체크포인트의 `quantization_config`)로 들어온다. 프로파일러에는
`--quantization` 인자가 없다. 즉 도움말이 구버전 vLLM 기준으로 낡았고,
서브모듈을 수정하지 않는 한 이 플래그로는 방법이 없다.

## 2. 우회로: 양자화 체크포인트의 config.json을 주입한다

프로파일러는 **가중치를 로드하지 않는다**. `configs/model/<org>/<name>.json`
(순수 HF config.json)을 임시 디렉터리에 쓰고 `load_format="dummy"`로 vLLM을
띄운다(`profiler/core/config.py`). 따라서 프로파일 대상의 정밀도는 오직
config의 `quantization_config` 블록이 결정한다 — 실제 양자화된 가중치 파일은
필요 없다.

`--model-config-root`로 config 탐색 경로를 바꿀 수 있으므로, 양자화
체크포인트의 config.json을 **우리 repo 안에** 두면 서브모듈은 깨끗한 상태로
유지된다(`CLAUDE.md`의 서브모듈 수정 금지 규칙).

```
configs/upstream_model/<org>/<checkpoint-name>.json    # HF config.json 그대로
```

fp8 프로파일을 뽑은 실제 명령:

```bash
cd backends/upstream
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python -m profiler profile \
  RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
  --model-config-root /path/to/LLMServingOptimizer/configs/upstream_model \
  --hardware A5000 --tp 1,2,4,8 --skip-skew \
  --out-root /path/to/LLMServingOptimizer/profiles/upstream
```

산출물은
`profiles/upstream/A5000/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8/bf16/tp<N>/`.

### `--variant`를 주지 말 것

variant 폴더 이름은 **양자화 스킴이 아니라 config의 `torch_dtype`(+ KV dtype)**
을 따른다. compressed-tensors FP8 체크포인트의 `torch_dtype`은 `bfloat16`이므로
variant는 `bf16`이 되는 게 정상이다.

시뮬레이터가 런타임에 같은 계산으로 폴더를 찾기 때문에
(`serving/core/trace_generator.py::resolve_variant` =
`_short_dtype(dtype or torch_dtype)`) 여기서 `--variant fp8` 같은 이름을 주면
프로파일은 만들어지지만 **시뮬레이션이 폴더를 못 찾는다**:

```
FileNotFoundError: Profile variant folder not found:
  ../profiler/perf/A5000/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8/bf16
```

정밀도는 이미 모델 ID가 표현하므로 variant에 중복으로 넣을 이유도 없다.
`--variant`는 같은 체크포인트를 서로 다른 KV dtype으로 여러 벌 뽑을 때만 쓴다.

### 모델 ID = 정밀도

정밀도는 별도 차원이 아니라 **체크포인트 ID 자체**로 표현한다. 실측 오라클
(`profiles/measured/`)이 이미 그 규칙이고, 사용자가 vLLM에 넘기는 ID와도
같으며, 어댑터의 `list_hardware()`가 variant 폴더를 머지해버리기 때문에
정밀도를 variant에 넣으면 카탈로그에서 구분되지 않는다.

## 3. TP 에뮬레이션과 양자화 스킴의 궁합

프로파일러는 TP=N을 GPU 1장에서 흉내낸다 — `tensor_parallel_size=1`을 유지한
채 `hf_overrides`로 hidden/intermediate/heads를 N으로 나눠 랭크 1개의 커널
shape을 재현한다. 여기서 양자화 스킴의 granularity가 제약이 된다:

| 스킴 | granularity | TP 에뮬레이션 |
|---|---|---|
| fp8 (compressed-tensors) | per-tensor | 항상 안전 |
| int8 w8a8 | weight per-channel, act per-token | 항상 안전 |
| int4 AWQ | **group_size=128** | 축소된 차원이 128의 배수여야 함 |

Llama-3.1-8B(hidden 4096, intermediate 14336)는 tp16까지 안전하다
(4096/8=512, 14336/8=1792, 4096/16=256, 14336/16=896 — 모두 128의 배수).

## 4. 이 프로파일이 유효한지 검증

dummy 가중치라도 **커널 타이밍은 실제**다(shape과 양자화 커널이 동일).
bf16 tp1과 fp8 tp1의 `dense.csv`를 비교하면 fp8 경로가 실제로 동작했음이
바로 드러난다:

| layer | 1 tok bf16→fp8 | 비율 | 2048 tok bf16→fp8 | 비율 |
|---|---|---|---|---|
| qkv_proj | 81.4 → 44.4 µs | **0.55** | 1029 → 1141 µs | 1.11 |
| o_proj | 56.8 → 30.3 µs | **0.53** | 689 → 757 µs | 1.10 |
| gate_up_proj | 356.8 → 177.3 µs | **0.50** | 4795 → 5896 µs | 1.23 |
| down_proj | 169.1 → 89.3 µs | **0.53** | 2437 → 2552 µs | 1.05 |
| layernorm | 3.02 → 3.03 µs | 1.01 | 79.8 → 79.2 µs | 0.99 |

읽는 법:

- **메모리 바운드(1토큰)**: GEMM이 일제히 0.50~0.55× — 가중치 바이트가 절반이
  되어 나오는 2배 가속. 이게 fp8 경로가 켜졌다는 직접 증거다.
- **컴퓨트 바운드(2048토큰)**: 1.05~1.23×로 **오히려 느리다**. A5000(SM 8.6)은
  FP8 텐서코어가 없어 dequant 오버헤드만 얹힌다.
- **비양자화 연산**(layernorm/rotary/embedding)은 1.0× 부근 — 대조군이 맞게
  나왔다는 뜻. GEMM만 변했다.

이 부호는 실측 정밀도 실험(`docs/PRECISION_STUDY_A5000.md`)의 "저동시성에서
이득, 고동시성에서 손실" 패턴과 같은 방향이다.

**주의**: dummy 가중치이므로 이 프로파일에서 **품질(정확도)은 얻을 수 없다**.
정확도는 실측 캠페인(`scripts/run_capacity_campaign.py --quality-samples`,
GSM8K)에서만 나온다.

### 엔드투엔드: 시뮬이 실제로 도는지

커널 CSV가 그럴듯한 것과 시뮬이 도는 것은 별개다. 합성 워크로드
(chat, 2 req/s, 10요청)로 네 조합을 돌린 결과:

| 구성 | 평균 지연 | 판정 |
|---|---|---|
| bf16 tp1 | 4,416 ms | 기준 |
| fp8 tp1 | 2,677 ms | **1.65× 빠름** |
| bf16 tp8 | 811 ms | ok |
| fp8 tp4 | 904 ms | ok |

저동시성에서 fp8이 1.65배 빠른 것은 커널 측정의 0.50~0.55× GEMM 시간과
같은 방향이고, 실측 서빙 실험의 저동시성 우세와도 일치한다. 프로파일을 추가한
뒤에는 **반드시 이런 스모크 시뮬까지 돌려볼 것** — 카탈로그에 뜨는 것만으로는
평가 가능함이 보장되지 않는다(§5, §6에서 실제로 두 번 막혔다).

## 5. 프로파일만으로는 부족하다 — 시뮬 시점 config도 필요

프로파일을 다 뽑아도 **시뮬레이션은 별도로 실패한다**. 서빙 쪽
`serving/core/utils.py::get_config()`는 서브모듈 안의 두 경로만 찾고,
프로파일러의 `--model-config-root` 같은 우회로가 없다:

```
FileNotFoundError: Config file for model
  'RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8' not found. Checked:
  backends/upstream/configs/model/RedHatAI/...json,
  backends/upstream/serving/configs/model/RedHatAI/...json
```

이 상태면 카탈로그에는 뜨는데 평가는 실패하는 **깨진 추천 경로**가 된다.
`scripts/setup.sh`가 프로파일을 심링크하는 것과 같은 방식으로 model config도
심링크한다(파일 단위로 걸어서 서브모듈이 이미 가진 vendor 디렉터리는
건드리지 않는다):

```
backends/upstream/configs/model/<org>/<name>.json
  -> ../../../../../configs/upstream_model/<org>/<name>.json
```

새 양자화 체크포인트를 추가할 때는 **프로파일 + 이 심링크 두 개가 세트**다.
`./scripts/setup.sh`를 다시 돌리면 자동으로 걸린다.

## 6. 알려진 격차: fp8 프로파일에는 skew 보정이 없다

fp8은 `--skip-skew`로 뽑았다(skew 스윕은 TP당 몇 시간이 더 걸린다). 반면
bf16 tp1/2/4에는 fitted skew가 있다. 시뮬레이터는 `meta.yaml::skew_fit.per_tp`
가 없는 TP에 대해 pooled 상수 alpha로 되돌아가므로, **bf16과 fp8을
시뮬레이션으로 비교할 때 attention skew 모델의 충실도가 다르다**. 이질적인
kv 길이가 섞인 decode 배치에서 fp8 쪽 오차가 더 클 수 있다.

또한 bf16 변종의 tp8도 같은 이유로 skew fit이 없다(tp1/2/4만 있음).

> 주의: `--skip-skew`로 기존 변종을 다시 프로파일하면 그 변종의 `meta.yaml`이
> **그 세션 기준으로 덮어써진다** — 이전 TP들의 `skew_fit` 항목이 사라져
> (CSV는 남아도) 보정이 조용히 꺼진다. tp8을 추가할 때 실제로 겪었고
> 되돌렸다. 기존 변종에 TP를 추가한 뒤에는 `meta.yaml`의 `tp_degrees`와
> `skew_fit.per_tp`를 반드시 확인할 것.

## 7. 카탈로그 노출

`/api/models`는 (model, hw)의 TP를 fidelity ladder의 모든 단계에서 **합집합**
으로 모으고 각 TP가 어느 단계에서 왔는지 함께 준다:

```
A5000  meta-llama/Llama-3.1-8B   tps=[1,2,4,8]
       sources={1: measured, 2: measured, 4: upstream, 8: upstream}
```

계획(추천) 경로는 여전히 fidelity 우선이라 최상위 단계가 가진 TP만
탐색한다. 시뮬레이터의 넓은 TP 공간까지 탐색하려면 요청에
`force_backend: "upstream"`을 준다(confidence는 medium으로 보고된다).
