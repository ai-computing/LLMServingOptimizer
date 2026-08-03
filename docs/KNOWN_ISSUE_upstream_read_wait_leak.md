# 알려진 문제: upstream 시뮬레이터의 `read_wait()` EOF 무한 루프 (메모리 폭주)

날짜: 2026-07-31 (수정 2026-08-03) · 영향: Stage-2 후보 평가 중 호스트 OOM 위험
상태: **가드로 우회 + 근본 수정 적용**

> **근본 수정 적용됨** — `backends/upstream` 서브모듈의 로컬 브랜치
> `fix/read-wait-eof-guard` (커밋 `95d13be`, upstream `c84e58b` 위)에 아래
> "근본 수정 방향"을 구현했습니다. 검증 결과는 이 문서 끝의 「수정 검증」 참고.
> 부모 저장소의 서브모듈 포인터는 **의도적으로 올리지 않았습니다**(그 커밋이 아직
> 원격에 없어 다른 체크아웃에서 깨짐). 공유하려면 포크에 push 후 포인터를 올리세요.

## 증상

Stage-2에서 후보 하나의 `python -m serving` 프로세스가 **분당 약 13GB**씩 RSS를 늘리며
멈추지 않습니다. 형제 후보들은 0.1–3.6GB로 정상입니다. 시뮬레이션 시각은 첫 로그
구간(`[1.0s]`)에서 **전혀 진행하지 않습니다**. 93GB 머신에서 4개 후보를 병렬 평가하던
중 가용 메모리가 10GB까지 떨어져 OOM 킬러 직전까지 갔습니다(실제 관측).

## 원인 (측정으로 확정)

`backends/upstream/serving/core/controller.py` (v1.1.0+ 기준 14–20행):

```python
def read_wait(self, p):
    out = [""]
    while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
        line = p.stdout.readline()
        out.append(line)          # ← EOF('')도 그대로 append
        p.stdout.flush()          # ← 읽기 스트림에 flush (무의미)
    return out
```

`readline()`은 스트림이 EOF면 **블록하지 않고 즉시 `''`를 반환**합니다. `''`는 두 종료
조건(`"Waiting" in ...`, `"Checking Non-Exited Systems ...\n"`) 어느 것도 만족하지
못하므로 루프가 끝나지 않고, 빈 문자열을 리스트에 무한히 append 합니다.

부수 문제 두 가지가 같은 7행 함수에 있습니다: (1) 마지막 줄만 필요한데 읽은 모든 줄을
`out`에 보관, (2) 읽기 전용 스트림에 `flush()` 호출.

### 계측 결과 (동일 설정 4/4 재현)

| 경과 | RSS | `out` 원소 수 |
|---|---|---|
| 15s | 3.05 GB | 31,603,386 |
| 31s | 6.56 GB | 66,587,533 |
| 48s | 10.41 GB | 107,370,426 |
| 63s | 13.99 GB | 143,204,901 |
| 79s | 17.42 GB | 178,092,001 |

초당 약 225만 회 append(빈 문자열), RSS 증가율 약 13GB/분 — 원래 사고에서 관측한
증가율과 일치합니다. 원소는 전부 `''`(빈 문자열)입니다.

### 트리거 (부분 확인)

`read_wait`가 읽는 파이프가 EOF에 도달하는 상황에서 발생합니다. 최초 사고 당시
해당 python 프로세스 밑에 **살아있는 AnalyticalAstra 하나 + 좀비(`<defunct>`) 하나**가
동시에 존재했고, 재현 실험에서도 자식이 `R`(실행) 상태인데 python은 EOF를 받고
있었습니다. 즉 **이미 종료된 astra-sim 핸들을 대상으로 `read_wait`가 호출되는 경로**가
있다는 뜻입니다(자식 종료 원인 자체는 미확정 — 정상 종료인지 크래시인지 추가 조사 필요).
자식이 왜/언제 종료하든 python 측에 EOF 처리가 없다는 점이 문제의 본질입니다.

## 재현 방법

```bash
# 70B, A40 tp4 x2, 1노드, 50요청 (합성 워크로드)
cd backends/upstream
.venv/bin/python -m serving \
  --cluster-config ../../output/memprobe/A.json \
  --dtype bfloat16 --block-size 16 \
  --dataset ../../<50-request jsonl> \
  --num-reqs 50 --max-num-batched-tokens 2048 \
  --request-routing-policy RR --log-interval 1.0 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --output ../../output/memprobe/A.csv
# 30~90초 내 RSS가 수 GB/분으로 증가하고 시뮬 시각은 [1.0s]에 정체
```

진단에 쓴 프로브(서브모듈 무수정, `PYTHONPATH`로 실행하며 메인 스레드 스택과 거대
지역변수를 덤프)는 이 문제를 다시 볼 때 재사용할 수 있습니다.

## 우회 (구현 완료)

`planner/sim_evaluator.run_guarded()` — Stage-2의 각 시뮬레이션을 독립 세션에서 돌리고
**프로세스 트리 RSS 상한**(기본 16GiB, `LLMSS_SIM_MEM_LIMIT_GB`)을 넘으면 그룹째 종료해
해당 후보만 `Infeasible`로 떨어뜨립니다. 서비스는 20B 이상 모델에서 Stage-2 병렬도를
4→2로 낮춥니다. 즉 이 버그가 재발해도 **호스트는 죽지 않고 다른 후보로 추천이
완료**됩니다.

## 근본 수정 방향 (upstream 저장소에서)

```python
    def read_wait(self, p):
        last = ""
        while "Waiting" not in last and last != "Checking Non-Exited Systems ...\n":
            last = p.stdout.readline()
            if last == "":                      # EOF: 자식 종료
                raise RuntimeError(
                    f"astra-sim stream closed (exit={p.poll()})")
        return last
```

핵심은 (a) EOF에서 예외/종료, (b) 전체 줄을 보관하지 않음.

`check_end()`는 다르게 다뤄야 합니다: 루프 조건이 `out[-2]`를 보기 때문에 **종료
마커 뒤에 EOF를 한 번 읽는 것이 정상 종료 경로**입니다. 따라서 EOF에서 무조건
예외를 던지면 정상 완주가 깨집니다 — 마커를 못 본 상태의 EOF에서만 예외를 던져야
합니다.

## 수정 검증 (2026-08-03)

`backends/upstream` 브랜치 `fix/read-wait-eof-guard` (커밋 `95d13be`):
`read_wait()`는 EOF에서 즉시 `RuntimeError`, `check_end()`는 마커 없는 EOF에서만
예외, 양쪽 모두 마지막 8줄만 보관(`_TAIL_LINES`), 읽기 스트림의 `flush()` 제거.

| 검증 | 결과 |
|---|---|
| 단위(가짜 프로세스 6종) | 정상 파싱(`out[-2]`)·EOF 즉시 예외(0.0000s)·1만 줄 입력에도 tail 8줄 유지·정상/에러 마커 종료·마커 없는 EOF 예외 모두 통과 |
| 폭주 재현 설정 (70B tp4×2, 50요청) | 이전 79초에 17.4GB → **10분 동안 최대 78MB**, Python·astra-sim 모두 ~100% CPU(누수 없이 계산만 느림) |
| 자식 강제 종료(`kill -9`) | 이전 무한 spin → **약 2초 만에 종료**, `RuntimeError: ASTRA-Sim closed its output stream before reporting 'Waiting' (exit code: -9)` |
| 회귀 (8B tp1, 10요청) | exit 0, 요청별 CSV 정상 생성 — `check_end()` 정상 EOF 경로 유지 |

플래너 관점에서는 이제 메모리 가드가 16GiB에서 죽이는 대신 시뮬이 스스로 **의미 있는
메시지와 함께 non-zero 종료**하므로, `Infeasible` 사유에 원인이 그대로 실립니다.

### 운영 주의

`scripts/setup.sh`의 `git submodule update --init --recursive`는 부모가 기록한 SHA
(`c84e58b`)로 **detach** 시키므로 이 수정이 워킹트리에서 사라집니다(브랜치 ref는 남으므로
`git -C backends/upstream checkout fix/read-wait-eof-guard`로 복구). 영구 반영은
포크 push + 포인터 갱신, 또는 upstream(casys-kaist)에 PR이 필요합니다.

### 부수 발견 (수정과 무관, 기존 버그)

`serving/__main__.py:503`의 `RATIO = FREQ // INTERVAL`은 `--log-interval > 1.0`이면
0이 되어 리포트 단계에서 `ZeroDivisionError`가 납니다(예: `--log-interval 5.0`).
플래너는 항상 1.0을 쓰므로 영향받지 않습니다.

## 함께 발견한 갭

웹앱을 `--reload`로 재시작하면 진행 중이던 Stage-2 시뮬 서브프로세스가 **고아로
남습니다**(소유 잡은 in-memory `jobs` dict와 함께 소멸). 컨테이너에는 주기적
reconcile이 있지만 시뮬 프로세스에는 없습니다. 실측에서 고아 프로세스가 CPU/메모리를
계속 쓰는 것을 확인해 수동 정리했습니다.
