# Development Status (DEV STATUS)

> **이 문서의 목적**: ReMEmbR × NPU 포팅 프로젝트의 **살아있는 추적 문서**.
> "현재 어디까지 왔는지, 무엇이 검증되었는지, 무엇이 남았는지, 왜 이렇게 결정했는지"에 대한 단일 진실 공급원.
> 작업할 때마다 아래 **① 상태 보드**와 **⑥ 변경 이력**을 갱신할 것.
> 상세 절차와 설계는 [문서 맵](#7-document-map)의 개별 문서를 참고.

_최종 수정: 2026-07-21 · branch `main`_

---

## 1. Status board (at a glance)

**목표**: ReMEmbR 파이프라인(captioning → vector memory → QA)을 3B 모델
(Qwen2.5-VL captioner + Llama-3.2-3B LLM)로 **BOS NPU** 위에서 end-to-end로 구동하고, 논문의 NaVQA/CODa로 평가한다.

| Milestone | Status |
|---|---|
| Unified NPU inference server (LLM+VLM, HTTP) | ✅ Done |
| NonAgent `tt/` backend + JSON-constrained decoding | ✅ Done |
| NPU captioner (Qwen2.5-VL) | ✅ Done |
| end-to-end smoke test | ✅ Done |
| Original paper data acquired (data.csv + CODa seq0) | ✅ Done |
| ROS-free CODa→.pkl extraction | ✅ Done (8213 frames) |
| Captioning pipeline (checkpoint/resume) | ✅ Done (전체 실행에서 resume 검증됨) |
| Server stabilization (clean shutdown) | ✅ Done |
| Slowdown fix ① Inspector | ✅ Done (`TT_METAL_INSPECTOR=0`) — 19→2.1 it/s 감쇠 제거 |
| Slowdown fix ② Tracy build | ✅ Done — 기본 빌드로 교체(`ENABLE_TRACY=OFF`, Tracy symbols 705→0). 측정: RSS +33MB/request → **+0~1MB**, latency +1.0s/request → **flat 36s** |
| **Question JSON generation** (form_question_jsons) | ⚠️ 30 questions 생성됨 — captions 재생성 후 **반드시 재생성해야 함**(optimal context가 captions에서 추출되므로) |
| Server temperature support | ✅ Done (`b2dab02`) — eval의 무한 retry 루프 해결. 측정: temp=0 동일 / 0.7 매번 다름 |
| **Caption regeneration** (quality consistency) | 🔄 **109/272** — Tracy 재빌드 후 재개, **flat 36s/caption**(누수 없음) |
| **Evaluation run** (eval.py, optimal+tt/) | ⬜ caption 재생성 이후 |
| Multi-frame captions (paper fidelity) | ⬜ Optional |

### Measured hardware specs (`tt_server/npu_info.py`)

| Item | Value |
|---|---|
| arch / devices | `BLACKHOLE` / 1 device (`/dev/bos/0`, module `bos` 2.3.0) |
| **DRAM** | **12 GiB** = 4 GiB × **3 banks** (`dram_grid_size (3,1)`) |
| Tensix compute grid | **5×4 = 20 cores** (harvesting mask `0x0` = 모든 코어 정상) |
| L1 / worker core | 1.5 MB (eth cores 512 KB) |
| PCIe | `0000:01:00.0` Co-processor (Synopsys DWC) |

> **12 GiB**라는 수치가 "3B+3B는 공존 가능, 8B는 불가"의 근거다. 평가 시 8B를 쓰려면
> 단계별로 모델을 하나씩만 로드해야 한다(`TT_ENABLE_VLM=0`).

---

## 2. What works today (verified)

- **NPU server**: Llama-3.2-3B + Qwen2.5-VL-3B 공존, 단일 worker + queue 직렬화, `/health` `/caption` `/v1/chat/completions`.
- **JSON-constrained decoding**: NonAgent이 100% 파싱되는 JSON 응답을 받음, retry 루프 없음(~10–12s/query).
- **Captioning**: 단일 CODa frame → 정상 caption(거부 없음), 연속 3 요청에서 안정적.
- **Data**: `data.csv` (210 questions) + 8213 CODa seq0 `.pkl` files (`/home/user/coda_raw/coda_data/0/`).
- **Board recovery**: wedged 상태의 좀비 프로세스 정리 → 재시작으로 복구 성공(리부팅 없이).

## 3. In progress / next steps

1. 🔄 **caption 재생성 완료 대기** — 모두 256 tokens로 통일(~2.9h). 완료 시 최종 JSON을 쓰고 partial을 자동 삭제.
2. **question JSON 재생성** — captions가 바뀌었으므로 optimal context를 갱신하려면 `form_question_jsons.py`를 다시 돌려야 함.
3. **평가 실행** — `eval.py --model optimal+tt/Llama-3.2-3B-Instruct --sequence_id 0 ...` (temperature 기본값 0.7).
4. (Optional) **multi-frame captions**로 논문 fidelity 개선 — 절차는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 4-1.

## 4. Known issues / gotchas

| Issue | Status / response |
|---|---|
| **Server slowdown** (두 가지 원인) | **① tt-metal Inspector** (debug, 기본 ON): workload마다 YAML `flush()` → caption당 ~5,000 synchronous I/O 호출, 실행당 153MB / 5.34M lines 누적 → 19→2.1 it/s. `TT_METAL_INSPECTOR=0`로 해결([8be8bc6]).<br>**② Tracy profiling build**: Inspector를 끈 뒤 드러남(84→50 it/s). 같은 이미지 12회 실행 동안 **RSS +33MB/request, latency +1.0s/request, 선형 증가**. `build -> build_Release_tracy` (`ENABLE_TRACY=ON`)라서 Tracy가 zone을 계속 누적. **compile-time switch라 env var로 끌 수 없음** → ✅ 기본 빌드로 교체(`./build_metal.sh`, TRACY OFF) ([458bf09], symbols 705→0). 측정: **flat 36s, RSS +0MB**. Rollback: `ln -nsf build_Release_tracy build`. |
| **Caption sentences truncated** | 서버 기본 max_gen_len=128에서 잘림. captioner가 256을 보내도록 해서 해결([8caa7c9]). 재시작 후 적용됨. |
| **Board wedged** | 원인 = captioning/compilation 중 강제 종료(특히 Ctrl-C 연타). shutdown cleanup 로직 추가([f9bc7bf]). **`-9`는 절대 쓰지 말 것.** 복구는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 6장. |
| **Reboot constraint** | 이 PC는 부팅 전에 NPU가 켜져 있어야만 NPU를 인식함. wedged 상태에서는 리부팅 전에 프로세스 정리 + 재시작을 먼저 시도할 것. |
| **Device node = `/dev/bos/0`** | Tenstorrent이 아님(`/dev/tenstorrent`가 없음). `fuser -v /dev/bos/0`, `lsmod \| grep bos`로 확인. |
| ~~**eval infinite retry**~~ | ✅ **Fixed** — 단, **두 커밋에 걸쳐 있다.** ① [b2dab02]가 서버를 고쳤다: temperature를 무시하고 greedy로 돌아서 eval의 retry가 매번 같은 답을 냈던 문제(0은 greedy 유지, >0은 sampling). ② 그런데 `NonAgent.query()`의 `while True`에 **상한이 없어서 루프 자체가 남아 있었다** — `temperature=0`이면 여전히 무한 루프였다. [c997922]가 `MAX_PARSE_RETRIES=5`로 닫았다. **①만 보고 "해결됨"으로 읽으면 오진한다**(실제로 그랬다 — 아래 2026-08-03 changelog 참고). |
| **3B quality** | JSON validity는 보장되지만 grounding은 3B의 한계에 부딪힘 — 질문 유형에 맞는 필드를 채우지 못하는 경우가 있음(retry로 흡수됨). **왜 8B로 올리지 않는지는 design decision 5 참고**(로봇 배치에는 공존이 필요함). |
| ~~**Mixed caption quality**~~ | 🔄 진행 중인 재생성으로 해소 중. (기존 mixed set은 `*.MIXED-BACKUP.json`으로 보관) |
| **Fidelity deviations** | Captioner VILA→Qwen, LLM 8B→3B(포팅상 불가피), frames 6→1(타협). fidelity 표는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 참고. |
| **venv dependency trap** | 새 패키지가 `langchain-core`를 0.3으로 또는 `numpy`를 2.x로 끌어올리면 스택이 깨짐. pin을 지킬 것 ([RUNBOOK](../tt_server/RUNBOOK.md) 1-3). |

## 5. Key design decisions (why we did it this way)

- **두 venv를 HTTP 뒤로 분리** — ttnn/transformers(server)와 langchain/pymilvus(remembr) 사이의 버전 충돌을 피함.
- **NonAgent 채택** (ReMEmbRAgent 대신) — 3B에서 tool-calling은 불안정. 단일 JSON 호출이 NPU에 적합.
- **LLM을 3B로 유지(8B로 안 감)** — 평가만 보면 eval 시점에는 VLM이 필요 없으므로
  `TT_ENABLE_VLM=0`으로 8B만 로드해 더 나은 품질을 얻을 수 있음(8B는 12GiB DRAM에 단독으로만 들어감). **하지만 최종
  목표는 로봇 배치**이고, 주행 중에는 caption 생성과 QA가 동시에 일어나므로 **두 모델이 반드시 공존해야 함.**
  8B+VLM은 12GiB에 들어가지 않음 → **3B+3B가 유일하게 배치 가능한 구성.** 시간 분리 로딩은
  실제 배치에서 쓸 수 없는 eval 전용 최적화라, **배치와 동일한 3B 구성으로 평가함**(평가의 대표성 유지).
- **서버가 temperature를 존중** — 원본 eval은 `temperature=0.7`(기본)으로 retry하면 다른 답이 나온다는
  전제로 retry 루프를 돌림. 서버가 greedy로 하드코딩되어 있으면 그 계약이 깨지고 루프가
  끝나지 않음. eval 코드를 패치하는 대신 **서버가 계약을 지키도록** 함(temperature=0은 기존처럼 greedy 유지).
- **JSON-constrained decoding** — Ollama의 `format="json"`을 대체. 3B의 무한 retry와 헛소리 출력을 방지.
- **Captioner = HTTP client** — VLM을 remembr 프로세스에 직접 로드하지 않음(dependency / device-ownership 충돌).
- **`<video>` 제거, video→image 프롬프트** — VILA 잔재. still frame에서 Qwen의 거부를 유발함.
- **ROS-free extractor** — 원본 `preprocess_coda.py`의 ROS 의존성은 시각화용일 뿐. 텍스트 pose + PNG만으로 동일한 .pkl 생성.
- **Checkpointing (JSONL append)** — 긴 captioning 실행(~70–95 min)이 중단돼도 손실 없이 resume.
- **Clean shutdown + teardown 중 signal 무시** — worker가 daemon이라 device close가 건너뛰어지던 문제를 고치고, Ctrl-C 연타로 인한 wedging을 방지.

## 6. Changelog

### 2026-08-14 — try0_plane 벡터 메모리 적재 및 대화식 RetrieveAgent CLI

`remembr/data/try0_plane_preprocessed_5/0/`의 PKL과 기존 Qwen3-VL terse 캡션/1024차원 임베딩
29개를 검증한 뒤 전용 Milvus 컬렉션에 적재하고, `RetrieveAgent`로 단일 또는 반복 질의를 보내는
`remembr/scripts/interactive_retrieve.py`를 추가했다. 기본 `auto` 적재는 비어 있는 컬렉션만
채우며, 기존 항목 수가 캡션 수와 다르면 데이터 혼합을 막기 위해 중단한다. 대화식 질의마다
working memory를 초기화해 이전 검색 결과가 다음 답변 컨텍스트에 남지 않는다.

검증: host-only 데이터 경계 테스트와 CLI help 통과. PKL 559개에 대한 캡션 JSON의 58개
시작/끝 참조와 임베딩 차원 확인 완료. 로컬 Milvus에 전용 컬렉션을 생성해 29개 적재 및 flush 후
entity count 29와 재실행 시 중복 없는 재사용을 확인했다. `llm-8b` 실제 질의 "Where is the
white plastic box?"에서 검색어 `White plastic storage container`를 생성했고, 정답 캡션/좌표가
Milvus top-1로 검색되어 최종 position `[-0.708, -0.059, 1.215]`까지 end-to-end 응답했다.

### 2026-08-14 — NonAgent 응답을 position 하나로 고정

`NonAgent`가 답변 유형을 별도 LLM 호출로 분류한 뒤 7개 필드를 모두 생성하던 경로를 position
전용 단일 생성으로 바꿨다. NPU JSON schema는 `{"position": [x, y, z]}`만 허용하며 좌표를 숫자
3개로 강제한다. 프롬프트에서도 text/time/duration/binary/orientation 지시를 제거해 스키마와
모델 지시가 충돌하지 않게 했다. 외부 `AgentOutput`/eval 계약은 유지하기 위해 파싱 후 나머지
필드는 `None`으로 채운다. position 외 유형의 질문은 이 경로의 평가 대상이 아니다.

검증: position-only schema/prompt, bounded retry, `RetrieveAgent` 배선 self-check 통과. 실제 NPU
요청은 서버가 내려가 있어 대기 중이다.

### 2026-08-12 — VLM caption 해상도 실험과 프레임 샘플링 버그 수정

- `/caption`이 모든 frame을 무조건 224x224로 찌그러뜨리던 경로에 `TT_VLM_IMAGE_LONG_SIDE`와
  `TT_VLM_IMAGE_FIT`(`stretch`/`letterbox`/`contain`)를 추가했다. 기본값은 기존과 같은
  `224 + stretch`라 기존 호출 shape는 바뀌지 않는다.
- tt-metal 기본 warmup은 224x224 dummy라서, 224가 아닌 요청 실패가 "크기 문제"인지
  "warmup shape mismatch 문제"인지 분리되지 않았다. 실제 요청 shape를 먼저 warmup하고,
  `TT_VLM_WARMUP_SHAPE=WIDTHxHEIGHT`, `TT_VLM_WARMUP_LAST=1`로 실험 순서를 조정할 수 있게 했다.
- `/memory`를 추가해 DRAM/L1 allocator의 `allocated`, `free`, `largest_free`를 읽는다.
  반복 caption 중 누수와 조각화를 구분하기 위한 read-only 관측 endpoint다.
- device 작업 실패가 HTTP caller에게만 전달되고 서버 로그/snapshot에는 남지 않던 문제를 줄이기 위해
  worker exception을 로그에 남기고 runtime state를 `FAILED`로 전환한다. `run_server.sh`는
  `TT_METAL_OPERATION_TIMEOUT_SECONDS=300`을 기본으로 설정해 device poll 무한 spin을 timeout으로 바꾼다.
- `preprocess_captions.py`의 frame sampling을 stride(`images[::30//n]`)에서 `np.linspace` 기반
  균등 샘플링으로 바꿨다. 이제 `--num-video-frames 1`이면 segment마다 정확히 한 frame만 들어간다.
- 검색 index용 짧은 caption prompt를 `--caption_style terse`로 추가했다. 논문 재현 기본값은
  `paper` 그대로다.
- 보드 관측 스크립트 `tt_server/tests/board/caption_sample.py`,
  `tt_server/tests/board/probe_aspect_ratio.py`를 추가했다. 둘 다 서버 lifecycle은 건드리지 않고
  `/health`, `/caption`, `/memory`만 읽거나 호출한다.

### 2026-08-07 - `/health`에 device worker queue 상태 노출

- 8B VLM captioning이 몇 처리 후 오래 멈춘 것처럼 보일 때, 서버가 죽은 것인지 device worker가 긴 `/caption`  요청을 처리 중인지 구분하기 어려웠다.
- `RuntimeSnapshot`에 worker 상태를 포함하고, `/health.worker`로 `busy`, `queue_size`, `active_job`, `active_elapsed_s`를 노출했다.
- 이 값들은 Python worker/queue 상태만 읽는 가벼운 관측 정보이며, NPU device에 추가 query를 보내지 않는다. 
- 기존 `caption`, `/v1/chat/completions`, `/mode` 응답 shape와 NPU lifecycle은 변경하지 않았다.

### 2026-08-06 — `/health`에 LLM debug flag 노출

- `TT_DEBUG_DECODE`/`TT_DEBUG_SAMPLE`은 서버 시작 시 env에서 읽히지만, 정식 문서와 `/health`에 노출되지 않아 실행 중인 서버가 어떤 debug 출력 모드인지 확인하기 어려웠다.
- `RuntimeSnapshot`에 debug flag를 포함하고 `/health` 응답에 `"debug": {"decode": ..., "sample": ...}`를 추가했다.
- README/RUNBOOK의 env var 표에 두 debug env의 용도와 주의점을 추가했다. 
- 런타임 토글, decode 경로, NPU lifecycle은 변경하지 않았다.

### 2026-08-05 (2) — 답변 스키마가 실제로 제약하지 않던 문제 + 유형 고정 (PR #37, #40)

**① `maxLength`가 적혀 있는데 안 먹혔다 (PR #37).** 보드에서 `orientation`이 폭주했다 —
`"[-0.92, 0.21, 0.12, 0.12, 0.21, -0.92, ...`를 max_tokens까지 반복하다 JSON이 문자열
중간에서 잘려 `Unterminated string starting at: line 1 column 103`으로 파싱이 죽었다.
스키마에는 `maxLength: 40`이 있었다.

원인은 **lm-format-enforcer가 JSON-Schema 타입이 리스트면 형제 키워드를 무시하는 것**이다.
실제 Llama-3.1 토크나이저로 문자열을 한 글자씩 넣어보며 측정:

| 스키마 | `maxLength: 40` |
|---|---|
| `{'type': 'string', 'maxLength': 40}` | 40자에서 멈춤 |
| `{'anyOf': [{'type':'string','maxLength':40},{'type':'null'}]}` | 40자에서 멈춤 |
| `{'type': ['string','null'], 'maxLength': 40}` | **60자 넘어서도 계속** |

즉 nullable을 union으로 쓰면 제약이 사실상 없다. `#31`에서 `position`을 `anyOf`로 바꿨더니
폭주가 멈춘 것이 우연이 아니었다 — **제약이 살아남는 형태였던 것**이다. 두 스키마의 union
필드 7개를 모두 `anyOf`로 바꿨다. `time`/`duration`은 `maxLength`조차 없는 union이어서
문자열 분기가 완전히 무제한이었다.

**② 그런데 `orientation`은 애초에 문자열이 아니었다 (사용자 지적, PR #37에 추가).**
상한을 씌우는 것으로 끝낼 문제가 아니었다:

| 출처 | 선언 |
|---|---|
| `AgentOutput.orientation` (upstream NVIDIA) | `float` |
| `MemoryItem.theta` | `float` (지상 로봇이라 yaw 하나) |
| 모델이 읽는 메모리 문자열 | `"with an average orientation of {theta} radians"` |

**하나의 float을 세 곳이 다르게 적고 있었다** — `NonAgent`은 문자열, `ReMEmbRAgent`은 숫자
1개 배열, 프롬프트 예시는 `"[euler_rotation]"`. eval이 이 필드를 채점하지 않아서 아무도
눈치채지 못했고, 그래서 여기가 폭주 지점이 됐다. `number`로 선언하면 폭주를 **상한으로
묶는 게 아니라 없앤다** — 숫자 문법은 유한하므로 `maxLength`가 불필요하다.
**교훈: "제약이 없다"보다 "타입이 틀렸다"를 먼저 의심할 것.** 내 첫 수정은 증상 완화였다.

**③ 좌표가 텍스트 안에만 있고 필드는 null이었다 (PR #40).** position 질문에서 모델이
정답 좌표를 답변 텍스트에 써놓고 필드를 비웠다. 조각상 질문, 정답
`[-60.1973617419355, 196.54061870967743, -1.096614193548387]`:

    type = "text"    position = None
    text = "I'm not sure where the statue ... is, but ... Based on my previous
            location at [-60.197, 196.541, -1.097] and th..."

**소수점 3자리까지 일치한다.** 검색도 인식도 좌표도 맞았고 필드만 비었는데, eval은 빈 필드를
미답변으로 채점하고 attempt 5회를 소진한다. 모델은 프롬프트를 위반하지 않는다 —
`type="text"`를 골라 **우회**한다(text 유형에는 null position이 일관적이다).

수정: 답변 유형을 별도 호출로 확정하고(enum 제약이라 5개 외는 생성 불가, 출력이 단어
하나라 질문당 ~1초), 그 유형의 필드를 null 불가로 못 박은 스키마로 답을 받는다. `type`도
`const`로 고정해 갈아타기를 막는다. **질문을 정규식으로 분류하지 않고, eval 라벨도 보지
않는다** — 모델이 결정한다(검색어 생성에서 통한 것과 같은 접근).

분류기 조정 2건도 보드에서 재고 고쳤다: `max_tokens` 16 → 48(문법이 선행 공백을 허용하고
모델이 개행을 뱉어서 예산이 잘렸다), 프롬프트에 "Where is X는 X의 설명이 길어도 항상
position" 명시(조각상처럼 긴 수식어가 붙으면 text로 분류했다). 분류 정확도 7/9 → **9/9**.

**측정 (9문제, temperature=0, 8B):**

| | 미답변 | position 답변 | 5m내(/4) | time 평균 | 총 시간 |
|---|---|---|---|---|---|
| optimal+ 전 | 2 | 2/4 | 1/4 | 1.47 | 543s |
| **optimal+ 후** | **0** | **4/4** | **2/4** | **1.26** | **299s** |
| retrieve+ 전 | 2 | 2/4 | 0/4 | 6.51 | 597s |
| **retrieve+ 후** | **0** | **4/4** | 0/4 | **1.25** | **333s** |

**호출을 하나 더 넣었는데 전체가 45% 빨라졌다** — 미답변으로 소진되던 재시도 5회가 사라지는
이득이 분류 호출(~1초)보다 훨씬 크다.

**대가가 있다(정직하게):** 조각상이 미답변 → **171.40m(틀림)** 이 됐다. 필드를 강제하면
모델이 커밋하지만 그 좌표가 맞다는 보장은 없다. 그리고 고정 **전에는** 정답 좌표를 텍스트에
정확히 써놓고 필드를 비웠는데, 고정 **후에는** 필드를 채우며 **다른** 좌표를 골랐다 —
"텍스트의 좌표를 필드로 옮기기"와 "필드 강제"는 같은 동작이 아니다. 로봇에 좌표를 넘기는
관점에서는 중립이 아니므로, n=12로 미답변→틀린답 비율을 재야 판단할 수 있다(진행 중).

**측정 설계 주의**: 위 수치는 position 4문제 기준이라 **한 문제가 25%p를 움직인다.**
`미답변 2→0`과 `시간 45%↓`는 메커니즘/집계라 단단하지만, `5m내 1/4→2/4`는 약하다.
그리고 SUBSET에는 **binary 질문이 0개**여서 `binary` enum 제약이 보드에서 검증되지 않았다
(`human_qa` 30문제에는 4개).

### 2026-08-05 — 먼저 온 요청 종류가 이기고 나머지는 영구히 깨지던 문제 (#21)

LLM 엔진에 decode 경로가 두 개 있었다. plain + `temperature=0`은 device가 argmax까지
해서 **토큰 번호**를 돌려주는 경로, JSON 제약은 **logits 전체**를 호스트로 읽어 마스킹하는
경로. 그런데 tt-metal은 decode trace를 `hasattr(self, "trace_ids_text")` 하나로만 캐시한다
(`generator.py:370`) — **trace가 무엇을 돌려주는지는 캐시 키에 없다.** 그래서 프로세스에
먼저 도착한 요청 종류가 trace를 차지하고, 다른 종류는 그 trace를 재생해 엉뚱한 버퍼를 읽는다.
프로세스가 죽을 때까지 복구되지 않는다.

같은 서버에서 요청 순서만 바꿔 확인한 2×2:

| 서버 상태 | plain 요청 | JSON 제약 요청 |
|---|---|---|
| 재시작 직후, 한 종류만 | 정상 | 정상 |
| plain을 먼저 1회 | 정상 | **`'{"!!!!!!!!...'`** |
| JSON을 먼저 1회 | **`'Hello9001/+,,+,**+++'`** | 정상 |

`!`인 이유는 토큰 0이 `!`이고, 허용 토큰 집합에서 argmax를 하는데 넘어온 버퍼가 logits가
아니라서 사실상 임의 값이 되기 때문이다. enforcer는 정상이다 — 오프라인에서 실제
토크나이저로 재현해 첫 6스텝 허용 토큰 수가 283 → 270 → 254 → 229 → 209 → 179 으로
한 번도 비지 않는 것을 확인했다.

**수정: device 경로를 삭제하고 host 경로 하나만 남겼다** (`-53/+29`). trace가 한 종류만
캡처되므로 불일치가 구조적으로 불가능해진다. tt-metal 소스는 건드리지 않았다. 비용은
토큰당 logits 513 KB 호스트 read(PCIe ~20 µs) 대 측정된 decode 293.5 ms/token이라
0.1% 미만이고, 앞서 두 경로를 end-to-end로 비교했을 때도 ~3%였다(위 2026-08-03 항목).
대안이었던 "tt-metal trace 캐시 키에 `argmax_on_device`를 추가"는 더 옳지만 tt-metal
수정이 필요하고, device 경로의 이득이 0.1% 수준이라 유지할 이유가 없어 택하지 않았다.

**이 버그가 실제로 망친 것**: 측정을 조용히 오염시켰다. planner의 검색어 생성 원인을
분리하려던 실험이 질문마다 `DESC(평문) → JSON1 → JSON4` 순서로 돌아서, 첫 평문 호출이
서버를 오염시키고 뒤이은 JSON 변형 전부가 `!!!!`가 됐다. **그 실험의 "JSON1 0/4,
JSON4 0/4"는 모델 능력이 아니라 이 버그의 결과이고 무효다** (평문 경로인 DESC 2/4만 유효).
#21을 "eval 경로에는 영향 없다 / 우선순위 낮음"으로 열어둔 것도 틀렸으므로 정정했다.

검증: `tt_server/tests/test_single_decode_path.py` — decode 호출 수와
`sampling_params=None` 수가 같은지 본다. 옛 코드에 돌리면 `2 != 1`로 실패한다.
보드가 필요한 진짜 검사 대신, 속도를 위해 device 경로를 되살리는 재발을 막는 가드다.

### 2026-08-03 — eval을 27분간 멈춘 무한 루프 + 서버가 잘못된 모델을 보고한 문제

**① `NonAgent` 무한 재시도 ([c997922], PR #19).** 30문제 eval이 10번 질문에서 27분간
멈췄다. 연쇄: 모델이 `position`에 `[0, 0, 0], `를 끝없이 반복 → `max_tokens=256`에 걸려
JSON이 문자열 중간에서 잘림 → `json.loads`가 `Unterminated string` → `NonAgent.query()`의
**상한 없는 `while True`**가 재시도 → `temperature=0`이라 출력이 바이트 단위로 동일 →
**무한 루프**. 로그에 증거가 있었다(`"Generate call failed"` 27회 vs `"Attempt elapsed"`
21회 → 차이 26회가 내부 재시도, 26×~35초 ≈ 910초 = 관측된 975초).
수정은 두 층: 스키마의 `position`/`orientation`에 `maxLength`(80/40) 부여 — **없는 길이
제한이 폭주를 구조적으로 가능하게 한 원인** — 그리고 `while True` → `MAX_PARSE_RETRIES=5`
후 raise(eval의 `MAX_ATTEMPTS`가 unanswered로 기록하고 다음 질문으로 진행).
검증: `remembr/agents/test_non_agent_retry.py` 3개 통과, 그리고 **테스트를 옛 `while True`
코드에 돌리면 timeout(exit 124)으로 무한루프를 재현**하는 것까지 확인.

**② 서버가 요청받은 모델 이름을 그대로 되돌려줬다 ([acd6aa3], PR #17).** 보드는 한 번에
모델 하나만 올리고 요청의 `model`은 존중되지 않는데(교체는 `/mode`), 응답에
`request.model or snapshot.llm_model`을 넣어서 **돌지도 않은 모델이 돌았다고 보고**했다.
`eval.py`는 결과 파일 이름을 자기 `--model` 인자로 짓기 때문에 8B가 만든 결과가 3B 결과로
저장된다 — 이 세션에서 실제로 그렇게 오인한 측정을 했다. 항상 `snapshot.llm_model`을 보고한다.

**진단 과정에서 얻은 성능 실측** (8B/accuracy, `AGENTS.md`에도 반영):
지연 ≈ 요청당 **3초** + 생성 **토큰당 0.26초(≈3.8 tok/s)**. JSON 강제 디코딩은 같은 토큰
수에서 device greedy와 **~3% 차이뿐**(plain 28.6s vs json 29.6s) — 기존 `AGENTS.md`의
"device-side greedy보다 느리다"는 서술은 과장이라 정정했다. 프롬프트 길이 영향도 작다
(22→529 words에서 3.0→4.3초). `temperature=0.7`로 바꾸는 것만으로 동일 9문제에서
LLM 824초→**445초**, attempts 25→12, 미답변 4→**0**.

**교훈(문서에 반영)**: `## 4`의 "eval infinite retry ✅ Fixed" 항목이 ①의 앞쪽 절반만
가리키고 있어서, 이 세션에서 나는 그 항목을 보고 무한 루프 가능성을 배제한 뒤 엉뚱한 곳
(JSON 디코딩 속도)을 의심했다. 부분 수정을 "Fixed"로 적으면 다음 사람이 오진한다.

### 2026-08-03 — docs: `run_server.sh` 예시 명령에서 불필요한 tt-metal cd/activate 제거

- `run_server.sh`가 상대경로로 실행하면 안 뜨던 버그는 아래 "중복 NPU 서버 기동 방지
  startup guard" 커밋이 `BASH_SOURCE` 기준 `SCRIPT_DIR`을 도입하며 이미 고쳐져 있었다
  (같은 문제를 별도로 다시 고치려던 시도는 폐기하고 `SCRIPT_DIR`을 그대로 재사용).
- 그런데 README.md/PAPER_EVAL_RUNBOOK.md의 예시 명령은 여전히 `cd /home/user/tt-metal` 후
  절대경로로 `run_server.sh`를 실행하도록 안내하고 있었다 — venv는 스크립트가 알아서
  켜므로 그 cd는 애초에 불필요했다. `cd tt_server && ./run_server.sh`로 단순화.
- **검증**: `bash -n`; 문서에 적은 명령을 실행 중인 보드에서 실제로 돌려 잘못된 TT_MODE로
  device 건드리기 전에 fail-fast하는 것과 기존 서버(`/health`)가 영향받지 않는 것을 확인.

### 2026-08-02 (4) — NPU 서버 lifecycle을 사람 사용자 전용으로 명문화

- 공유 BOS 보드의 안전을 위해 서버 기동·종료·재시작은 **사람 사용자가 지정된 tmux 세션에서만**
  직접 수행하도록 `AGENTS.md`와 서버 `RUNBOOK.md`에 운영 원칙을 추가했다.
- AI 에이전트는 `/health`, process, tmux pane/log의 읽기 전용 확인만 허용하며,
  `run_server.sh`, background server, `tmux send-keys`, signal/kill/restart를 금지한다. lifecycle
  변경이 필요하면 사용자에게 직접 조작을 요청한다.
- 문서 전용 변경. `git diff --check`로 형식을 검증했다.
### 2026-08-02 — 중복 NPU 서버 기동 방지 startup guard

- `run_server.sh`가 tt-metal import/device open 전에 `server_preflight.sh`를 실행하도록 변경했다.
  기존 `/health`가 응답하면 health를 출력하고 exit 0 no-op, health 없이 server process 또는
  `/dev/bos/0` holder가 있으면 원인을 출력하고 exit 1로 차단한다.
- server lifetime 동안 상속되는 전역 `flock`을 추가해 두 launcher가 동시에 idle 검사를 통과하는
  race를 막았다. script 위치도 `BASH_SOURCE`의 absolute path로 고정해 launcher가 `cd`한 뒤에도
  entrypoint 경로가 깨지지 않는다.
- **검증**: 실제 서버 lifecycle은 건드리지 않고 mock curl/pgrep/fuser와 실제 flock을 사용한
  preflight unit test 6개(ready no-op, unhealthy process, device holder, free device, concurrent lock,
  server process의 lock FD 상속), 전체 server test 20개, `bash -n`, `git diff --check` 통과.
### 2026-08-02 (3) — NPU 서버 레이어드 아키텍처 마이그레이션 (`refacoter` branch)

- 772줄 단일 `tt_infer_server.py`를 `presentation/http → application → domain`과
  `infrastructure/tenstorrent` 계층으로 분리. 기존 파일은 `run_server.sh` 실행 경로를 보존하는
  얇은 entrypoint로 유지했다. 외부 endpoint와 OpenAI 응답 shape는 유지한다.
- 전역 `_jobs/_state/_mesh` 대신 `TenstorrentRuntime` actor가 device와 엔진을 소유한다.
  모든 open/load/infer/switch/close는 하나의 non-daemon worker에서 실행되고 외부에는 immutable
  lifecycle snapshot만 노출된다.
- 초기 model load 예외에도 startup wait가 반드시 풀리고 device가 닫히도록 수정. 모드 로딩 실패는
  이전 mode를 거짓으로 유지하지 않고 `failed` 상태와 오류를 노출한다.
- Pydantic HTTP schema와 application 오류 매핑 추가. 검증되지 않은 6-frame 이상 요청은 device에
  도달하기 전에 400으로 거부한다(`TT_MAX_CAPTION_FRAMES`, 기본 5).
- **검증**: compileall 및 fake device/engine 기반 unit·HTTP contract test 14개를
  기본 venv·tt-metal 기본 venv·`tf457`에서 통과(직렬 실행, close→reopen, switch 실패 후 복구,
  startup failure cleanup, 기존 응답 shape). 두 tt-metal 환경에서 Llama/Qwen2.5/Qwen3 backend import도
  확인했다. 실제 BOS NPU에서 `3b+3b` LLM JSON + VLM caption 추론 성공, `caption-3b → 3b+3b`
  close/reopen 왕복 전환과 중간 LLM 409·복귀 후 추론 성공, 단일 Ctrl-C 후
  `NPU released cleanly`와 port 해제까지 확인했다.

### 2026-08-02 (2) — eval 중 Ctrl-C가 안 먹히던 문제, 근본 원인 확정 + 수정
사용자 보고: "remembr eval 돌리다 서버가 잘 안 돌아가서 Ctrl-C 했는데 30분 기다려도 안 내려갔다."

**근본 원인**: `uvicorn.run()`의 `timeout_graceful_shutdown` 기본값이 `None` — SIGINT를 받아도 진행 중인
HTTP 요청이 끝날 때까지 **무한정** 대기 후에야 리턴한다. 서버 핸들러(`/caption`, `/v1/chat/completions`)는
sync `def`라 스레드풀에서 돌고, `_submit()`의 `fut.result()`엔 타임아웃이 없어 워커가 끝낼 때까지 블록.
`remembr+` 에이전트 질문은 평균 70~100초(WORKLOG §3.5)라, 그 사이 SIGINT가 오면 uvicorn이 그 요청을
기다리며 멈춘다. **여기서 멈추는 게 문제였다** — `main()`의 안전 종료 로직(`finally`에서
`worker.join(timeout=180)`으로 워커가 device를 깨끗이 닫을 때까지 기다리는 코드, 이번 세션 초반에 만든
것)은 이미 올바른데, `uvicorn.run()`이 리턴해야 그 코드에 도달하므로 **아예 실행되지 못하고 있었다.**
30분 뒤 사용자가 결국 강제 종료했을 가능성이 높고, 그게 연산 중이었다면 이 세션에서 반복 확인된
보드 wedge(force-kill 중 device teardown 스킵)로 이어졌을 것.

**수정** (`tt_server/tt_infer_server.py`, `main()`): `uvicorn.run()`에 `timeout_graceful_shutdown`
(기본 5초, `TT_SHUTDOWN_TIMEOUT`으로 조정 가능) 추가. uvicorn이 HTTP 연결단에서 무한 대기하는 것만
막을 뿐, 워커의 안전 종료(진행 중이던 실제 device 작업은 끝까지 실행 후 `close_mesh_device()`)는
그대로 보존 — 워커의 job queue가 종료 시그널을 진행 중인 작업 **완료 후**에만 집어들기 때문에 안전.

**실측 검증**: 8B 캡션(약 40초) 요청을 발사 후 3초 뒤 SIGINT → **38초 만에 프로세스 정상 종료**(진행
중이던 캡션이 끝까지 실행된 뒤 device 정리), 클라이언트는 11.7초 후 명확한 500 에러 수신(무한 행 아님),
종료 후 보드 독립 테스트로 생존 확인. 이전 "30분+ 무응답"에서 근본적으로 개선.

### 2026-08-02
- **모드 기반 모델 전환** 서버 구현 (`tt_server/tt_infer_server.py`): 클라이언트가 `POST /mode`로
  모델 조합(`3b+3b` / `caption-3b` / `llm-8b` / `caption-8b`)을 선택. 전환은 워커에서
  `synchronize → close_mesh_device → open → load`로 **device를 통째 반환**(회수 실패/누수 없음,
  tt-metal 내부 무접근). 불일치 요청 → 409, 전환 중 → 503. 검증: 반복 close+reopen 20회 무먹통,
  모드 전환 6회(4.53)·4회(4.57) 서버+보드 생존, 엔드포인트 매칭 정확. [[VLM_MODE_SWITCHING_PLAN]]
- **venv 통일 (transformers 4.57.1)**: qwen3_vl(4.57 요구)와 기존 3B 백엔드를 한 venv로. 게이트에서
  qwen25_vl 로드가 4.57에서 깨짐 — **원인: tt-metal `model_config.py:1435`가 빈 `_name_or_path`(4.57
  동작 변경)를 덮어쓰기 조건으로 오인** → 잘못된 AutoModel 분기. **load-bearing tt-metal 수정 1줄**:
  `if "_name_or_path" in config:` → `if config.get("_name_or_path"):` (하위호환, 4.53 불변). 이후
  4.57.1에서 3B 로드+LLM/VLM 추론+모드 전환 전부 정상, downstream 크래시 break 없음. 통일 venv =
  레포의 통합 venv `.venv` 하나로 네 모드를 모두 돌린다.
  - 미완: caption-8b(qwen3) 백엔드 배선 + 8B 가중치 다운로드; 출력 품질 4.53↔4.57 A/B.

### 2026-07-15
- `458bf09` **두 번째 slowdown 원인 = Tracy profiling build**: Inspector를 끈 뒤 드러남(84→50 it/s).
  **같은 이미지 12회 실행** 동안 RSS +33MB/request, latency +1.0s/request가 **선형** 증가(입력이 동일하므로 caching이 아니라 누적).
  `build -> build_Release_tracy` (`ENABLE_TRACY=ON`, 705 Tracy symbols)라서 `TRACY_ON_DEMAND` 없이
  Tracy가 모든 zone을 계속 수집. **compile-time switch라 env var로 끌 수 없음** → `./build_metal.sh`로 재빌드(기본 OFF, 7 min).
  **검증**: Tracy symbols 705→0, latency **8회 실행 내내 flat 36.0s**, RSS **+0MB**. 272 captions ~13h → **~2.9h**.
  tt-metal source 변경 없음; rollback은 `ln -nsf build_Release_tracy build`.
- `b2dab02` **서버가 temperature 존중 → eval의 무한 retry 해결**: `chat_completions`가 `temperature`를
  전혀 읽지 않았음; JSON 경로는 masked argmax, non-JSON 경로는 `SamplingParams(temperature=0.0)`을 써서 **greedy로 하드코딩**되어 있었음.
  원본 eval은 `--temperature 0.7`로 retry하면 다른 답이 나온다고 가정하므로, greedy 하에서는 retry가 영원히 루프.
  이제 temperature가 `chat_completions → generate() → sampler`로 전달되고, `_masked_argmax`가 `_masked_sample`로 바뀜
  (temp≤0 = 기존대로 greedy, >0 = grammar 허용 token에서 sampling). grammar constraint는 그대로라 JSON validity에는 영향 없음.
  **측정**: temp=0 → 3 요청에 unique 응답 1개; temp=0.7 → 3개(둘 다 schema-valid).
- 🔄 **Caption 재생성 시작**: mixed 품질(128/256 tokens 혼재)을 해소하려고 전부 재생성. 기존 set은 `*.MIXED-BACKUP.json`으로 보관.
  초기 측정: 길이가 \~1150자 부근으로 일관적, vision **83\~182 it/s**, caption당 VisionBlock 1개 — 앞선 수정들이 모두 동작함을 확인.
- `8aea788` **eval optimal 경로 버그 수정**: `TextMemory.memory_to_string`이 이미 string인 optimal
  context를 문자 단위로 순회하다 `doc.time`에서 터짐(→ eval retry 무한 루프). 이제 string이 그대로 통과.
- ✅ **Question JSON generation**: 30 questions로 `human_qa.json` (position 12 / time 7 / text 5 / binary 4 / duration 2).
  eval은 text를 건너뛰므로 **25 questions가 채점됨**. 나머지 seq는 caption이 없어 자동으로 건너뜀.
- ⛔ **평가 시작 → blocked**: 서버가 temperature를 무시(greedy)해서 eval.py의 retry 루프가 영원히 돌았음. 방향 결정 대기.
- `9fe023c` **npu_info 수정**: 추측했던 device 메서드가 실제로 존재하지 않아 전부 `(n/a)`로 돌아옴 → 이제
  `open_mesh_device`와 실제 MeshDevice API만 사용. MeshDevice에 DRAM 크기 API가 없어서 **SoC descriptor YAML**에서 읽음.
  ttnn의 loguru + C++ logs는 **fd redirection으로 억제**(`-v`로 표시). → **실측 스펙 확보(DRAM 12GiB 등)**.
- `4f81c2e` **중복 prefill 제거**: demo(`vision_demo.py`)는 `compile_prefill` + `inference_prefill`로 **일부러 두 번 prefill**함
  (compile vs. inference 시간을 따로 측정하려는 benchmarking 장치)이지만, 서버에서는 첫 요청 이후 cache가 warm이라
  두 번째는 순수 중복이었음. 첫 결과는 VLM의 안 쓰는 `masks`로, LLM은 **변수에 할당조차 안 됐음**.
  이제 VLM과 LLM 둘 다 prefill 1회. 측정: caption당 VisionBlock 2→**1**, ~76s→**~38s**.
- ✅ **seq0 captioning 완료**: 272 captions 생성(`captions_tt-qwen25-vl-3b_3_secs.json`). Checkpoint resume 실제로 검증(중단 후 이어서 → 완료 시 partial 자동 삭제).
- `8be8bc6` **Slowdown 근본 원인 규명 및 수정**: tt-metal의 **Inspector**(debug, 기본 ON)가 mesh workload마다 YAML `flush()` → caption당 ~5,000 synchronous disk I/O 호출, 단일 실행에서 153MB / 5.34M lines 누적 → 파일이 클수록 flush가 더 오래 block → vision block이 **19에서 2.1 it/s로 단조 감쇠**. `run_server.sh`에서 tt-metal 자체 switch로 비활성화(`TT_METAL_INSPECTOR=0`, `_RPC=0`) — **tt-metal source 변경 없음**. 측정: **~42 it/s로 안정 유지**, YAML 생성 안 됨. 앞선 오진 수정(`describe()`의 `ttnn.deallocate`)도 되돌림.
- `f9bc7bf` **서버 안정화**: shutdown 시 worker를 join해 `close_mesh_device()`를 보장하고, teardown 중 SIGINT/SIGTERM 무시(Ctrl-C 연타로 인한 wedging 방지).
- `804bcf6` **eval import**: `ChatNVIDIA`(NIM)를 lazy-import → NIM 패키지 없이도 eval 실행됨.
- `8caa7c9` **Captioner**: import fallback(안쪽·바깥쪽 경로 둘 다), `max_gen_len`(기본 256) 전송으로 문장 잘림 해결.
- `16a253e` **Captioning pipeline**: `--captioner {vila,tenstorrent}` 선택 + VILA lazy import, 프롬프트를 video→image로 중화, JSONL checkpoint/resume, early-exit을 "final JSON 존재" 기준으로 변경.
- `334c0b4` **ROS-free extractor**: `preprocess_coda_rosfree.py` — pose txt + cam0 PNG → .pkl.
- `c5cc220` **Docs**: PAPER_EVAL_RUNBOOK 추가; RUNBOOK에 langgraph/opencv pin과 shutdown 절차 추가.
- (ops) 원본 논문 데이터 확보: `data.csv` 다운로드; CODa seq0(42GB)에서 8213 .pkl 파일을 다운로드·압축 해제·추출.
- (ops) venv 의존성 정리: `langgraph==0.2.40`/`langsmith==0.1.147`, `numpy<2`/`opencv-python-headless<5` 설치(pin 충돌 회피).
- (diagnosis) wedged board 근본 원인 규명: `/dev/tenstorrent`가 아니라 **`/dev/bos`** 칩임. captioning 중 강제 종료가 원인이었음.

### 2026-07-13
- `05fff77` VILA `<video>` token 제거, per-frame captions. Smoke test 강화.
- `7d96cbc` LLM JSON-constrained (grammar) decoding.
- `165e846` NonAgent NPU end-to-end 강화.
- `8f9db90` end-to-end smoke test.
- `bb169e2` Unified NPU inference server (Llama+Qwen).
- `d29661c`/`ef0729d` NonAgent `tt/` backend + NPU captioner.

---

## 7. Document map

| Document | Contents |
|---|---|
| **[DEV_STATUS.md](DEV_STATUS.md)** (이 문서) | Status, issues, decisions, changelog (여기서 시작) |
| [TENSTORRENT_INTEGRATION.md](TENSTORRENT_INTEGRATION.md) | Architecture, design decisions, debugging log |
| [PAPER_EVAL_RUNBOOK.md](PAPER_EVAL_RUNBOOK.md) | 논문 NaVQA/CODa 평가 + 복구 전체 절차 |
| [../tt_server/README.md](../tt_server/README.md) | Server structure, requirements, env |
| [../tt_server/RUNBOOK.md](../tt_server/RUNBOOK.md) | Server startup, venv setup, troubleshooting, shutdown |
