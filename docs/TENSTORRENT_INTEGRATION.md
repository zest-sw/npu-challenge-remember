# ReMEmbR × NPU 통합 — 아키텍처 & 설계

로봇 시공간 메모리 시스템 **ReMEmbR**를 NPU 위에서 end-to-end로 돌리기 위한 통합 작업 기록.
VLM(캡셔너)과 LLM(질의응답)을 모두 NPU에서 구동하고, 그 사이를 Milvus 벡터 메모리로 잇는다.

> **진행 상태·변경 로그·알려진 이슈는 [DEV_STATUS.md](DEV_STATUS.md)를 참조.** 이 문서는 아키텍처와
> 설계 결정, 디버깅 기록에 집중한다.
>
> **하드웨어 주의**: 이 장비는 **BOS 칩**이다(Tenstorrent 아님). tt-metal 호환 스택(`bos-metal`)으로
> ttnn을 쓰기에 로그엔 `BlackholeTTDevice`로 뜨지만, 실제 device 노드는 **`/dev/bos/0`**, 커널 모듈은
> **`bos`**다. device 점유 확인은 `fuser -v /dev/bos/0` / `lsmod | grep bos`. (`/dev/tenstorrent`·`lspci 1e52`는
> 이 장비에 없음.)
>
> **실측 스펙** (`python tt_server/npu_info.py`): arch `BLACKHOLE`, device 1개, **DRAM 12 GiB**
> (4 GiB × 3 뱅크), **Tensix 5×4 = 20 코어**(harvesting `0x0`), L1 1.5 MB/worker 코어.
> DRAM 12 GiB가 "3B+3B co-resident 가능 / 8B 불가"의 근거다.

---

## 1. 목표

ReMEmbR 파이프라인:

```
카메라 프레임 ──▶ [VLM 캡션] ──▶ [벡터 메모리 저장] ──▶ [질문] ──▶ [검색] ──▶ [LLM 답변]
                Qwen2.5-VL        Milvus                          top-k        Llama-3.2-3B
                (NPU)                                                          (NPU)
```

- 원본 ReMEmbR는 캡셔너로 **VILA**, LLM으로 **GPT-4o / llama3.1(8B급)** 을 썼다.
- 우리는 둘 다 **Tenstorrent NPU에 올린 3B 모델**로 대체한다.
  - VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
  - LLM: `meta-llama/Llama-3.2-3B-Instruct`

---

## 2. 아키텍처

### 2.1 통합 추론 서버 (`remembr/tt_server/tt_infer_server.py`)

NPU 모델을 remembr 프로세스에 직접 로드하지 않고 **HTTP 서버로 분리**했다.
이유: ttnn/tt-metal 스택이 remembr의 의존성 핀과 충돌하고, 한 디바이스를 두 프로세스가 동시에 점유할 수 없기 때문.

- **포트 8100**, FastAPI.
- 엔드포인트
  - `POST /v1/chat/completions` — LLM (OpenAI 호환)
  - `POST /caption` — VLM (이미지 base64 리스트 + 프롬프트 → 캡션)
  - `GET /health` — 로드 상태
- **단일 워커 스레드 + 잡 큐** 구조: 디바이스 작업(open/load/prefill/decode)은 전부 한 스레드에서 직렬화. FastAPI 핸들러는 잡을 큐에 넣고 Future를 blocking 대기. → 여러 HTTP 요청이 와도 디바이스는 한 번에 하나씩만.
- LLM·VLM 두 엔진을 **startup 시 co-resident로 device에 로드**(`TT_ENABLE_LLM`/`TT_ENABLE_VLM`, 기본 둘 다 on). `_ready` 이벤트로 로드 완료까지 startup 차단.
- 주요 env
  - `TT_LLM_MAX_SEQ_LEN` (기본 4096), `TT_VLM_MAX_SEQ_LEN` (기본 2048)
  - `TT_PORT` (8100), `TT_HOST` (0.0.0.0)
- **JSON 문법 제약 디코딩(grammar-constrained decoding)** 지원: OpenAI `response_format`을 존중.
  - `json_object` → 임의의 유효 JSON (Ollama `format="json"` 등가)
  - `json_schema` → 주어진 스키마로 제약 (`TokenEnforcer` + `JsonSchemaParser`, host-side 마스크 후 argmax)
- VLM 입력 이미지는 서버에서 **224×224로 리사이즈**.

### 2.2 캡셔너 (`remembr/captioners/tenstorrent_captioner.py`)

`Captioner` 인터페이스 구현. `caption(images: list[PIL.Image]) -> str`.

- 서버 `/caption`으로 base64 JPEG 리스트 + 프롬프트를 POST.
- env: `TT_VLM_URL` (기본 `http://localhost:8100/caption`), `TT_VLM_TIMEOUT` (기본 900초).
  - 콜드 스타트 첫 캡션은 ttnn trace(prefill+decode) 컴파일로 수 분 걸릴 수 있어 타임아웃을 크게 잡음.
- **인터페이스는 프레임 리스트(윈도우) → 캡션 1개**. 실제 시퀀스 데이터는 윈도우를 통째로 넘기는 것이 원래 설계.

### 2.3 LLM 에이전트 (`remembr/agents/non_agent.py`)

`NonAgent` — 도구 없이 **단일 LLM 호출**로 답을 생성하는 경량 에이전트.
`llm_type`이 `tt/<model>` 형식이면 Tenstorrent 백엔드 사용.

- `ChatOpenAI`로 서버 `/v1/chat/completions` 호출 (`TT_LLM_BASE_URL`, 기본 `http://localhost:8100/v1`).
- **JSON schema로 grammar-constrained decoding 강제**: `AgentOutput`이 요구하는 모든 키(`type, text, binary, position, orientation, time, duration`)를 `required`로 지정, `text`는 `maxLength: 300`, `additionalProperties: false`.
  - 소형 모델이 reasoning 필드에서 헛돌아 예산을 태우는 것을 막고, 파싱 실패로 인한 무한 retry를 방지.
- `TT_LLM_MAX_TOKENS` (기본 256) — 생성 토큰 상한.

### 2.4 메모리 (`remembr/memory/milvus_memory.py`)

- Milvus 컬렉션에 `caption` + `text_embedding`(1024d, `mxbai-embed-large-v1`) + `position`(3d) + `time`(2d) 저장.
- 검색 3종: `search_by_text`(의미), `search_by_position`((x,y,z)), `search_by_time`(H:M:S). 각각 `working_memory`에 결과를 누적.
- `NonAgent`는 `get_working_memory()`를 context로 사용 → **질의 전에 `search_*`를 먼저 호출**해야 context가 채워짐.

### 2.5 스모크 테스트 (`examples/tt_smoke_test.py`)

end-to-end 경로(캡션→저장→검색→답변)를 한 번에 검증하는 스크립트.
사용법: `PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <이미지들...>`

---

## 3. Agent vs NonAgent (설계 선택)

| | `ReMEmbRAgent` | `NonAgent` (우리가 사용) |
|---|---|---|
| 검색 주체 | LLM이 능동적으로 도구 선택·반복 (LangGraph 루프) | 외부에서 미리 검색, LLM은 안 함 |
| 검색 방식 | text/position/time 3종 자유 선택 | 넘겨받은 context 고정 |
| LLM 호출 | 다회 (도구 루프 ≤3회 + generate) | **1회** |
| tool calling | **필수** | 불필요 |
| 소형 모델 적합성 | 낮음 (3B는 tool call 불안정) | **높음** |

→ NPU의 Llama-3.2-3B는 tool calling이 불안정하므로 **`NonAgent` 채택**. 대신 위치/시간 특화 검색을 LLM이 스스로 못 하므로 검색은 우리 코드가 채워준다.

---

## 4. 디버깅 로그 — 발견한 문제와 수정

### 문제 ① 벡터 검색이 엉뚱한 결과 (스케이트보드 탄 개 환각)

- **증상**: pasta 이미지를 넣었는데 "개를 봤다"고 답변.
- **원인**: `MilvusMemory.__init__`이 `reset(drop_collection=False)`로 생성 → **컬렉션을 절대 drop하지 않음**. 이전 실행들의 옛 캡션(dog.jpg 등)이 `tt_smoke_test` 컬렉션에 계속 누적. 일반적 질문("What did you see?")은 특정 캡션과 딱 안 붙어 옛 항목이 섞여 검색됨.
- **수정**: 스모크 테스트 시작 시 `memory.reset(drop_collection=True)`로 컬렉션 초기화.
  - ⚠️ 실제 운용에선 메모리를 누적해야 하므로 drop 금지 → 그땐 **세션마다 컬렉션 이름을 다르게** 주는 방식이 맞음.

### 문제 ② VLM이 캡션을 거부 ("I'm sorry, but I can't assist with that.")

- **증상**: 이미지 2장을 한 번에 넣으니 거부 응답. 1장일 땐 정상.
- **조사 결과**:
  - `<video>` + 2장 → **거부**
  - `<video>` + 1장 → 정상
  - `<video>` 없이 2장 → 거부는 안 하나 두 이미지를 **한 장면으로 합쳐 환각**
- **원인**: 프롬프트가 리터럴 `<video>` 토큰으로 시작. 이건 원본 ReMEmbR의 **VILA용 관습**(VILA는 `<video>`를 멀티프레임 placeholder 특수 토큰으로 처리). **Qwen2.5-VL은 `<video>`를 모름** — processor가 `<|vision_start|>...<|vision_end|>`를 자동 삽입하는 구조라, 리터럴 `<video>` + 다중 이미지가 학습 형식과 충돌해 거부.
- **수정**: `DEFAULT_PROMPT`에서 `<video>` 제거, "These are sequential frames from a robot's camera..." 로 교체. → 다중 프레임도 정상 처리.

> **`<video>` 토큰이란?** VLM은 텍스트 토큰과 이미지 임베딩을 한 시퀀스로 이어붙이는데, 이미지 임베딩이 들어갈 **위치를 표시하는 자리표시 토큰**. VILA는 사용자가 프롬프트에 직접 쓰지만, Qwen2.5-VL은 processor가 자동으로 넣으므로 사용자가 쓰면 안 된다.

### 문제 ③ 여러 프레임을 캡션 1개·메모리 1개로 뭉뚱그림

- **증상**: 서로 무관한 데모 이미지(dog+pasta)를 한 캡션으로 합쳐 저장 → 검색할 의미 단위가 없음.
- **수정**: 스모크 테스트를 **프레임별 개별 캡션 → 개별 `MemoryItem` 저장**으로 변경. 각 프레임에 staggered `time`(+10s씩)과 `position`(x축) 부여. 검색된 context도 출력해 눈으로 확인 가능.
  - ⚠️ 이는 **스모크 테스트 스크립트만** 바뀐 것. **캡셔너 인터페이스(윈도우 → 캡션 1개)는 그대로** 유지. 실제 시퀀스 영상은 같은 장면의 연속 프레임이므로 윈도우로 묶는 것이 맞다.

### 문제 ④ 캡션이 다시 거부 ("I can't see the video")
- **증상**: 논문 프롬프트로 CODa 프레임 캡션 시 "비디오를 못 본다"며 거부.
- **원인**: 논문 원문 프롬프트가 VILA **비디오**용("the few seconds of the video")인데, 우리는 Qwen에 **정지 프레임 1장**(`--num-video-frames 1`)을 줌 → 프롬프트-입력 불일치.
- **수정**: 원문을 그대로 쓰되 video 문구만 중화(`<video>` 제거, "the few seconds of the video" → "the image"). "Think step by step" 등 나머지는 논문 그대로 → 충실도 유지 + 거부 해소.

### 문제 ⑤ 캡션 문장이 중간에 잘림
- **원인**: 서버 `/caption`이 `max_gen_len` 미지정 시 **128 토큰**으로 상한. step-by-step 캡션이 그 전에 EOT를 못 내 잘림.
- **수정**: 캡셔너가 `max_gen_len`(기본 256, `TT_VLM_MAX_GEN_LEN`) 전송. 서버는 이미 이 파라미터를 존중하므로 **서버 코드 무변경**, 클라이언트 재실행으로 적용.

### 문제 ⑥ 캡션이 갈수록 느려짐 (21s → 80s) — **범인: tt-metal Inspector**
- **측정**: 캡션 길이·호스트 RAM·스왑 모두 안정. 서버 로그의 비전 블록 속도가 **19 → 2.1 it/s로 단조 하락**(9배)하는 것이 결정적 단서였다.
- **원인**: tt-metal의 **Inspector**(디버그 기능)가 **기본 ON**이다(`rtoptions.hpp`: `InspectorSettings::enabled = true`). 그 logger가 **mesh workload마다 YAML을 append하고 매번 `fstream::flush()`** 를 호출한다(`logger.cpp`) → 여기선 **캡션당 약 5,000회 동기 디스크 flush**. 한 번의 캡셔닝 run에서 `generated/inspector/mesh_workloads_log.yaml`이 **153MB / 534만 라인**까지 커졌고, 파일이 커질수록 flush가 점점 blocking되어 VLM이 끌려 내려갔다.
  - 메모리 문제가 **아니다**: inspector의 `unordered_map`은 `erase`로 정리되며, 실제로 호스트 RSS는 11.3GB로 평평했다.
- **수정**: **tt-metal 소스는 건드리지 않고**, tt-metal이 제공하는 스위치를 `run_server.sh`에서 설정 —
  `TT_METAL_INSPECTOR=0`, `TT_METAL_INSPECTOR_RPC=0`.
- **실측 결과**: 비전 블록이 **~42 it/s로 안정 유지**(하락 소멸), inspector YAML 미생성. 서버 로그의 `Inspector RPC 50051` 경고도 함께 사라진다.
- **오진 기록**: 처음엔 device 텐서 누적으로 보고 `describe()`에 `ttnn.deallocate`를 추가했으나 **원인이 아니었고**(LLM 디코딩 텐서를 겨냥했음) 되돌렸다. `TT_METAL_DEVICE_PROFILER`도 무관(`'1'`일 때만 활성인데 미설정이라 no-op).

### 문제 ⑦ Ctrl-C(특히 연타)로 보드가 wedged
- **원인**: device를 닫는 `close_mesh_device()`가 **daemon 워커의 finally**라 인터프리터 종료 시 실행 안 됨. Ctrl-C 연타는 uvicorn을 강제종료시켜 teardown을 중단 → device 활성 상태로 남아 **wedged**(재기동 시 `failed to initialize FW`).
- **수정**: 종료 시 워커에 정지 신호 후 `join()`으로 `close_mesh_device()` 보장 + **teardown 구간 SIGINT/SIGTERM 무시**(연타 안전). `-9`(SIGKILL)만은 여전히 우회하므로 **금지**.

### 문제 ⑫ slowdown의 **두 번째** 원인 — tt-metal이 Tracy 프로파일링 빌드
- **증상**: Inspector를 끈 뒤에도 비전 블록이 **84 → 50 it/s**로 계속 하락. Inspector가 워낙 지배적이라
  (19→2.1) 그 아래 숨어 있었다.
- **정량 확증**: **완전히 동일한 이미지**로 12회 연속 요청 →
  ```
  RSS : 5280 → 5649 MB   (요청당 +33MB, 편차 거의 없음)
  지연: 55.8s → 67.8s    (요청당 +1.0초 단조 증가)
  ```
  입력이 같은데 선형 증가 = 캐시/컴파일이 아니라 **요청마다 쌓이는 구조**.
- **원인**: 이 tt-metal은 **`build_metal.sh --enable-profiler`로 빌드된 Tracy 빌드**다
  (`build -> build_Release_tracy`, `CMakeCache: ENABLE_TRACY:BOOL=ON`, `_ttnn.so`에 Tracy 심볼 705개).
  `build_metal.sh`의 기본값은 `enable_profiler="OFF"`이므로 누군가 의도적으로 `-p`를 준 것.
  Tracy는 `TRACY_ON_DEMAND` 없이 컴파일되면 **UI 연결 여부와 무관하게 프로그램 시작부터 모든 zone을
  메모리에 쌓는다.** `ReadDeviceProfilerResults`는 profiler 비활성 시 내부 블록을 건너뛰지만
  그 위의 **`ZoneScoped;`(Tracy 매크로)** 는 `#if defined(TRACY_ENABLE)` 안에서 그대로 실행된다.
  게다가 `decode_forward`가 **디코딩 토큰마다**(캡션당 ~256회), 비전 블록 루프가 **블록마다**(32회)
  `ttnn.ReadDeviceProfiler`를 호출한다 → 캡션당 수만 zone 누적.
- **왜 env로 못 끄나**: `TT_METAL_DEVICE_PROFILER`는 profiler *동작*만 끌 뿐, Tracy는 **컴파일 타임**
  스위치라 런타임에 못 끈다.
- **수정**: 프로파일러 없는 **기본 빌드**로 교체 — `./build_metal.sh`(기본 `ENABLE_TRACY=OFF`).
  tt-metal 소스 무수정, 빌드 디렉토리가 분리되고(`build_Release`) 심볼릭 링크만 바뀌므로
  **롤백은 `ln -nsf build_Release_tracy build` 한 줄**.
- **교훈**: 성능 이상을 볼 때 **환경이 프로파일링 빌드인지** 먼저 의심할 것. 두 원인(Inspector, Tracy)
  모두 "디버그 도구가 기본/의도적으로 켜져 있었다"였다.

### 문제 ⑨ 요청당 prefill 2회 (비전 인코더 2배 낭비)
- **발견**: 서버 로그에 `32 x Qwen2_5_VLVisionBlock`이 캡션 1개당 **2번** 찍혔다.
- **원인**: 원본 데모(`qwen25_vl/demo/vision_demo.py`)가 **의도적으로** prefill을 2번 호출한다 —
  `profiler("compile_prefill")`로 program cache를 데우고, `profiler("inference_prefill")`로 순수 추론
  시간을 잰다. **컴파일 vs 추론 ms를 분리 보고하려는 벤치마크 장치**이며 데모로선 올바르다.
  그러나 서버는 **첫 요청에서만 컴파일**되고 이후엔 전부 warm이라, 2회차는 순수 중복 forward였다.
  게다가 첫 결과는 VLM에선 미사용 변수 `masks`에, LLM(`generate`)에선 **변수에 받지도 않았다**.
- **수정**: VLM·LLM 모두 **1회**(= 데모의 `inference_prefill`)만 유지. 실측: 캡션당 VisionBlock 2회→1회,
  ~76s → **~38s**.

### 문제 ⑩ optimal 평가 경로가 첫 질문에서 크래시 → 무한 재시도
- **증상**: `AttributeError: 'str' object has no attribute 'time'` (`text_memory.py`).
- **원인**: eval.py의 `optimal+` 경로는 `TextMemory`에 **이미 렌더링된 문자열**(`qa_instance['context']`)을
  넣는다. `get_working_memory()`는 str이면 문자열을 그대로 반환하는데, `NonAgent`가 그걸 다시
  `memory_to_string()`에 넘기면서 **문자열을 문자 단위로 순회** → `doc.time`에서 폭발. upstream 버그.
- **수정**: `memory_to_string`이 이미 문자열이면 그대로 통과(= `get_working_memory`의 주석 의도와 일치).

### 문제 ⑪ (미해결) 서버가 temperature를 무시 → eval 재시도 루프가 무한
- **증상**: 3B가 time 질문에 `type='text', time=None`으로 답하면 `evaluate_output`이
  `abs(answer - None)`으로 크래시 → `answer_squad_question`의 `while True`가 영원히 재시도.
- **원인**: 원본은 **`temperature=0.7`**(eval.py 기본값)이라 재시도하면 다른 답이 나온다는 전제인데,
  우리 `/v1/chat/completions`는 **temperature를 읽지도 않고** `_masked_argmax`(greedy)로만 디코딩한다
  → 재시도해도 **동일한 답** → 무한 루프. (`tt_server/README.md`가 경고한 "greedy → identical retries"의
  재발이며, 그때는 JSON 유효성 문제였고 이번은 **JSON은 유효하나 내용이 틀린** 경우.)
- **선택지**: (A) 서버가 temperature 존중(원본 의도 복원) / (B) `evaluate_output`에 None 가드 /
  (C) JSON schema로 type별 필드 강제. **방향 미정.**

### 문제 ⑧ "보드 죽음" 오진 (device 노드 이름)
- **원인**: 진단 시 `/dev/tenstorrent`·`lspci 1e52`로 확인 → 계속 "없음" → "버스 이탈" 오진 반복. 실제는 **BOS 칩**이라 노드가 `/dev/bos/0`.
- **교훈**: device 확인은 `fuser -v /dev/bos/0`, `lsmod | grep bos`, `/proc/<pid>/fd | grep bos`.

---

## 5. 남은 이슈 / 다음 단계

### 3B LLM 품질
- 3B는 RAG에 "쓸 수는 있으나 경계선". 깨끗한 context를 주면 grounded하지만, context가 비거나 오염되면 `有有` 같은 degenerate(반복·외국어 토큰 누출)가 발생.
- greedy(temperature=0) + 반복 페널티 없음이 degenerate를 부추김.
- **선택지**
  1. **8B로 교체** — DRAM 제약으로 co-resident 불가 가능성 → 캡션/질의 단계를 시간분리(한 번에 한 모델만 로드)하는 폴백 구성.
  2. **3B 유지 + 튜닝** — 서버에 repetition penalty 추가, `TT_LLM_MAX_TOKENS` 축소.

### 실제 시퀀스 데이터 (진행됨)
- 원 논문 데이터(CODa)용 파이프라인이 구축됨: `preprocess_coda_rosfree.py`(ROS-free .pkl 추출) → `preprocess_captions.py --captioner tenstorrent`(체크포인트 캡셔닝) → `form_question_jsons.py` → `eval.py`. 상세는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md).
- **충실도 편차**: 프레임 6→1(절충). 다중 프레임은 Qwen도 지원하나 첫 컴파일 비용/안정성 때문에 1로 시작. 올리려면 `--num-video-frames` 상향(첫 컴파일 인내 필요).
- 진행 상태는 [DEV_STATUS](DEV_STATUS.md) 상태 보드 참조.

### 정리성
- PyMilvus/LangChain deprecation 경고 다수 (`connections.connect`, ORM 스타일 API 등). 기능엔 무해하나 추후 `MilvusClient` / `langchain-milvus`로 마이그레이션 권장.

---

## 6. 실행 방법 (요약)

```bash
# 1) NPU 추론 서버 기동 (별도 터미널, tt-metal 환경)
python remembr/tt_server/tt_infer_server.py      # 포트 8100, LLM+VLM 로드까지 대기

# 2) end-to-end 스모크 테스트 (remembr_venv)
cd ~/remembr
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py \
  /home/user/tt-metal/models/bos_model/qwen25_vl/demo/images/*.jpg

# 질문 바꾸기
TT_SMOKE_QUESTION="did you see a dog?" PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <img>
```

서버 재기동이 필요한 건 서버(`tt_server/`) 코드 변경 시뿐. 캡셔너/에이전트/스모크 테스트(remembr 측) 수정은 재기동 불필요.

---

## 7. 추가/수정한 파일

| 파일 | 역할 |
|---|---|
| `tt_server/tt_infer_server.py` | NPU 통합 추론 서버 (LLM+VLM, 포트 8100, JSON 제약 디코딩, clean shutdown, 텐서 해제) |
| `tt_server/run_server.sh` | 서버 런처 (env 세팅) |
| `remembr/captioners/tenstorrent_captioner.py` | Qwen2.5-VL 캡셔너 (HTTP 클라이언트, `max_gen_len`, import fallback) |
| `remembr/agents/non_agent.py` | Llama-3.2-3B `tt/` 백엔드, JSON schema 제약 |
| `remembr/agents/remembr_agent.py` | `ChatNVIDIA` lazy import |
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa → .pkl 추출기 |
| `remembr/scripts/preprocess_captions.py` | 캡셔닝(`--captioner tenstorrent`, 체크포인트/resume, 프롬프트 중화) |
| `examples/tt_smoke_test.py` | end-to-end 스모크 테스트 |

*(tt-metal 원본은 건드리지 않고, 추가 코드는 remembr 쪽에 모음.)*
