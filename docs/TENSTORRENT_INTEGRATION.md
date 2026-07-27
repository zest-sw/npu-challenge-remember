# ReMEmbR × NPU 통합 — 아키텍처 및 설계

로봇 시공간 메모리 시스템 **ReMEmbR** 을 NPU 위에서 end-to-end로 구동하기 위해 필요했던 통합 작업의 기록.
VLM(captioner)과 LLM(질의응답) 모두 NPU에서 실행되며, Milvus 벡터 메모리가 이 둘을 이어준다.

> **진행 상태, 변경 이력, 알려진 이슈는 [DEV_STATUS.md](DEV_STATUS.md) 를 참고.** 이 문서는
> 아키텍처, 설계 결정, 디버깅 기록에 집중한다.
>
> **하드웨어 참고**: 이 머신은 **BOS chip** 이다(Tenstorrent 아님). tt-metal 호환 스택
> (`bos-metal`)을 통해 ttnn을 사용하기 때문에 로그에는 `BlackholeTTDevice` 가 찍히지만, 실제 device 노드는 **`/dev/bos/0`** 이고 커널
> 모듈은 **`bos`** 다. device 점유 여부는 `fuser -v /dev/bos/0` / `lsmod | grep bos` 로 확인한다. (`/dev/tenstorrent` 와
> `lspci 1e52` 는 이 머신에 존재하지 않는다.)
>
> **측정된 스펙** (`python tt_server/npu_info.py`): arch `BLACKHOLE`, device 1개, **DRAM 12 GiB**
> (4 GiB × 3 banks), **Tensix 5×4 = 20 cores** (harvesting `0x0`), worker core당 L1 1.5 MB.
> DRAM 12 GiB가 "3B+3B는 co-resident 가능 / 8B는 불가"의 근거다.

---

## 1. 목표

ReMEmbR 파이프라인:

```
Camera frame ──▶ [VLM caption] ──▶ [store in vector memory] ──▶ [question] ──▶ [search] ──▶ [LLM answer]
                  Qwen2.5-VL        Milvus                                      top-k        Llama-3.2-3B
                  (NPU)                                                                      (NPU)
```

- 원본 ReMEmbR은 captioner로 **VILA** 를, LLM으로 **GPT-4o / llama3.1 (8B급)** 을 사용했다.
- 우리는 둘 모두를 **Tenstorrent NPU에서 구동되는 3B 모델** 로 교체한다.
  - VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
  - LLM: `meta-llama/Llama-3.2-3B-Instruct`

---

## 2. 아키텍처

### 2.1 통합 추론 서버 (`remembr/tt_server/tt_infer_server.py`)

NPU 모델을 remembr 프로세스에 직접 로드하는 대신, **HTTP 서버로 분리** 한다.
이유: ttnn/tt-metal 스택이 remembr의 의존성 핀과 충돌하고, 두 프로세스가 하나의 device를 동시에 점유할 수 없기 때문이다.

- **포트 8100**, FastAPI.
- Endpoints
  - `POST /v1/chat/completions` — LLM (OpenAI 호환)
  - `POST /caption` — VLM (base64 이미지 리스트 + 프롬프트 → caption)
  - `GET /health` — 로드 상태
- **단일 worker thread + job queue** 설계: 모든 device 작업(open/load/prefill/decode)이 하나의 스레드로 직렬화된다. FastAPI 핸들러는 job을 큐에 넣고 Future에서 블록한다. → HTTP 요청이 아무리 많이 들어와도 device는 정확히 한 번에 하나씩만 처리한다.
- LLM과 VLM 엔진 모두 **시작 시 device에 co-resident로 로드** 된다(`TT_ENABLE_LLM`/`TT_ENABLE_VLM`, 둘 다 기본 on). `_ready` 이벤트가 로딩이 끝날 때까지 startup을 블록한다.
- 주요 env vars
  - `TT_LLM_MAX_SEQ_LEN` (기본 4096), `TT_VLM_MAX_SEQ_LEN` (기본 2048)
  - `TT_PORT` (8100), `TT_HOST` (0.0.0.0)
- **JSON에 대한 grammar-constrained decoding** 을 지원한다: OpenAI `response_format` 을 존중한다.
  - `json_object` → 임의의 유효한 JSON (Ollama `format="json"` 과 동등)
  - `json_schema` → 주어진 schema로 제약 (`TokenEnforcer` + `JsonSchemaParser`, host-side masking 후 argmax)
- VLM 입력 이미지는 서버에서 **224×224로 resize** 된다.

### 2.2 Captioner (`remembr/captioners/tenstorrent_captioner.py`)

`Captioner` 인터페이스를 구현한다. `caption(images: list[PIL.Image]) -> str`.

- base64 JPEG 리스트 + 프롬프트를 서버의 `/caption` 으로 POST한다.
- env: `TT_VLM_URL` (기본 `http://localhost:8100/caption`), `TT_VLM_TIMEOUT` (기본 900 s).
  - cold start에서의 첫 caption은 ttnn trace (prefill+decode) 컴파일 때문에 수 분이 걸릴 수 있으므로 timeout을 넉넉히 잡는다.
- **인터페이스는 frame 리스트(window) → 하나의 caption 이다.** window 전체를 한 번에 넘기는 것은 실제 sequence 데이터를 위한 원래 설계다.

### 2.3 LLM 에이전트 (`remembr/agents/non_agent.py`)

`NonAgent` — **단일 LLM 호출** 로 답을 생성하는 경량 에이전트, tool 없음.
`llm_type` 이 `tt/<model>` 형태이면 Tenstorrent 백엔드를 사용한다.

- `ChatOpenAI` 를 통해 서버의 `/v1/chat/completions` 를 호출한다 (`TT_LLM_BASE_URL`, 기본 `http://localhost:8100/v1`).
- **JSON schema로 grammar-constrained decoding을 강제한다**: `AgentOutput` 이 요구하는 모든 키(`type, text, binary, position, orientation, time, duration`)를 `required` 로 표시하고, `text` 는 `maxLength: 300`, `additionalProperties: false` 로 둔다.
  - 이는 작은 모델이 reasoning 필드에서 맴돌며 budget을 소진하는 것을 막고, parse 실패로 인한 무한 재시도를 방지한다.
- `TT_LLM_MAX_TOKENS` (기본 256) — 생성 토큰의 상한.

### 2.4 메모리 (`remembr/memory/milvus_memory.py`)

- `caption` + `text_embedding` (1024d, `mxbai-embed-large-v1`) + `position` (3d) + `time` (2d) 를 Milvus collection에 저장한다.
- 세 종류의 search: `search_by_text` (semantic), `search_by_position` ((x,y,z)), `search_by_time` (H:M:S). 각각이 결과를 `working_memory` 에 누적한다.
- `NonAgent` 은 `get_working_memory()` 를 context로 사용한다 → **질의 전에 반드시 `search_*` 를 호출해야** 하며, 그러지 않으면 context가 비어 있게 된다.

### 2.5 Smoke test (`examples/tt_smoke_test.py`)

전체 end-to-end 경로(caption → store → search → answer)를 한 번에 검증하는 스크립트.
사용법: `PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <images...>`

---

## 3. Agent vs NonAgent (설계 선택)

| | `ReMEmbRAgent` | `NonAgent` (우리가 쓰는 것) |
|---|---|---|
| 누가 search하나 | LLM이 능동적으로 tool을 고르고 반복 (LangGraph loop) | search를 외부에서 미리 수행; LLM은 search하지 않음 |
| search 방법 | text/position/time 중 자유 선택 | 넘겨받은 context로 고정 |
| LLM 호출 | 여러 번 (tool loop ≤3 + generate) | **1회** |
| tool calling | **필수** | 불필요 |
| 작은 모델 적합성 | 낮음 (3B는 tool call이 불안정) | **높음** |

→ NPU의 Llama-3.2-3B는 tool calling이 불안정하므로 **`NonAgent` 을 채택** 한다. 그 대가로 LLM이 스스로 position/time 특화 search를 수행할 수 없으므로, 우리 코드가 대신 search 결과를 채워 넣는다.

---

## 4. 디버깅 로그 — 발견한 문제와 해결

### 문제 ① 벡터 search가 엉뚱한 결과를 반환 (스케이트보드 탄 개 환각)

- **증상**: 파스타 이미지를 넣었는데 "개를 봤다"가 돌아옴.
- **원인**: `MilvusMemory.__init__` 이 `reset(drop_collection=False)` 로 생성 → **collection이 절대 drop되지 않음**. 이전 실행들의 오래된 caption(dog.jpg 등)이 `tt_smoke_test` collection에 계속 누적되었다. 일반적인 질문("What did you see?")은 특정 caption에 강하게 묶이지 않으므로, 오래된 항목들이 search 결과에 섞여 들어갔다.
- **해결**: smoke test 시작 시 `memory.reset(drop_collection=True)` 로 collection을 reset.
  - ⚠️ 실제 운용에서는 메모리가 누적되어야 하므로 drop이 금지된다 → 그 경우 올바른 접근은 **세션마다 다른 collection 이름을 쓰는 것** 이다.

### 문제 ② VLM이 caption을 거부 ("I'm sorry, but I can't assist with that.")

- **증상**: 이미지 2장을 한 번에 넘기면 거부가 발생. 1장이면 정상 동작.
- **조사**:
  - `<video>` + 이미지 2장 → **거부**
  - `<video>` + 이미지 1장 → 정상
  - `<video>` 없이 이미지 2장 → 거부는 없으나, 두 이미지가 **하나의 환각 scene으로 병합** 됨
- **원인**: 프롬프트가 리터럴 `<video>` 토큰으로 시작했다. 이는 원본 ReMEmbR의 **VILA 관례** 다 (VILA는 `<video>` 를 multi-frame placeholder special token으로 취급). **Qwen2.5-VL은 `<video>` 를 모른다** — 그 processor는 자동으로 `<|vision_start|>...<|vision_end|>` 를 삽입하므로, 리터럴 `<video>` 에 다중 이미지가 더해지면 학습된 형식과 충돌하여 거부를 유발한다.
- **해결**: `DEFAULT_PROMPT` 에서 `<video>` 를 제거하고 "These are sequential frames from a robot's camera..." 로 교체. → multi-frame 입력이 이제 정상 동작한다.

> **`<video>` 토큰이란?** VLM은 text token과 image embedding을 하나의 sequence로 이어붙이는데, `<video>` 토큰은 **image embedding이 들어갈 자리를 표시하는 placeholder** 다. VILA에서는 사용자가 이를 프롬프트에 직접 써넣지만, Qwen2.5-VL에서는 processor가 자동 삽입하므로 사용자가 써서는 안 된다.

### 문제 ③ 여러 frame이 하나의 caption과 하나의 메모리 항목으로 뭉침

- **증상**: 서로 무관한 데모 이미지(개+파스타)가 하나의 caption으로 병합되어 저장됨 → search할 만한 의미 있는 단위가 남지 않음.
- **해결**: smoke test를 **frame마다 별도 caption → frame마다 별도 `MemoryItem`** 을 만들도록 변경. 각 frame은 시차를 둔 `time` (각 +10s)과 `position` (x축을 따라)을 갖는다. 검색된 context도 육안 확인이 가능하도록 출력한다.
  - ⚠️ 이는 **smoke test 스크립트만** 바꾼 것이다. **captioner 인터페이스(window → 하나의 caption)는 그대로다.** 실제 sequence 영상에서 frame들은 같은 scene의 연속된 view이므로, window로 묶는 것이 올바른 처리다.

### 문제 ④ caption이 또 거부 ("I can't see the video")
- **증상**: CODa frame을 논문 프롬프트로 caption하니 영상을 볼 수 없다는 거부가 나옴.
- **원인**: 논문의 원본 프롬프트는 VILA **video** 용("the few seconds of the video")인데, 우리는 Qwen에 **single still frame** 을 넘긴다 (`--num-video-frames 1`) → 프롬프트/입력 불일치.
- **해결**: 원본 문구를 유지하되 video 표현만 중립화한다(`<video>` 제거, "the few seconds of the video" → "the image"). 그 외("Think step by step" 등)는 논문 그대로 유지 → 충실성을 보존하면서 거부를 해소했다.

### 문제 ⑤ caption 문장이 중간에 잘림
- **원인**: 서버의 `/caption` 은 `max_gen_len` 미지정 시 생성을 **128 tokens** 로 제한한다. step-by-step caption은 그 전에 EOT를 내지 못해 잘린다.
- **해결**: captioner가 이제 `max_gen_len` (기본 256, `TT_VLM_MAX_GEN_LEN`) 을 보낸다. 서버는 이미 이 파라미터를 존중하므로 **서버 코드는 변경 없음** — 클라이언트만 다시 실행하면 된다.

### 문제 ⑥ caption이 시간이 갈수록 느려짐 (21s → 80s) — **범인: tt-metal Inspector**
- **측정**: caption 길이, host RAM, swap은 모두 안정적이었다. 결정적 단서는 서버 로그의 vision block 속도가 **19에서 2.1 it/s로 단조 감소** (9배)한 것이었다.
- **원인**: tt-metal의 **Inspector** (디버그 기능)가 **기본 ON** 이다 (`rtoptions.hpp`: `InspectorSettings::enabled = true`). 그 logger가 **mesh workload마다 YAML을 append하고 매번 `fstream::flush()` 를 호출** 한다 (`logger.cpp`) → 우리 경우 caption당 대략 **5,000회의 동기 disk flush**. 한 번의 captioning 실행에서 `generated/inspector/mesh_workloads_log.yaml` 이 **153MB / 5.34M lines** 까지 커졌고, 파일이 커질수록 각 flush가 더 오래 블록하며 VLM을 함께 끌어내렸다.
- 이것은 메모리 문제가 **아니다**: inspector의 `unordered_map` 은 `erase` 로 정리되고, host RSS는 실제로 11.3GB에서 평평했다.
- **해결**: **tt-metal 소스를 건드리지 않고**, tt-metal이 이미 제공하는 스위치를 `run_server.sh` 에서 설정 —
  `TT_METAL_INSPECTOR=0`, `TT_METAL_INSPECTOR_RPC=0`.
- **측정 결과**: vision block이 **~42 it/s로 안정 유지** (감쇠가 사라짐), inspector YAML도 생성되지 않는다. 서버 로그의 `Inspector RPC 50051` 경고도 사라진다.
- **오진 기록**: 처음에는 device tensor가 쌓이는 것으로 보여 `describe()` 에 `ttnn.deallocate` 를 추가했으나, 이것은 **원인이 아니었고** (LLM decoding tensor를 대상으로 했음) 되돌렸다. `TT_METAL_DEVICE_PROFILER` 도 무관하다 (`'1'` 로 설정될 때만 활성화되고, 설정되지 않았으므로 no-op였다).

### 문제 ⑦ Ctrl-C (특히 연타)가 보드를 먹통으로 만듦
- **원인**: device를 닫는 `close_mesh_device()` 가 **daemon worker의 finally block** 에 있어서, 인터프리터 종료 시 실행되지 않는다. Ctrl-C 연타는 uvicorn을 강제 종료하고 teardown을 중단시킨다 → device가 활성 상태로 남아 **먹통** 이 된다 (재시작 시: `failed to initialize FW`).
- **해결**: shutdown 시 worker에 중단 신호를 보낸 뒤 `join()` 하여 `close_mesh_device()` 실행을 보장하고, 추가로 **teardown 구간 동안 SIGINT/SIGTERM을 무시** 한다 (키 연타에 안전). `-9` (SIGKILL)만 여전히 이를 우회하므로 **금지** 다.

### 문제 ⑫ 느려짐의 **두 번째** 원인 — tt-metal이 Tracy profiling 빌드였음
- **증상**: Inspector를 끈 뒤에도 vision block이 여전히 **84에서 50 it/s로** 하락. Inspector가 워낙 지배적이어서
  (19→2.1) 이것이 그 아래 가려져 있었다.
- **정량 확인**: **완전히 동일한 이미지** 로 12회 연속 요청 →
  ```
  RSS     : 5280 → 5649 MB   (+33MB per request, almost no variance)
  Latency : 55.8s → 67.8s    (+1.0s per request, monotonic increase)
  ```
  동일 입력인데도 선형 증가 = 캐싱이나 컴파일이 아니라, **요청마다 무언가가 누적** 되고 있음.
- **원인**: 이 tt-metal은 **`build_metal.sh --enable-profiler` 로 만든 Tracy 빌드** 다
  (`build -> build_Release_tracy`, `CMakeCache: ENABLE_TRACY:BOOL=ON`, `_ttnn.so` 안 705개의 Tracy symbol).
  `build_metal.sh` 의 기본값은 `enable_profiler="OFF"` 이므로, 누군가 의도적으로 `-p` 를 넘긴 것이다.
  Tracy를 `TRACY_ON_DEMAND` 없이 컴파일하면 **UI 연결 여부와 무관하게 프로그램 시작부터 모든 zone을 메모리에 누적** 한다.
  `ReadDeviceProfilerResults` 는 profiler가 비활성일 때 내부 블록을 건너뛰지만, 그 위의 **`ZoneScoped;` (Tracy 매크로)** 는 여전히
  `#if defined(TRACY_ENABLE)` 안에서 실행된다.
  게다가 `decode_forward` 는 decode된 토큰마다 (caption당 ~256회) `ttnn.ReadDeviceProfiler` 를 호출하고,
  vision block loop는 block마다 (32회) 호출한다 → caption당 수만 개의 zone이 누적된다.
- **env var로 끌 수 없는 이유**: `TT_METAL_DEVICE_PROFILER` 는 profiler의 *동작* 만 비활성화한다; Tracy는
  **컴파일 타임** 스위치이므로 런타임에 끌 수 없다.
- **해결**: profiler 없는 **기본 빌드** 로 교체 — `./build_metal.sh` (기본 `ENABLE_TRACY=OFF`).
  tt-metal 소스 변경 없음; 빌드 디렉터리가 분리되어 있고 (`build_Release`) symlink만 바뀌므로,
  **rollback은 한 줄이다: `ln -nsf build_Release_tracy build`**. 빌드는 24 core에서 **7분** 걸린다.
- **검증 (동일 조건 재측정)**: `_ttnn.so` 의 Tracy symbol이 **705 → 0**.
  ```
              Tracy build              Default build
  RSS growth  +33MB/request (linear)   +0~1MB (flat)
  Latency     55.8 → 67.8s             36.0 → 36.0s (constant)
  Vision      84 → 50 it/s (falling)   84~185 it/s (steady)
  ```
  **누수와 latency 증가가 완전히 사라졌다.** 272개 caption 추정: ~13시간 → **~2.9시간**.
- **교훈**: 성능 이상을 조사할 때는 **먼저 환경이 profiling 빌드임을 의심하라**. 두 원인
  (Inspector, Tracy) 모두 "디버그 도구가 기본으로 또는 의도적으로 켜져 있었다"로 귀결된다.

### 문제 ⑨ 요청당 prefill 2회 (vision encoder에서 2배 낭비)
- **발견**: 서버 로그가 caption당 `32 x Qwen2_5_VLVisionBlock` 을 **두 번** 출력했다.
- **원인**: 원본 데모(`qwen25_vl/demo/vision_demo.py`)는 prefill을 **의도적으로 두 번** 호출한다 —
  `profiler("compile_prefill")` 이 program cache를 warm하고, `profiler("inference_prefill")` 이 순수 inference
  시간을 측정한다. 이는 **compile vs. inference ms를 따로 보고하기 위한 벤치마킹 장치** 이고, 데모로서는 옳다.
  하지만 서버는 **첫 요청에서만 컴파일** 하고 그 뒤로는 warm하므로, 두 번째 pass는 순수 중복 forward였다.
  더 나쁜 것은, 첫 결과가 VLM에서는 사용되지 않는 변수 `masks` 로 들어갔고, LLM(`generate`)에서는 **변수에 할당조차 되지 않았다.**
- **해결**: VLM과 LLM 모두 **한 번** 의 pass(= 데모의 `inference_prefill`)만 남긴다. 측정: caption당 VisionBlock 2회 → 1회,
  ~76s → **~38s**.

### 문제 ⑩ optimal eval 경로가 첫 질문에서 크래시 → 무한 재시도
- **증상**: `AttributeError: 'str' object has no attribute 'time'` (`text_memory.py`).
- **원인**: eval.py의 `optimal+` 경로가 **이미 렌더링된 문자열** (`qa_instance['context']`)을
  `TextMemory` 에 넣는다. `get_working_memory()` 는 값이 str이면 문자열을 그대로 반환하고, `NonAgent` 은 그것을 다시
  `memory_to_string()` 에 넘기는데, 이 함수가 **문자열을 문자 단위로 순회** 한다 → `doc.time` 에서 터진다. 상류 버그.
- **해결**: `memory_to_string` 이 이제 문자열을 그대로 통과시킨다(= `get_working_memory` 의 문서화된 의도와 일관).

### 문제 ⑪ (미해결) 서버가 temperature를 무시함 → eval 재시도 loop가 끝나지 않음
- **증상**: 3B가 시간 질문에 `type='text', time=None` 으로 답하면, `evaluate_output` 이
  `abs(answer - None)` 에서 크래시 → `answer_squad_question` 의 `while True` 가 영원히 재시도한다.
- **원인**: 원본은 **`temperature=0.7`** (eval.py의 기본값)을 가정하므로, 재시도하면 다른 답이 나온다. 하지만
  우리 `/v1/chat/completions` 는 **temperature를 읽지조차 않고** `_masked_argmax` (greedy)로만 decode한다
  → 재시도가 **동일한 답** 을 낸다 → 무한 loop. (이는 `tt_server/README.md` 의 "greedy → identical retries"
  경고의 재발이다; 그때는 JSON 유효성 문제였고, 이번에는 **JSON은 유효하지만 내용이 틀린** 경우다.)
- **선택지**: (A) 서버가 temperature를 존중하게 함 (원래 의도 복원) / (B) `evaluate_output` 에 None guard 추가 /
  (C) JSON schema로 type별 필드를 강제. **방향 미정.**

### 문제 ⑧ "보드가 죽었다" 오진 (device 노드 이름)
- **원인**: 진단이 `/dev/tenstorrent` 와 `lspci 1e52` 를 확인 → 항상 "not found" → "버스에서 떨어졌다"는 오진 반복. 실제로는 **BOS chip** 이므로 노드가 `/dev/bos/0` 이다.
- **교훈**: device는 `fuser -v /dev/bos/0`, `lsmod | grep bos`, `/proc/<pid>/fd | grep bos` 로 확인하라.

---

## 5. 남은 이슈 / 다음 단계

### 3B LLM 품질
- 3B는 RAG용으로 "쓸 만하지만 아슬아슬" 하다. 깨끗한 context를 주면 근거에 충실하지만, context가 비었거나 오염되면 붕괴한다(반복, 외국어 token 누출) — 예: `有有` 같은 출력.
- repetition penalty 없는 greedy (temperature=0)가 이 붕괴를 부추긴다.
- **선택지**
  1. **8B로 전환** — DRAM 제약으로 co-residency가 불가능할 수 있음 → captioning 단계와 query 단계를 시간 분리하는 구성으로 후퇴 (한 번에 하나의 모델만 로드).
  2. **3B 유지 + 튜닝** — 서버에 repetition penalty 추가, `TT_LLM_MAX_TOKENS` 축소.

### 실제 sequence 데이터 (진행 중)
- 원본 논문 데이터(CODa)용 파이프라인 구축 완료: `preprocess_coda_rosfree.py` (ROS-free .pkl 추출) → `preprocess_captions.py --captioner tenstorrent` (checkpoint 기반 captioning) → `form_question_jsons.py` → `eval.py`. 상세는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md).
- **충실성 편차**: frame 6→1 (타협). Qwen도 다중 frame을 지원하지만, first-compile 비용과 안정성 때문에 1에서 시작한다. 올리려면 `--num-video-frames` 를 늘려라 (first compile을 기다릴 각오 필요).
- 진행 상태는 [DEV_STATUS](DEV_STATUS.md) 상태 보드 참고.

### 정리 작업
- 다수의 PyMilvus/LangChain deprecation 경고 (`connections.connect`, ORM 스타일 API 등). 기능상 무해하지만, 나중에 `MilvusClient` / `langchain-milvus` 로 마이그레이션하는 것을 권장한다.

---

## 6. 실행 방법 (요약)

```bash
# 1) Start the NPU inference server (separate terminal, tt-metal environment)
python remembr/tt_server/tt_infer_server.py      # port 8100, waits until LLM+VLM are loaded

# 2) end-to-end smoke test (remembr_venv)
cd ~/remembr
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py \
  /home/user/tt-metal/models/bos_model/qwen25_vl/demo/images/*.jpg

# change the question
TT_SMOKE_QUESTION="did you see a dog?" PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <img>
```

서버 재시작은 서버(`tt_server/`) 코드가 바뀔 때만 필요하다. captioner/agent/smoke test(remembr 쪽) 수정은 재시작이 필요 없다.

---

## 7. 추가/수정된 파일

| File | Role |
|---|---|
| `tt_server/tt_infer_server.py` | 통합 NPU 추론 서버 (LLM+VLM, port 8100, constrained JSON decoding, clean shutdown, tensor deallocation) |
| `tt_server/run_server.sh` | 서버 런처 (env 설정) |
| `remembr/captioners/tenstorrent_captioner.py` | Qwen2.5-VL captioner (HTTP client, `max_gen_len`, import fallback) |
| `remembr/agents/non_agent.py` | Llama-3.2-3B `tt/` 백엔드, JSON schema 제약 |
| `remembr/agents/remembr_agent.py` | `ChatNVIDIA` lazy import |
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa → .pkl 추출기 |
| `remembr/scripts/preprocess_captions.py` | Captioning (`--captioner tenstorrent`, checkpoint/resume, prompt 중립화) |
| `examples/tt_smoke_test.py` | end-to-end smoke test |

*(tt-metal 원본은 손대지 않았다; 추가 코드는 remembr 쪽에 모아 두었다.)*
