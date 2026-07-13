# ReMEmbR × Tenstorrent NPU 통합 작업 정리

로봇 시공간 메모리 시스템 **ReMEmbR**를 Tenstorrent NPU 위에서 end-to-end로 돌리기 위한 통합 작업 기록.
VLM(캡셔너)과 LLM(질의응답)을 모두 NPU에서 구동하고, 그 사이를 Milvus 벡터 메모리로 잇는다.

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

---

## 5. 남은 이슈 / 다음 단계

### 3B LLM 품질
- 3B는 RAG에 "쓸 수는 있으나 경계선". 깨끗한 context를 주면 grounded하지만, context가 비거나 오염되면 `有有` 같은 degenerate(반복·외국어 토큰 누출)가 발생.
- greedy(temperature=0) + 반복 페널티 없음이 degenerate를 부추김.
- **선택지**
  1. **8B로 교체** — DRAM 제약으로 co-resident 불가 가능성 → 캡션/질의 단계를 시간분리(한 번에 한 모델만 로드)하는 폴백 구성.
  2. **3B 유지 + 튜닝** — 서버에 repetition penalty 추가, `TT_LLM_MAX_TOKENS` 축소.

### 실제 시퀀스 데이터 투입
- 캡셔너는 이미 윈도우 입력을 지원. 실제 파이프라인은 **"윈도우 → 캡션 1개 → 그 시점 time/position으로 insert"** 반복.
- 주의: 프레임당 vision token ≈ 64개, `TT_VLM_MAX_SEQ_LEN` 내에서 윈도우 크기 제한. 크게 잡으려면 seq len 상향.
- TODO: 영상 프레임 폴더/타임스탬프(rosbag 등)를 받아 **윈도우 슬라이딩 + 프레임별 time/position 부여** 로더 작성.

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
| `remembr/tt_server/tt_infer_server.py` | NPU 통합 추론 서버 (LLM+VLM, 포트 8100) |
| `remembr/captioners/tenstorrent_captioner.py` | Qwen2.5-VL 캡셔너 (HTTP 클라이언트) |
| `remembr/agents/non_agent.py` | Llama-3.2-3B `tt/` 백엔드 추가, JSON schema 제약 |
| `examples/tt_smoke_test.py` | end-to-end 스모크 테스트 |

*(tt-metal 원본은 건드리지 않고, 추가 코드는 remembr 쪽에 모음.)*
