# Tenstorrent NPU에서의 ReMEmbR

Tenstorrent 보드에서 ReMEmbR을 실행하는 데 필요한 모든 glue 코드는 **이
repo**에 있습니다. tt-metal SDK는 있는 그대로 사용되며(런타임에 `PYTHONPATH`로
import), tt-metal 소스 트리에 복사하거나 수정하는 것은 없습니다.

참고: [../docs/DEV_STATUS.md](../docs/DEV_STATUS.md) (프로젝트 상태), [RUNBOOK.md](RUNBOOK.md)
(설정 / 시작 / 종료), [../docs/PAPER_EVAL_RUNBOOK.md](../docs/PAPER_EVAL_RUNBOOK.md) (논문 평가),
[../docs/TENSTORRENT_INTEGRATION.md](../docs/TENSTORRENT_INTEGRATION.md) (아키텍처).

## 추가된 것들의 구성

```
remembr/
├── tt_server/
│   ├── tt_infer_server.py   # 기존 실행 경로를 보존하는 얇은 호환 entrypoint
│   ├── main.py              # 설정·runtime·HTTP app을 조립하는 composition root
│   ├── domain/              # lifecycle snapshot과 서버 오류
│   ├── application/         # chat/caption/mode 유스케이스와 RuntimePort
│   ├── presentation/http/   # FastAPI schema, route, HTTP 오류 변환
│   ├── infrastructure/
│   │   └── tenstorrent/     # 단일 worker, device lifecycle, LLM/VLM engine
│   ├── config/              # 환경변수와 mode catalog
│   ├── tests/               # NPU 없이 실행하는 contract/runtime 단위 테스트
│   ├── run_server.sh        # launcher (sets env, runs in tt-metal venv)
│   ├── npu_info.py          # hardware/spec dump (arch, DRAM, Tensix grid, L1)
│   ├── RUNBOOK.md           # start/venv/troubleshoot/shutdown
│   └── README.md            # this file
├── remembr/agents/non_agent.py                    # + 'tt/<model>' backend (ChatOpenAI -> server)
├── remembr/agents/remembr_agent.py                # lazy-import ChatNVIDIA (optional NIM backend)
├── remembr/captioners/tenstorrent_captioner.py    # Captioner -> POST /caption (max_gen_len, import fallback)
├── remembr/scripts/preprocess_coda_rosfree.py     # ROS-free CODa -> .pkl extractor
├── remembr/scripts/preprocess_captions.py         # + --captioner tenstorrent, checkpoint/resume
└── examples/tt_smoke_test.py                      # end-to-end: image -> caption -> memory -> query
```

> **HW note**: 이 머신은 **BOS chip**입니다(Tenstorrent 아님). Device node `/dev/bos/0`,
> kernel module `bos`. 누가 점유 중인지 확인: `fuser -v /dev/bos/0`, `lsmod | grep bos`.

## 아키텍처

```
 remembr (its own venv)                     tt_server/tt_infer_server.py (tt-metal venv, owns the NPU)
   NonAgent ── HTTP /v1/chat/completions ─────► Llama-3.2-3B  ┐  one process, device opened once,
   Captioner ─ HTTP /caption ─────────────────► Qwen2.5-VL-3B ┘  both models co-resident,
                                                                 all NPU work serialized by a queue
```

단일 프로세스가 보드를 소유하고(ttnn이 배타적으로 open) 두 모델을 모두
호스팅합니다. 단일 worker thread + queue가 모든 prefill/decode를 직렬화하므로
`/caption`과 `/v1/chat/completions` 요청을 동시에 보내도 안전합니다.

서버 내부 의존 방향은 `presentation → application → domain`이며,
`infrastructure/tenstorrent`가 application의 `RuntimePort`를 구현합니다. FastAPI route는
엔진이나 mutable device state를 직접 참조하지 않습니다. NPU의 open/load/infer/switch/close는
모두 non-daemon device worker가 소유하고, 외부에는 immutable runtime snapshot만 노출됩니다.

## 요구 사항

- **두 개의 venv.** 서버는 tt-metal의 `python_env`(ttnn, transformers)에서
  실행됩니다. remembr는 자체 venv(langchain, pymilvus, sentence-transformers)에서
  실행됩니다 — 버전이 충돌하기 때문에 HTTP 뒤로 분리되어 있습니다.
  - remembr venv pin(최신 langchain은 `ChatOllama`를 제거함):
    `langchain-core==0.2.40 langchain-community==0.2.16 langchain-openai==0.1.25`
    그리고 `pillow requests pymilvus langchain-huggingface==0.0.3 sentence-transformers==3.1.0`.
  - 서버 추가 의존성(tt-metal venv에 설치): `fastapi uvicorn[standard] lm-format-enforcer`.
- **HF 접근.** `meta-llama/Llama-3.2-3B-Instruct`는 gated이므로 `HF_TOKEN`을
  설정하세요(run_server.sh가 ~/.bashrc에서 읽음). `Qwen/Qwen2.5-VL-3B-Instruct`는 공개.
- 메모리 DB용 **Milvus** 실행(docker).

## 실행

```bash
# 1) start the NPU server (tt-metal venv)
bash remembr/tt_server/run_server.sh
#    wait for: {"ready":true,"llm":true,"vlm":true}  at http://localhost:8100/health

# 2) run remembr against it (remembr venv)
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py path/to/frame.jpg
```

`run_server.sh`는 tt-metal을 import하거나 NPU를 열기 전에 startup preflight를 수행합니다.
이미 `/health`가 응답하면 다음처럼 기존 상태를 출력하고 exit 0으로 종료하며, 새 서버를 띄우지 않습니다.

```text
[tt-server] startup skipped: server already ready at http://127.0.0.1:8100/health
[tt-server] health: {"ready":true,...}
```

동시에 실행된 두 launcher는 전역 `flock`으로 직렬화됩니다. health가 준비되지 않았더라도 기존
`tt_infer_server.py` process나 `/dev/bos/0` holder가 발견되면 안전하게 startup을 차단하고 exit 1로
원인을 출력합니다.

## HTTP API

인터랙티브 문서는 FastAPI가 **`http://localhost:8100/docs`**(Swagger UI),
`/redoc`, `/openapi.json`에서 제공합니다. 핸들러가 raw `dict`를 받으므로 생성된
schema에는 `{"type": "object"}`만 표시됩니다 — request/response 필드는 대신 여기에
문서화되어 있습니다. 세 엔드포인트 모두 caption 실행 중에 호출해도 안전합니다. NPU
작업은 worker queue로 직렬화되므로 요청은 그저 자기 차례를 기다립니다.

### `GET /health`

device worker와 각 engine의 준비 상태. 가볍고 NPU 작업 없음.

```jsonc
// 200
{
  "ready": true,
  "mode": "3b+3b",
  "switching": false,
  "llm": true,
  "vlm": true,
  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "llm_precision": "performance",
  "state": "ready",
  "error": null,
  "debug": {
    "decode": false,
    "sample": false,
  },
  "worker": {
    "busy": false,
    "queue_size": 0,
    "active_job": null,
    "active_elapsed_s": null,
  },
}
```

`ready`는 현재 모드의 모델이 전부 device에 로드된 후에만 true로 바뀝니다.
`switching`이 true인 동안은 `/caption`·`/v1/chat/completions`가 503을 반환합니다.
`debug`는 서버 시작 시점에 읽은 `TT_DEBUG_DECODE`/`TT_DEBUG_SAMPLE` 상태를 보고합니다. (런타임 토글은 제공하지 않습니다)
`worker`는 device worker의 현재 작업 상태입니다. `busy=true`이고 `active_elapsed_s`가 계속 증가하면 서바가 죽은 것이 아니라 NPU 작업 하나를 아직 처리 중이라는 뜻입니다. `queue_size`는 대기 중인 작업 수입니다.
`/health.worker`는 Python server가 응답 가능한 상태에서만 관측되는 best-effort 상태다.
TTNN/native 호출이 GIL을 잡고 장시간 블록되면 `/health` 자체가 응답하지 않을 수 있다.
이 경우 조용한 `/health`를 곧바로 서버 종료로 해석하지 말고, `py-spy dump --nonblocking` 등으로 worker thread가 `active+gil` 상태인지 확인한다. `queue_size`는 `Queue.qsize()` 기반 근사값이며 제어 로직에는 사용하지 않는다.

### `GET /memory`

device allocator 상태를 읽는 디버그 엔드포인트. NPU 작업을 새로 enqueue하지 않고,
반복 caption 요청이 메모리를 돌려주는지 확인할 때 쓴다.

```jsonc
// 200
{
  "dram": {
    "allocated": 123456,
    "free": 123456789,
    "largest_free": 123456789,
    "banks": 3
  },
  "l1": {
    "allocated": 1234,
    "free": 123456,
    "largest_free": 123456,
    "banks": 20
  }
}
```

같은 요청을 반복했는데 `allocated`가 계속 늘면 request-local buffer가 반환되지 않는 것이다.
`allocated`는 비슷한데 `largest_free`만 줄면 조각화로 봐야 한다. device가 아직 열리지 않은
상태에서는 `{}`를 반환할 수 있다.

### 모드 — 동시에 로드되는 모델 조합

보드 DRAM이 ~12GB라 조합 하나만 상주 가능합니다. **모드**는 그 조합에 이름을 붙인 것:

| 모드           | LLM          | VLM                                                              |
| -------------- | ------------ | ---------------------------------------------------------------- |
| `3b+3b` (기본) | Llama-3.2-3B | Qwen2.5-VL-3B                                                    |
| `caption-3b`   | —            | Qwen2.5-VL-3B                                                    |
| `llm-8b`       | Llama-3.1-8B | —                                                                |
| `caption-8b`   | —            | Qwen3-VL-8B (transformers 4.57 venv 필요, `TT_PYTHON_ENV=tf457`) |

**시작 시 선택**: `TT_MODE=caption-8b bash run_server.sh` — 잘못된 이름이면 device를 건드리기
전에 바로 실패합니다(조용히 멈추지 않음). 안 주면 `TT_ENABLE_LLM`/`TT_ENABLE_VLM` +
`TT_LLM_MODEL`/`TT_VLM_MODEL`로 기존 방식대로 뜹니다.

**실행 중 전환**: 다른 모드가 필요하면 요청이 자동으로 바꾸지 않습니다 — 반드시 `/mode`를
명시적으로 호출해야 합니다(로드에 수 분 걸리므로). 전환은 device를 완전히 반환(`synchronize
→ close_mesh_device → reopen → load`)한 뒤 다시 올리므로 자원 누수가 없습니다.

```jsonc
// GET /mode
{"mode": "3b+3b", "switching": false, "llm": "meta-llama/Llama-3.2-3B-Instruct",
 "vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "available": ["3b+3b", "caption-3b", "llm-8b", "caption-8b"]}

// POST /mode  {"mode": "caption-8b"}
// 200 (전환 완료 후 반환, 수 분 걸릴 수 있음)
{"mode": "caption-8b", "switched": true, ...}
// 400 unknown mode   503 already switching
```

현재 모드에 없는 엔드포인트를 호출하면 **409**(예: `caption-3b`에서 `/v1/chat/completions`) —
"모드를 바꾸라"는 안내 메시지와 함께. 조용히 블록하는 대신 즉시 명확한 에러를 냅니다.

### `POST /caption` — VLM (Qwen2.5-VL)

```jsonc
// request
{
  "images": ["<base64 image>", "..."],  // 필수, 1~5장. PIL이 읽을 수 있는 형식
                                        // (JPEG/PNG). 서버가 decode -> RGB 후
                                        // TT_VLM_IMAGE_LONG_SIDE/TT_VLM_IMAGE_FIT로 맞춤.
                                        // 여러 장은 한 장면의 프레임들로 취급.
  "prompt": "string",                   // 선택, 기본값은 서버의 DEFAULT_VLM_PROMPT.
                                        // 리터럴 "<video>" 토큰은 넣지 말 것: Qwen2.5-VL은
                                        // 이걸 안 쓰며 이미지와 함께 나오면 거부함.
  "max_gen_len": 128                    // 선택, 기본 128. 생성 최대 토큰; EOT에서 더 일찍
                                        // 멈춤. 128은 장황한 캡션을 문장 중간에서 자름
                                        // — captioner는 256을 보냄.
}
// 200
{"caption": "string"}
// 409 현재 모드에 VLM 없음(모드 전환 필요)   503 전환 중
// 400 이미지 없음/6장 이상/잘못된 base64 이미지
```

### `POST /v1/chat/completions` — LLM (Llama-3.2-3B), OpenAI 호환

이 서버가 구현한 OpenAI schema의 subset을 받으므로 `ChatOpenAI`
(`base_url=http://localhost:8100/v1`)가 수정 없이 동작합니다.

```jsonc
// request
{
  "messages": [{"role": "user", "content": "..."}],  // 하나의 prompt로 flatten됨
  "max_tokens": 256,        // 선택, 기본 256. max_seq_len//2로 clamp; 너무 긴 prompt는
                            // 에러 대신 잘림(head 유지).
  "temperature": 0.0,       // 선택, 기본 0. 0 = greedy(결정적);
                            // >0이면 sampling — 나쁜 답에 재시도하는 caller에겐 필수,
                            // greedy는 매번 동일한 답을 내므로.
  "model": "string",        // 선택, 그대로 echo만 됨
  "response_format": {      // 선택 -> grammar-constrained decoding
    "type": "json_object"   //   임의의 유효한 JSON
    // 또는:
    // "type": "json_schema",
    // "json_schema": {"name": "x", "schema": { /* JSON Schema */ }}
  }
}
// 200 (OpenAI shape)
{
  "id": "chatcmpl-<hex>", "object": "chat.completion", "created": 1234567890,
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}]
}
// 409 현재 모드에 LLM 없음(모드 전환 필요)   503 전환 중
```

지원하지 않는 OpenAI 필드(`stream`, `top_p`, `n`, `stop`, …)는 무시되며, usage/token
count는 보고되지 않습니다.

### 예제

```bash
curl -s localhost:8100/health
# {"ready":true,"llm":true,"vlm":true}

# LLM, JSON-constrained
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Give me an object with ok=true."}],
  "max_tokens":40, "temperature":0,
  "response_format":{"type":"json_object"}}'
# ... "content": "{\"ok\": true}" ...
```

`/caption`의 경우 **payload를 파일에 쓰고 `-d @file`을 사용하세요** — base64 이미지는
shell 인자에 들어가지 않습니다(1.7MB CODa frame은 base64로 약 2.3MB이며, ~2MB `ARG_MAX`를
초과하여 curl이 "Argument list too long"으로 죽습니다):

```bash
python3 - <<'EOF'
import base64, json
b64 = base64.b64encode(open("frame.png","rb").read()).decode()
json.dump({"images": [b64], "prompt": "Describe this image in one sentence.",
           "max_gen_len": 256}, open("/tmp/cap.json", "w"))
EOF
curl -s localhost:8100/caption -H 'Content-Type: application/json' -d @/tmp/cap.json
# {"caption":"The image shows an outdoor area with a concrete surface, a set of stairs ..."}
```

Python에서는 그냥 직접 post하면 됩니다 — `TenstorrentCaptioner`가 하는 방식입니다:

```python
import base64, requests
b64 = base64.b64encode(open("frame.png","rb").read()).decode()
r = requests.post("http://localhost:8100/caption",
                  json={"images": [b64], "prompt": "...", "max_gen_len": 256}, timeout=3600)
print(r.json()["caption"])
```

## Env vars (server)

| var                                         | default                            | note                                                                                                                                             |
| ------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TT_MODE`                                   | (unset)                            | 시작 시 로드할 모드 이름(`3b+3b`/`caption-3b`/`llm-8b`/`caption-8b`). 미지정 모드면 device 건드리기 전에 즉시 실패. 비우면 아래 개별 변수로 결정 |
| `TT_LLM_MODEL`                              | `meta-llama/Llama-3.2-3B-Instruct` | LLM (agent reasoning), `TT_MODE` 미지정 시만                                                                                                     |
| `TT_VLM_MODEL`                              | `Qwen/Qwen2.5-VL-3B-Instruct`      | VLM (captioning), `TT_MODE` 미지정 시만                                                                                                          |
| `TT_ENABLE_LLM` / `TT_ENABLE_VLM`           | `1` / `1`                          | `TT_MODE` 미지정 시만, 하나만 켜서 fallback 구성                                                                                                 |
| `TT_PYTHON_ENV`                             | (unset)                            | `tf457`로 주면 `python_env_tf457`(transformers 4.57.1) 사용 — `caption-8b`에 필요                                                                |
| `TT_SHUTDOWN_TIMEOUT`                       | `5`                                | Ctrl-C 시 uvicorn이 진행 중 요청을 기다리는 최대 초. 진행 중이던 실제 device 작업은 이 값과 무관하게 끝까지 실행된 뒤 안전하게 정리됨            |
| `TT_WORKER_SHUTDOWN_TIMEOUT`                | `180`                              | main thread가 device worker의 안전 종료를 기다리는 시간. worker는 non-daemon이라 이 시간이 지나도 interpreter가 강제로 버리지 않음               |
| `TT_SERVER_LOCK_FILE`                       | `/tmp/remembr-tt-npu-server.lock`  | launcher 간 NPU startup 전역 lock. 서버 process가 종료될 때까지 FD를 상속해 유지                                                                 |
| `TT_HEALTH_HOST` / `TT_HEALTH_TIMEOUT`      | `127.0.0.1` / `2`                  | startup preflight의 health 확인 host / timeout(초)                                                                                               |
| `TT_MAX_CAPTION_FRAMES`                     | `5`                                | 한 `/caption` 요청에서 허용할 최대 frame 수. 6장 이상은 검증되지 않은 vision attention deadlock을 막기 위해 거부                                 |
| `TT_LLM_MAX_SEQ_LEN` / `TT_VLM_MAX_SEQ_LEN` | `1024`                             | raise for longer context (uses more DRAM)                                                                                                        |
| `TT_TRACE_REGION_SIZE`                      | `15000000`                         | per-model trace buffer                                                                                                                           |
| `TT_PORT`                                   | `8100`                             |                                                                                                                                                  |
| `TT_VLM_IMAGE_LONG_SIDE`                    | `224`                              | `/caption` 입력 image의 긴 변 길이. 224가 기존 동작이며, 576 등은 실제 보드 검증 후 사용할 것                                                    |
| `TT_VLM_IMAGE_FIT`                          | `stretch`                          | `/caption` image 맞춤 방식. `stretch`=정사각형 강제(기존), `letterbox`=비율 유지+padding, `contain`=비율 유지 비정사각형                          |
| `TT_VLM_WARMUP_SHAPE`                       | (unset)                            | 실제 요청 shape를 미리 compile할 때 쓰는 `WIDTHxHEIGHT`. `contain`처럼 서버가 shape를 알 수 없을 때 지정                                         |
| `TT_VLM_WARMUP_LAST`                        | `0`                                | `1`이면 tt-metal 기본 224x224 warmup을 먼저 하고 요청 shape warmup을 나중에 실행                                                                  |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS`        | `300`                              | device polling이 무한 spin하는 상황을 TT timeout 예외로 바꾸기 위한 tt-metal timeout                                                              |
| `TT_DEBUG_DECODE`                           | `0`                                | `1`이면 LLM 생성 token을 decode된 text로 stdout에 출력. `/health.debug.decode`에서 현재 시작 상태 확인 가능                                      |
| `TT_DEBUG_SAMPLE`                           | `0`                                | `1`이면 constrained decoding의 sampling/masking diagnostics를 출력. 매우 verbose하므로 짧은 재현에만 사용. `/health.debug.sample`에서 확인 가능  |

remembr 쪽: `TT_LLM_BASE_URL`(default `http://localhost:8100/v1`),
`TT_VLM_URL`(default `http://localhost:8100/caption`),
`TT_VLM_TIMEOUT`(client timeout, default 900s — 느린 첫 compile을 위해 올리세요),
`TT_VLM_MAX_GEN_LEN`(max caption token, default 256 — 서버 default는 128이며 잘림).

## JSON-constrained decoding (format="json" equivalent)

`NonAgent`는 엄격한 JSON이 필요합니다. Ollama는 `format="json"`(grammar-constrained
decoding)으로 이를 보장했지만, NPU의 자유로운 greedy decoding은 그렇지 않아서 작은
3B는 code fence를 내보내거나 횡설수설하고 `NonAgent.query`는 무한 loop에
빠졌습니다(greedy → 동일한 retry).

서버는 이를 재현합니다. 요청이 OpenAI `response_format`을 설정하면 LLM은
`lm-format-enforcer`를 통해 **host-side grammar-constrained decoding**을 실행합니다:
`decode_forward_text(sampling_params=None)`이 raw host logits를 반환하고, 이는
argmax 전에 grammar가 허용하는 token id로 masking됩니다(`_masked_argmax`).

- `response_format={"type":"json_object"}` → 임의의 유효한 JSON.
- `response_format={"type":"json_schema","json_schema":{"schema":{...}}}` →
  schema로 제약. `NonAgent`의 `tt/` branch는 required key, `text`의 `maxLength`,
  `additionalProperties:false`가 포함된 schema를 보냅니다 — 이는 required key를
  강제하고, 자유 텍스트를 제한하며, reasoning 필드를 금지하여 3B가 token budget을
  넘어 횡설수설하지 못하게 합니다. 결과: Llama-3.2-3B에서 **100% parse 가능한 JSON,
  retry loop 없음, grounded된 답변, ~10-12s/query**.

비용: JSON mode는 매 step마다 full-vocab logits를 host로 읽으므로(on-device argmax
대비) 일반 decoding보다 느립니다 — 짧은 JSON 답변에는 괜찮습니다. Non-JSON 요청은
빠른 on-device greedy 경로를 유지합니다.

## 참고 / 알려진 한계

- **DRAM budget.** `npu_info.py`로 측정: **12 GiB = 4 GiB x 3 banks**
  (arch BLACKHOLE, Tensix 5x4 = 20 cores, L1 1.5MB/core). Llama-3.2-3B + Qwen-3B은
  `max_seq_len=1024`에서 함께 들어갑니다. 8B LLM은 **함께 들어가지 않으므로** —
  time-separated fallback(caption 단계에서 `TT_ENABLE_VLM=1 TT_ENABLE_LLM=0`, 이후
  교체) 또는 더 작은 모델을 사용하세요.
- **Inspector 꺼짐.** `run_server.sh`가 `TT_METAL_INSPECTOR=0` /
  `TT_METAL_INSPECTOR_RPC=0`을 설정합니다. tt-metal의 Inspector는 기본이 ON이며 mesh
  workload마다 YAML record를 flush하는데(caption당 ~5k회 synchronous disk flush),
  이것이 한 번의 실행에서 VLM을 19에서 2.1 it/s로 떨어뜨렸습니다. 디버깅할 때가
  아니면 꺼두세요.
- **Single prefill.** upstream 데모는 compile-vs-inference ms를 보고하기 위해
  prefill을 두 번(compile pass + timed pass) 실행하지만, 서버는 첫 요청 후 cache가
  warm하므로 하나만 유지합니다.
- **3B 답변 품질.** JSON 유효성은 이제 보장되지만, 답변 grounding은 여전히 모델에
  달려 있습니다. 3B는 대체로 grounded되어 있지만 가끔 prompt 예시를 복사합니다 —
  8B가 더 강력합니다(필요하면 DRAM fallback 사용).
- **종료.** 단일 `Ctrl-C`(또는 SIGTERM)로 중지하고 기다리세요 — 서버가 worker를
  join하여 `close_mesh_device()`가 실행되고 보드가 깨끗하게 해제되며, teardown 중
  추가 SIGINT/SIGTERM은 무시합니다(Ctrl-C를 연타해도 안전). **절대 `kill -9` 하지
  마세요** — SIGKILL은 teardown을 건너뛰어 보드를 먹통으로 만듭니다.
- **Caption마다 느려짐.** Inspector와 Tracy가 모두 꺼져 있는지 먼저 확인하세요. 과거의
  request별 명시적 tensor deallocation 접근은 효과가 없어 되돌렸으며, 현재 엔진은 tt-metal의
  lifecycle을 그대로 따릅니다. 그래도 느려지면 서버를 정상 종료 후 재시작하고 captioner
  checkpoint에서 재개하세요.
