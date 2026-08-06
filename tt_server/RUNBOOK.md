# ReMEmbR on Tenstorrent NPU — Runbook (처음부터)

이 문서는 VLM(captioning) = Qwen2.5-VL-3B 와 LLM(reasoning) = Llama-3.2-3B 를
**단일 Tenstorrent 보드**에서 기동하고, remembr 코어 파이프라인
(`image → caption → memory insert → question answering`)을 end-to-end로 실행하는 절차이다.

> ⚠️ **서버 lifecycle 소유권**: NPU 서버 기동·종료·재시작은 사람 사용자가 지정된 tmux
> 세션에서 직접 수행한다. AI 에이전트나 자동화는 `run_server.sh` 실행, `tmux send-keys`,
> signal/kill, background server 기동을 하지 않는다. 에이전트는 `/health`, process, tmux 로그를
> 읽기 전용으로 확인할 수 있으며, 서버를 올리거나 내려야 하면 사용자에게 요청해야 한다.

---

## 0. Architecture at a glance

```
 remembr (remembr/remembr_venv)              tt_infer_server.py (tt-metal python_env, owns the board)
   NonAgent ── HTTP /v1/chat/completions ─────► Llama-3.2-3B  ┐  1 process, device opened once,
   Captioner ─ HTTP /caption ─────────────────► Qwen2.5-VL-3B ┘  both models co-resident,
                                                                 single worker + queue serializes all NPU execution
```

- **venv 두 개**를 사용한다. 서버는 tt-metal의 `python_env`(ttnn/transformers)에서 돌고, remembr는
  자체 전용 venv(langchain/pymilvus)에서 돈다. 버전이 충돌하므로 HTTP로 분리한다.
- NPU는 한 번에 그래프 하나만 실행 → captioning과 LLM이 queue에 의해 자동으로 직렬화된다
  (동시 요청도 안전하다).

---

## 1. First-time setup (once)

### 1-0. ⚠️ tt-metal이 **profiler 빌드가 아닌지** 확인 (성능에 치명적)

```bash
readlink /home/user/tt-metal/build          # must be build_Release (build_Release_tracy ✗)
grep ENABLE_TRACY /home/user/tt-metal/build/CMakeCache.txt   # must be ENABLE_TRACY:BOOL=OFF
nm -D --defined-only /home/user/tt-metal/ttnn/ttnn/_ttnn.so | grep -ci tracy   # must be 0
```

`ENABLE_TRACY=ON`이면(즉 `build_metal.sh -p/--enable-profiler`로 빌드한 경우), **요청을 거듭할수록 추론이 느려진다.**
Tracy가 `TRACY_ON_DEMAND` 없이 컴파일되면, Tracy UI 연결 여부와 무관하게 **모든 zone을 메모리에 계속 누적**하며,
`decode_forward`는 디코딩되는 토큰마다(캡션당 ~256회) `ReadDeviceProfiler`를 호출한다.
측정 결과(같은 이미지, 12회): **요청당 RSS +33MB, latency가 55.8 → 67.8 s로 선형 증가**. 캡션 272개면
~2.9h에서 ~13h로 늘어난다.
**컴파일 타임 스위치라서 env로는 끌 수 없다.** profiler 빌드라면 기본값으로 재빌드하라:

```bash
cd /home/user/tt-metal && source env_set.sh
./build_metal.sh            # defaults to ENABLE_TRACY=OFF. ~7 min on 24 cores
#  -> creates a fresh build_Release and repoints the `build` symlink at it.
#     The existing build_Release_tracy is left intact, so rolling back is one line:
#     ln -nsf build_Release_tracy build
```
재빌드 후에는 서버를 재시작해야 반영된다. 확인 방법: 같은 이미지를 여러 번 caption 해보라 —
**latency가 평평하게 유지되고 서버 RSS가 늘지 않으면** 정상이다(Tracy 빌드에서는 둘 다 단조 증가한다).

### 1-1. HF token (gated model)
`meta-llama/Llama-3.2-3B-Instruct`는 gated 모델이다. 토큰이 `~/.bashrc`에
`export HF_TOKEN=...` 형태로 있어야 한다. `run_server.sh`는 비대화형 셸에서도 그 줄을 직접 읽는다.
(Qwen2.5-VL-3B는 open 모델이라 토큰이 필요 없다.)

### 1-2. Milvus (memory DB)
```bash
# Start it with the docker script in the remembr repo (127.0.0.1:19530)
bash /home/user/remembr/launch_milvus_container.sh start   # path is relative to the repo
docker ps | grep milvus                                    # confirm it is up
```
최초 실행 시 임베딩 모델 `mxbai-embed-large-v1`이 자동으로 다운로드된다.

### 1-3. remembr 전용 venv 생성
```bash
cd /home/user/remembr
python3 -m venv remembr_venv
source remembr_venv/bin/activate
pip install --upgrade pip          # ★ upgrade pip first (avoids the old pip resolver bug)

pip install "langchain-core==0.2.40" "langchain-community==0.2.16" \
            "langchain-openai==0.1.25" "langchain==0.2.16" \
            "langchain-huggingface==0.0.3" \
            "langgraph==0.2.40" "langsmith==0.1.147" \
            "numpy<2" "opencv-python-headless<5" \
            pillow requests pymilvus "sentence-transformers==3.1.0"
```
> **`opencv`/`numpy` 버전 핀에 주의하라.** `opencv-python-headless`는 CODa 전처리
> (`preprocess_coda_rosfree.py`, `cv2.imread`)에 필요하다. 그냥 `pip install
> opencv-python-headless`를 하면 opencv 5.x가 딸려오고, 이것이 **numpy 2.x를 끌어와** langchain 0.2.16 /
> sentence-transformers 3.1.0을 깨뜨린다(둘 다 numpy<2 요구). 반드시 `numpy<2` +
> `opencv-python-headless<5`(4.x)로 핀해야 한다. (GUI 없는 `-headless` 빌드를 써라.)
> **`langgraph`/`langsmith` 버전 핀에 주의하라.** `langgraph`는 `eval.py`가 import하는
> `ReMEmbRAgent`(RAG agent)의 핵심 의존성이다. 그냥 `pip install langgraph`를 하면
> 최신 langgraph가 딸려오고, 이것이 **langchain-core를 0.3.x로 올려 위의 모든 0.2.x 핀을 깨뜨린다.**
> 반드시 core 0.2.40과 호환되는 `langgraph==0.2.40` + `langsmith==0.1.147`로 핀해야 한다.
>
> **NVIDIA NIM 백엔드(`langchain_nvidia_ai_endpoints`)는 설치하지 마라.** 이 NPU
> 구성은 로컬 NPU만 사용하므로 불필요하다; 코드는 `nim/` 경로에서만 lazy-import 한다.
> pip가 `AssertionError: len(weights) == expected_node_count`로 죽으면, 그것이 옛 pip
> resolver 버그다. 위처럼 `pip install --upgrade pip`를 먼저 하면 해결된다. 그래도 실패하면
> `--use-deprecated=legacy-resolver` 플래그를 추가하거나 패키지를 더 작은 배치로 나눠 설치하라.

### 1-4. 서버 측 추가 의존성 (한 번, tt-metal python_env에)
```bash
source /home/user/tt-metal/env_set.sh          # activate python_env
pip install fastapi "uvicorn[standard]" lm-format-enforcer
```
`lm-format-enforcer`는 JSON 제약 디코딩(Ollama의 `format="json"` 대체)에 필요하다.

---

## 2. Starting the NPU server (tt-metal python_env)

이 절의 명령은 **사람 사용자가 tmux 서버 pane에서 직접 실행**한다. 에이전트가 대신 실행하거나
별도의 pane/process에 중복 서버를 띄우지 않는다. 시작 전 `curl localhost:8100/health`로 기존
서버가 없는지 확인한다.

```bash
cd /home/user/remembr/tt_server
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 ./run_server.sh
```

- **venv를 미리 활성화할 필요 없다.** 스크립트가 `env_set.sh`로 tt-metal venv를 직접 켜고
  `TT_METAL_HOME`으로 `cd`한다. **어느 디렉터리에서 어떤 방식으로 실행해도 된다**
  (`./run_server.sh`, `bash run_server.sh`, 절대경로 — 전부 동작).
- **`TT_LLM_MAX_SEQ_LEN=2048`은 필수다.** NonAgent 프롬프트는 2048까지 패딩되므로, 기본값
  (1024)이면 `Padded prefill end idx 2048 exceeds max seq len` 에러가 난다.
- VLM에는 1024로 충분하다(이미지는 서버에서 224×224로 리사이즈된다).
- **특정 모드로 바로 올리려면 `TT_MODE`를 지정한다** (기본은 `3b+3b`와 동일 구성):
  ```bash
  TT_MODE=caption-8b TT_PYTHON_ENV=tf457 ./run_server.sh   # Qwen3-VL-8B만, 4.57 venv
  TT_MODE=llm-8b ./run_server.sh                           # Llama-3.1-8B만
  ```
  오타 등 알 수 없는 모드는 device를 건드리기 전에 즉시 실패한다. 다른 사람이 이미 쓰고
  있는 서버라면 launcher가 자동으로 `/health`를 확인해 기존 상태를 출력하고 exit 0으로 종료한다.
  전역 startup lock, 기존 server process, `/dev/bos/0` holder도 device open 전에 검사하므로
  중복 서버를 띄우지 않는다. guard를 우회해 `tt_infer_server.py`를 직접 실행하지 말 것.
- 두 모델 로딩 + weight cache 빌드에 **몇 분**이 걸린다. 이 로그 줄이 나오면 준비 완료다:
  ```
  INFO:     Application startup complete.
  ```

### 준비 여부 확인 (다른 터미널에서)
```bash
curl -s http://localhost:8100/health | python -m json.tool
# -> {"ready": true, "mode": "default", "switching": false, "llm": true, ... }
```
`debug.decode`/`debug.sample`은 서버 시작 시 env에서 읽힌 상태만 보여준다. 실행 중 토글은 없다.

### 단독 sanity check (선택)
```bash
# LLM
curl -s http://localhost:8100/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Reply with JSON {\"ok\":true}"}]}'
```

### API 문서
- **Swagger UI: <http://localhost:8100/docs>** (`/redoc`, `/openapi.json`도 있음)
  — 다만 핸들러가 raw `dict`를 받으므로 **자동 생성된 스키마에는 필드가 표시되지 않는다**.
- **request/response 필드와 예시는 [README의 HTTP API 섹션](README.md#http-api)을 참고하라**
  (`/health`, `/caption`, `/v1/chat/completions` 모두 거기에 문서화되어 있다).

---

## 3. Running the smoke test (remembr_venv)

```bash
cd /home/user/remembr
source remembr_venv/bin/activate
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <image_path_1> <image_path_2> ...
```

예시 이미지 경로 (repo의 데모 이미지 사용):
```bash
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py \
  /home/user/tt-metal/models/bos_model/qwen25_vl/demo/images/*.jpg
```
인자 없이 실행하면 합성(단색) 프레임으로 경로만 점검한다.

기대 출력:
```
[1/4] Captioning ... on the NPU (Qwen2.5-VL)...
      caption: <the actual caption>
[2/4] Inserting into Milvus memory...
[3/4] Building NonAgent (Llama-3.2-3B on the NPU)...
[4/4] Querying...
      AgentOutput: AgentOutput(type=..., text=..., ...)
OK: end-to-end path completed.
```

---

## 4. Shutdown

**사람 사용자가 tmux의 서버 콘솔에서 `Ctrl-C`를 한 번만 누르고 기다려라.** 에이전트는
`tmux send-keys`나 signal 명령으로 대신 종료하지 않는다. (비상시 SIGTERM도 사람 사용자가 직접 판단한다.)
서버는 worker에게 정지 신호를 보내고 **`close_mesh_device()`로 device를 깨끗이 닫은 뒤** 종료한다.
이 로그 줄들이 나오면 안전하게 내려간 것이다:
```
Shutting down: releasing the NPU cleanly ...
NPU released cleanly. Safe to exit.
```
device teardown 동안(진행 중인 op가 끝나기를 최대 180 s 대기), **추가 Ctrl-C 입력은 무시된다**.

> ⚠️ **`Ctrl-C`를 연타하지 마라 / `-9`(SIGKILL)를 쓰지 마라.**
> 예전에는 device를 닫는 shutdown 코드가 **daemon worker의 finally 블록**에 있어서 인터프리터 종료 시
> 실행되지 않았고, **Ctrl-C 연타 → teardown 중단 → 보드 wedged**로 이어졌다. 이후 shutdown 경로는
> worker를 join하고 그 구간 동안 SIGINT/SIGTERM을 무시하도록 수정되어, 이제 연타는 안전하다.
> 하지만 **`-9`(SIGKILL) 단독은 커널이 곧바로 죽여버려 이 cleanup을 건너뛴다** → device가 활성 상태로
> 남고 보드가 wedged 된다. 절대 `-9`를 쓰지 마라(정말 죽지 않는 프로세스에 대한 최후의 수단으로만 —
> 그 뒤에는 board reset을 각오하라).

```bash
lsof -i :8100                        # no output means it is fully down
fuser -v /dev/bos/0                  # no output means the device is fully released
```

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|------|------------|
| **Captions get slower over time / server RSS keeps growing** | 원인 두 가지. ① tt-metal의 **Inspector**(기본 ON)가 워크로드마다 YAML을 flush → `run_server.sh`가 `TT_METAL_INSPECTOR=0`으로 비활성화한다(이미 적용됨). ② **Tracy profiler 빌드** → 1-0 참고, 기본 빌드로 재빌드해야 한다(env로는 끌 수 없다). 판단 기준: 서버 로그의 `32 x Qwen2_5_VLVisionBlock ... it/s`가 떨어지면 의심하라. |
| **The first caption is very slow (several minutes), client times out** | 새로 띄운 서버에서는 **첫 요청이 ttnn 트레이스(prefill/decode)를 처음으로 컴파일**한다 — 일회성 비용이다. 데드락이 아니다(서버 CPU는 계속 돌고 있다). 끝나면 이후 요청은 훨씬 빠르다. 클라이언트 타임아웃 기본값은 900 s이며 `TT_VLM_TIMEOUT`(초)로 조정 가능하다. |
| `address already in use (8100)` | 서버가 이미 떠 있다. 새로 띄울 필요 없다. 내리려면 4장 참고. |
| `[tt-server] startup skipped: server already ready ...` | 정상 no-op. 기존 서버의 health를 출력하고 새 process를 만들지 않은 것이다. |
| `[tt-server] startup blocked: an inference server process exists ...` | process는 있지만 health가 준비되지 않았다. 로딩 중인지 로그를 확인하고 중복 기동하지 않는다. |
| `[tt-server] startup blocked: NPU device ... already held ...` | 다른 process가 보드를 점유 중이다. 표시된 PID의 소유자를 확인하고 임의 종료하지 않는다. |
| `Padded prefill end idx 2048 exceeds max seq len` | `TT_LLM_MAX_SEQ_LEN=2048`로 서버를 재시작하라. |
| LLM emits code fences / broken JSON, NonAgent retries in an infinite loop | JSON 제약 디코딩이 적용되지 않은 것이다. `lm-format-enforcer`가 설치되었는지 확인하라(1-4). |
| HF 403 / gated error | `~/.bashrc`의 `HF_TOKEN`을 확인하라. `run_server.sh`가 거기서 읽는다. |
| OOM | 이 보드는 DRAM을 ~12GB(3 banks)만 노출한다. 3B+3B는 함께 올라가지만 8B LLM은 안 된다 → 타임셰어링 fallback(`TT_ENABLE_VLM=1 TT_ENABLE_LLM=0`으로 caption → 이후 교체)이나 더 작은 모델을 써라. |
| pip `AssertionError ... expected_node_count` | 옛 pip resolver 버그. `pip install --upgrade pip` 후 재설치하라(1-3). |

---

## 6. Key env vars

**Server (`run_server.sh` / tt_infer_server.py)**

| var | default | description |
|-----|--------|------|
| `TT_LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM |
| `TT_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM |
| `TT_LLM_MAX_SEQ_LEN` | `1024` (→ **2048로 실행**) | NonAgent needs 2048 |
| `TT_VLM_MAX_SEQ_LEN` | `1024` | captioning |
| `TT_ENABLE_LLM` / `TT_ENABLE_VLM` | `1` / `1` | fallback 시 하나만 로드 |
| `TT_TRACE_REGION_SIZE` | `15000000` | per-model trace buffer |
| `TT_MAX_CAPTION_FRAMES` | `5` | 요청당 최대 frame 수. 보드 안정성을 위해 6장 이상은 HTTP 400 |
| `TT_WORKER_SHUTDOWN_TIMEOUT` | `180` | device worker join 대기 시간(non-daemon worker라 timeout 뒤에도 강제 폐기되지 않음) |
| `TT_SERVER_LOCK_FILE` | `/tmp/remembr-tt-npu-server.lock` | server lifetime 동안 유지되는 전역 NPU startup lock |
| `TT_HEALTH_HOST` / `TT_HEALTH_TIMEOUT` | `127.0.0.1` / `2` | preflight health host / timeout(초) |
| `TT_PORT` | `8100` | |
| `TT_DEBUG_DECODE` | `0` | LLM token stream을 stdout에 출력. 생성 내용/JSON 깨짐 확인용 |
| `TT_DEBUG_SAMPLE` | `0` | token masking/sampling 내부 상태 출력. schema/debug 용도, 로그가 매우 큼 |

**remembr side**

| var | default | description |
|-----|--------|------|
| `TT_LLM_BASE_URL` | `http://localhost:8100/v1` | NonAgent → server |
| `TT_VLM_URL` | `http://localhost:8100/caption` | Captioner → server |
| `TT_VLM_TIMEOUT` | `900` | caption client timeout (seconds) |
| `TT_LLM_MAX_TOKENS` | `256` | max tokens the LLM generates |
