# ReMEmbR on Tenstorrent NPU — Runbook (from scratch)

This is the procedure for bringing up VLM (captioning) = Qwen2.5-VL-3B and LLM (reasoning) = Llama-3.2-3B
on **a single Tenstorrent board**, and running the remembr core pipeline
(`image → caption → memory insert → question answering`) end-to-end.

---

## 0. Architecture at a glance

```
 remembr (remembr_venv)                     tt_infer_server.py (tt-metal python_env, owns the board)
   NonAgent ── HTTP /v1/chat/completions ─────► Llama-3.2-3B  ┐  1 process, device opened once,
   Captioner ─ HTTP /caption ─────────────────► Qwen2.5-VL-3B ┘  both models co-resident,
                                                                 single worker + queue serializes all NPU execution
```

- **Two venvs** are used. The server runs in tt-metal's `python_env` (ttnn/transformers); remembr runs in
  its own dedicated venv (langchain/pymilvus). Their versions conflict, so they are separated over HTTP.
- The NPU only runs one graph at a time → captioning and the LLM are automatically serialized by the queue
  (concurrent requests are safe).

---

## 1. First-time setup (once)

### 1-0. ⚠️ Check tt-metal is **NOT a profiler build** (critical for performance)

```bash
readlink /home/user/tt-metal/build          # must be build_Release (build_Release_tracy ✗)
grep ENABLE_TRACY /home/user/tt-metal/build/CMakeCache.txt   # must be ENABLE_TRACY:BOOL=OFF
nm -D --defined-only /home/user/tt-metal/ttnn/ttnn/_ttnn.so | grep -ci tracy   # must be 0
```

If `ENABLE_TRACY=ON` (i.e. built with `build_metal.sh -p/--enable-profiler`), **inference gets slower with every request.**
When Tracy is compiled without `TRACY_ON_DEMAND`, it **keeps accumulating every zone in memory** regardless of whether a
Tracy UI is connected, and `decode_forward` calls `ReadDeviceProfiler` on every decoded token (~256 times per caption).
Measured (same image, 12 times): **RSS +33MB per request, latency growing linearly from 55.8 → 67.8 s**. 272 captions
would stretch from ~2.9h to ~13h.
**It is a compile-time switch, so it cannot be turned off via env.** If you have a profiler build, rebuild with the defaults:

```bash
cd /home/user/tt-metal && source env_set.sh
./build_metal.sh            # defaults to ENABLE_TRACY=OFF. ~7 min on 24 cores
#  -> creates a fresh build_Release and repoints the `build` symlink at it.
#     The existing build_Release_tracy is left intact, so rolling back is one line:
#     ln -nsf build_Release_tracy build
```
The server must be restarted after the rebuild for it to take effect. To verify: caption the same image several times —
**if latency stays flat and server RSS does not grow**, you are fine (with a Tracy build, both increase monotonically).

### 1-1. HF token (gated model)
`meta-llama/Llama-3.2-3B-Instruct` is gated. The token must be present in `~/.bashrc` in the form
`export HF_TOKEN=...`. `run_server.sh` reads that line directly, even in a non-interactive shell.
(Qwen2.5-VL-3B is open, so no token is needed.)

### 1-2. Milvus (memory DB)
```bash
# Start it with the docker script in the remembr repo (127.0.0.1:19530)
bash /home/user/remembr/launch_milvus_container.sh start   # path is relative to the repo
docker ps | grep milvus                                    # confirm it is up
```
On the first run, the embedding model `mxbai-embed-large-v1` is downloaded automatically.

### 1-3. Create the remembr-only venv
```bash
python3 -m venv ~/remembr_venv
source ~/remembr_venv/bin/activate
pip install --upgrade pip          # ★ upgrade pip first (avoids the old pip resolver bug)

pip install "langchain-core==0.2.40" "langchain-community==0.2.16" \
            "langchain-openai==0.1.25" "langchain==0.2.16" \
            "langchain-huggingface==0.0.3" \
            "langgraph==0.2.40" "langsmith==0.1.147" \
            "numpy<2" "opencv-python-headless<5" \
            pillow requests pymilvus "sentence-transformers==3.1.0"
```
> **Mind the `opencv`/`numpy` version pins.** `opencv-python-headless` is required for CODa preprocessing
> (`preprocess_coda_rosfree.py`, `cv2.imread`). A plain `pip install
> opencv-python-headless` pulls in opencv 5.x, which **drags in numpy 2.x** and breaks langchain 0.2.16 /
> sentence-transformers 3.1.0 (both require numpy<2). You must pin `numpy<2` +
> `opencv-python-headless<5` (4.x). (Use the GUI-less `-headless` build.)
> **Mind the `langgraph`/`langsmith` version pins.** `langgraph` is a core dependency of
> `ReMEmbRAgent` (the RAG agent) that `eval.py` imports. A plain `pip install langgraph` pulls in
> the latest langgraph, which **bumps langchain-core to 0.3.x and breaks every 0.2.x pin above.**
> You must pin `langgraph==0.2.40` + `langsmith==0.1.147`, which are compatible with core 0.2.40.
>
> **Do not install the NVIDIA NIM backend (`langchain_nvidia_ai_endpoints`).** This NPU
> setup only uses the local NPU, so it is unnecessary; the code lazy-imports it only on the `nim/` path.
> If pip dies with `AssertionError: len(weights) == expected_node_count`, that is the old pip
> resolver bug. Doing `pip install --upgrade pip` first, as above, fixes it. If it still fails,
> add the `--use-deprecated=legacy-resolver` flag or install the packages in smaller batches.

### 1-4. Extra server-side dependencies (once, into the tt-metal python_env)
```bash
source /home/user/tt-metal/env_set.sh          # activate python_env
pip install fastapi "uvicorn[standard]" lm-format-enforcer
```
`lm-format-enforcer` is required for JSON-constrained decoding (the replacement for Ollama's `format="json"`).

---

## 2. Starting the NPU server (tt-metal python_env)

```bash
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 \
  bash /home/user/remembr/tt_server/run_server.sh
```

- **`TT_LLM_MAX_SEQ_LEN=2048` is mandatory.** NonAgent prompts are padded up to 2048, so with the
  default (1024) you get the error `Padded prefill end idx 2048 exceeds max seq len`.
- 1024 is enough for the VLM (images are resized to 224×224 on the server).
- Loading both models + building the weight cache takes **several minutes**. It is ready when this log line appears:
  ```
  INFO:     Application startup complete.
  ```

### Checking readiness (from another terminal)
```bash
curl -s http://localhost:8100/health
# -> {"ready":true,"llm":true,"vlm":true}
```

### Standalone sanity check (optional)
```bash
# LLM
curl -s http://localhost:8100/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Reply with JSON {\"ok\":true}"}]}'
```

### API documentation
- **Swagger UI: <http://localhost:8100/docs>** (also `/redoc`, `/openapi.json`)
  — but the handlers take a raw `dict`, so **the auto-generated schema does not show the fields**.
- **For request/response fields and examples, see [the HTTP API section of the README](README.md#http-api)**
  (`/health`, `/caption`, and `/v1/chat/completions` are all documented there).

---

## 3. Running the smoke test (remembr_venv)

```bash
source ~/remembr_venv/bin/activate
cd /home/user/remembr
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <image_path_1> <image_path_2> ...
```

Example image paths (using the demo images in the repo):
```bash
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py \
  /home/user/tt-metal/models/bos_model/qwen25_vl/demo/images/*.jpg
```
Run it with no arguments and it checks the path only, using synthetic (solid-color) frames.

Expected output:
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

**Press `Ctrl-C` once in the server console and wait.** (Or `kill -TERM <pid>` / `pkill -f tt_infer_server.py` = SIGTERM.)
The server signals the worker to stop and exits **after closing the device cleanly with `close_mesh_device()`**.
It came down safely if these log lines appear:
```
Shutting down: releasing the NPU cleanly ...
NPU released cleanly. Safe to exit.
```
During device teardown (which waits for any in-flight op to finish, up to 180 s), **further Ctrl-C presses are ignored**.

> ⚠️ **Do not spam `Ctrl-C` / do not use `-9` (SIGKILL).**
> Previously the shutdown code that closed the device lived in the **daemon worker's finally block**, so it did not
> run at interpreter exit, and **spamming Ctrl-C → aborted teardown → wedged board**. The shutdown path has since
> been fixed to join the worker and ignore SIGINT/SIGTERM during that window, so spamming is now safe.
> But **`-9` (SIGKILL) alone is killed outright by the kernel and skips this cleanup** → the device is left active
> and the board is wedged. Never use `-9` (last resort only for a truly stuck process that will not die — expect a
> board reset afterwards).

```bash
lsof -i :8100                        # no output means it is fully down
fuser -v /dev/bos/0                  # no output means the device is fully released
```

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|------|------------|
| **Captions get slower over time / server RSS keeps growing** | Two causes. ① tt-metal's **Inspector** (ON by default) flushes YAML per workload → `run_server.sh` disables it with `TT_METAL_INSPECTOR=0` (already applied). ② A **Tracy profiler build** → see 1-0, you must rebuild with the default build (it cannot be turned off via env). Gauge: suspect this if `32 x Qwen2_5_VLVisionBlock ... it/s` in the server log drops. |
| **The first caption is very slow (several minutes), client times out** | On a fresh server, the **first request compiles the ttnn traces (prefill/decode) for the first time** — a one-time cost. It is not a deadlock (the server CPU keeps spinning). Once it finishes, subsequent requests are much faster. The client timeout defaults to 900 s and is tunable via `TT_VLM_TIMEOUT` (seconds). |
| `address already in use (8100)` | The server is already up. No need to start another one. To bring it down, see section 4. |
| `Padded prefill end idx 2048 exceeds max seq len` | Restart the server with `TT_LLM_MAX_SEQ_LEN=2048`. |
| LLM emits code fences / broken JSON, NonAgent retries in an infinite loop | JSON-constrained decoding is not in effect. Check that `lm-format-enforcer` is installed (1-4). |
| HF 403 / gated error | Check `HF_TOKEN` in `~/.bashrc`. `run_server.sh` reads it from there. |
| OOM | This board only exposes ~12GB of DRAM (3 banks). 3B+3B co-fits, but an 8B LLM does not → use the time-sharing fallback (caption with `TT_ENABLE_VLM=1 TT_ENABLE_LLM=0` → then swap) or a smaller model. |
| pip `AssertionError ... expected_node_count` | Old pip resolver bug. Run `pip install --upgrade pip` and reinstall (1-3). |

---

## 6. Key env vars

**Server (`run_server.sh` / tt_infer_server.py)**

| var | default | description |
|-----|--------|------|
| `TT_LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM |
| `TT_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM |
| `TT_LLM_MAX_SEQ_LEN` | `1024` (→ **run with 2048**) | NonAgent needs 2048 |
| `TT_VLM_MAX_SEQ_LEN` | `1024` | captioning |
| `TT_ENABLE_LLM` / `TT_ENABLE_VLM` | `1` / `1` | load only one when falling back |
| `TT_TRACE_REGION_SIZE` | `15000000` | per-model trace buffer |
| `TT_PORT` | `8100` | |

**remembr side**

| var | default | description |
|-----|--------|------|
| `TT_LLM_BASE_URL` | `http://localhost:8100/v1` | NonAgent → server |
| `TT_VLM_URL` | `http://localhost:8100/caption` | Captioner → server |
| `TT_VLM_TIMEOUT` | `900` | caption client timeout (seconds) |
| `TT_LLM_MAX_TOKENS` | `256` | max tokens the LLM generates |
