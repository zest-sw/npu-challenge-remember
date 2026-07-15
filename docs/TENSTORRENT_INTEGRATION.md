# ReMEmbR × NPU Integration — Architecture & Design

A record of the integration work needed to run the robot spatio-temporal memory system **ReMEmbR** end-to-end on an NPU.
Both the VLM (captioner) and the LLM (question answering) run on the NPU, with Milvus vector memory bridging the two.

> **For progress status, change log, and known issues see [DEV_STATUS.md](DEV_STATUS.md).** This document focuses on
> architecture, design decisions, and debugging records.
>
> **Hardware note**: this machine is a **BOS chip** (not Tenstorrent). Since we use ttnn through a tt-metal-compatible
> stack (`bos-metal`), the logs show `BlackholeTTDevice`, but the actual device node is **`/dev/bos/0`** and the kernel
> module is **`bos`**. To check device occupancy use `fuser -v /dev/bos/0` / `lsmod | grep bos`. (`/dev/tenstorrent` and
> `lspci 1e52` do not exist on this machine.)
>
> **Measured specs** (`python tt_server/npu_info.py`): arch `BLACKHOLE`, 1 device, **DRAM 12 GiB**
> (4 GiB × 3 banks), **Tensix 5×4 = 20 cores** (harvesting `0x0`), L1 1.5 MB per worker core.
> The 12 GiB of DRAM is the basis for "3B+3B can be co-resident / 8B cannot".

---

## 1. Goal

The ReMEmbR pipeline:

```
Camera frame ──▶ [VLM caption] ──▶ [store in vector memory] ──▶ [question] ──▶ [search] ──▶ [LLM answer]
                  Qwen2.5-VL        Milvus                                      top-k        Llama-3.2-3B
                  (NPU)                                                                      (NPU)
```

- The original ReMEmbR used **VILA** as the captioner and **GPT-4o / llama3.1 (8B class)** as the LLM.
- We replace both with **3B models running on the Tenstorrent NPU**.
  - VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
  - LLM: `meta-llama/Llama-3.2-3B-Instruct`

---

## 2. Architecture

### 2.1 Unified inference server (`remembr/tt_server/tt_infer_server.py`)

Rather than loading the NPU models directly into the remembr process, we **split them out into an HTTP server**.
Reason: the ttnn/tt-metal stack conflicts with remembr's dependency pins, and two processes cannot occupy a single device at the same time.

- **Port 8100**, FastAPI.
- Endpoints
  - `POST /v1/chat/completions` — LLM (OpenAI compatible)
  - `POST /caption` — VLM (list of base64 images + prompt → caption)
  - `GET /health` — load status
- **Single worker thread + job queue** design: all device work (open/load/prefill/decode) is serialized onto one thread. FastAPI handlers push a job onto the queue and block on a Future. → No matter how many HTTP requests arrive, the device handles exactly one at a time.
- Both the LLM and VLM engines are **loaded onto the device co-resident at startup** (`TT_ENABLE_LLM`/`TT_ENABLE_VLM`, both on by default). The `_ready` event blocks startup until loading completes.
- Key env vars
  - `TT_LLM_MAX_SEQ_LEN` (default 4096), `TT_VLM_MAX_SEQ_LEN` (default 2048)
  - `TT_PORT` (8100), `TT_HOST` (0.0.0.0)
- **Grammar-constrained decoding for JSON** is supported: OpenAI `response_format` is honored.
  - `json_object` → any valid JSON (equivalent to Ollama `format="json"`)
  - `json_schema` → constrained to the given schema (`TokenEnforcer` + `JsonSchemaParser`, host-side masking followed by argmax)
- VLM input images are **resized to 224×224** on the server.

### 2.2 Captioner (`remembr/captioners/tenstorrent_captioner.py`)

Implements the `Captioner` interface. `caption(images: list[PIL.Image]) -> str`.

- POSTs a list of base64 JPEGs + a prompt to the server's `/caption`.
- env: `TT_VLM_URL` (default `http://localhost:8100/caption`), `TT_VLM_TIMEOUT` (default 900 s).
  - The first caption on a cold start can take several minutes due to ttnn trace (prefill+decode) compilation, so the timeout is set generously.
- **The interface is frame list (window) → one caption.** Passing an entire window at once is the original design for real sequence data.

### 2.3 LLM agent (`remembr/agents/non_agent.py`)

`NonAgent` — a lightweight agent that produces an answer with a **single LLM call**, no tools.
If `llm_type` has the form `tt/<model>`, the Tenstorrent backend is used.

- Calls the server's `/v1/chat/completions` via `ChatOpenAI` (`TT_LLM_BASE_URL`, default `http://localhost:8100/v1`).
- **Forces grammar-constrained decoding with a JSON schema**: every key required by `AgentOutput` (`type, text, binary, position, orientation, time, duration`) is marked `required`, `text` is `maxLength: 300`, and `additionalProperties: false`.
  - This stops small models from spinning in reasoning fields and burning their budget, and prevents infinite retries caused by parse failures.
- `TT_LLM_MAX_TOKENS` (default 256) — upper bound on generated tokens.

### 2.4 Memory (`remembr/memory/milvus_memory.py`)

- Stores `caption` + `text_embedding` (1024d, `mxbai-embed-large-v1`) + `position` (3d) + `time` (2d) in a Milvus collection.
- Three kinds of search: `search_by_text` (semantic), `search_by_position` ((x,y,z)), `search_by_time` (H:M:S). Each accumulates results into `working_memory`.
- `NonAgent` uses `get_working_memory()` as context → **you must call `search_*` before querying** or the context stays empty.

### 2.5 Smoke test (`examples/tt_smoke_test.py`)

A script that validates the whole end-to-end path (caption → store → search → answer) in one shot.
Usage: `PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <images...>`

---

## 3. Agent vs NonAgent (design choice)

| | `ReMEmbRAgent` | `NonAgent` (what we use) |
|---|---|---|
| Who searches | The LLM actively picks and repeats tools (LangGraph loop) | Search is done externally in advance; the LLM does not search |
| Search method | Free choice among text/position/time | Fixed to the context handed in |
| LLM calls | Multiple (tool loop ≤3 + generate) | **1** |
| tool calling | **Required** | Not needed |
| Suitability for small models | Low (3B is unreliable at tool calls) | **High** |

→ Since Llama-3.2-3B on the NPU is unreliable at tool calling, we **adopt `NonAgent`**. The tradeoff is that the LLM cannot perform position/time-specialized searches on its own, so our code fills in the search results for it.

---

## 4. Debugging log — problems found and fixes

### Problem ① Vector search returns nonsense (hallucinated dog on a skateboard)

- **Symptom**: fed a pasta image, got back "I saw a dog".
- **Cause**: `MilvusMemory.__init__` creates with `reset(drop_collection=False)` → **the collection is never dropped**. Old captions from previous runs (dog.jpg, etc.) kept accumulating in the `tt_smoke_test` collection. A generic question ("What did you see?") doesn't bind tightly to any particular caption, so stale entries got mixed into the search results.
- **Fix**: reset the collection at the start of the smoke test with `memory.reset(drop_collection=True)`.
  - ⚠️ In real operation memory must accumulate, so dropping is forbidden → in that case the right approach is to **use a different collection name per session**.

### Problem ② The VLM refuses to caption ("I'm sorry, but I can't assist with that.")

- **Symptom**: passing 2 images at once produced a refusal. With 1 image it worked fine.
- **Investigation**:
  - `<video>` + 2 images → **refusal**
  - `<video>` + 1 image → fine
  - 2 images without `<video>` → no refusal, but the two images get **merged into a single hallucinated scene**
- **Cause**: the prompt began with a literal `<video>` token. That is a **VILA convention** from the original ReMEmbR (VILA treats `<video>` as a multi-frame placeholder special token). **Qwen2.5-VL does not know `<video>`** — its processor automatically inserts `<|vision_start|>...<|vision_end|>`, so a literal `<video>` plus multiple images conflicts with the trained format and triggers a refusal.
- **Fix**: removed `<video>` from `DEFAULT_PROMPT` and replaced it with "These are sequential frames from a robot's camera...". → Multi-frame input now works correctly.

> **What is the `<video>` token?** A VLM concatenates text tokens and image embeddings into one sequence, and the `<video>` token is a **placeholder marking where the image embeddings go**. In VILA the user writes it into the prompt directly, but in Qwen2.5-VL the processor inserts it automatically, so the user must not write it.

### Problem ③ Multiple frames lumped into one caption and one memory entry

- **Symptom**: unrelated demo images (dog+pasta) were merged into a single caption and stored → no meaningful unit left to search on.
- **Fix**: changed the smoke test to do **a separate caption per frame → a separate `MemoryItem` per frame**. Each frame gets a staggered `time` (+10s each) and `position` (along the x axis). The retrieved context is also printed so it can be eyeballed.
  - ⚠️ This changed **only the smoke test script**. **The captioner interface (window → one caption) is unchanged.** For real sequence video the frames are consecutive views of the same scene, so grouping them into a window is the right thing to do.

### Problem ④ Captioning refuses again ("I can't see the video")
- **Symptom**: captioning CODa frames with the paper's prompt produced a refusal saying it cannot see the video.
- **Cause**: the paper's original prompt is for VILA **video** ("the few seconds of the video"), but we hand Qwen **a single still frame** (`--num-video-frames 1`) → prompt/input mismatch.
- **Fix**: keep the original wording but neutralize only the video phrasing (remove `<video>`, "the few seconds of the video" → "the image"). Everything else ("Think step by step", etc.) stays exactly as in the paper → fidelity preserved and the refusal resolved.

### Problem ⑤ Caption sentences get cut off mid-way
- **Cause**: the server's `/caption` caps generation at **128 tokens** when `max_gen_len` is unspecified. A step-by-step caption doesn't emit EOT before that and gets truncated.
- **Fix**: the captioner now sends `max_gen_len` (default 256, `TT_VLM_MAX_GEN_LEN`). The server already honors this parameter, so **no server code changed** — just rerun the client.

### Problem ⑥ Captions get slower over time (21s → 80s) — **culprit: tt-metal Inspector**
- **Measurement**: caption length, host RAM, and swap were all stable. The decisive clue was the vision block speed in the server log **falling monotonically from 19 to 2.1 it/s** (9×).
- **Cause**: tt-metal's **Inspector** (a debug feature) is **ON by default** (`rtoptions.hpp`: `InspectorSettings::enabled = true`). Its logger **appends YAML for every mesh workload and calls `fstream::flush()` each time** (`logger.cpp`) → in our case roughly **5,000 synchronous disk flushes per caption**. In a single captioning run `generated/inspector/mesh_workloads_log.yaml` grew to **153MB / 5.34M lines**, and the bigger the file got the more each flush blocked, dragging the VLM down with it.
- It is **not** a memory problem: the inspector's `unordered_map` is cleaned up via `erase`, and host RSS was in fact flat at 11.3GB.
- **Fix**: **without touching the tt-metal source**, set the switches tt-metal already provides in `run_server.sh` —
  `TT_METAL_INSPECTOR=0`, `TT_METAL_INSPECTOR_RPC=0`.
- **Measured result**: the vision block **holds steady at ~42 it/s** (the decay is gone) and no inspector YAML is produced. The `Inspector RPC 50051` warning in the server log also disappears.
- **Misdiagnosis record**: at first this looked like device tensors piling up, so `ttnn.deallocate` was added to `describe()`, but that **was not the cause** (it targeted LLM decoding tensors) and was reverted. `TT_METAL_DEVICE_PROFILER` is also unrelated (it is only active when set to `'1'`, and it was unset, so it was a no-op).

### Problem ⑦ Ctrl-C (especially mashing it) wedges the board
- **Cause**: `close_mesh_device()`, which closes the device, lives in **the finally block of a daemon worker**, so it does not run when the interpreter exits. Mashing Ctrl-C force-kills uvicorn and aborts teardown → the device is left active and **wedged** (on restart: `failed to initialize FW`).
- **Fix**: on shutdown, signal the worker to stop and then `join()` to guarantee `close_mesh_device()` runs, plus **ignore SIGINT/SIGTERM during the teardown window** (safe against key mashing). Only `-9` (SIGKILL) still bypasses this, so it is **forbidden**.

### Problem ⑫ The **second** cause of the slowdown — tt-metal was a Tracy profiling build
- **Symptom**: even after turning off the Inspector, the vision block still declined **from 84 to 50 it/s**. The Inspector was so dominant
  (19→2.1) that this was hidden underneath it.
- **Quantitative confirmation**: 12 consecutive requests with **exactly the same image** →
  ```
  RSS     : 5280 → 5649 MB   (+33MB per request, almost no variance)
  Latency : 55.8s → 67.8s    (+1.0s per request, monotonic increase)
  ```
  Identical input yet linear growth = not caching or compilation, but **something accumulating per request**.
- **Cause**: this tt-metal is **a Tracy build produced by `build_metal.sh --enable-profiler`**
  (`build -> build_Release_tracy`, `CMakeCache: ENABLE_TRACY:BOOL=ON`, 705 Tracy symbols in `_ttnn.so`).
  The default in `build_metal.sh` is `enable_profiler="OFF"`, so someone deliberately passed `-p`.
  When Tracy is compiled without `TRACY_ON_DEMAND` it **accumulates every zone in memory from program start,
  regardless of whether a UI is connected.** `ReadDeviceProfilerResults` skips its inner block when the profiler
  is disabled, but the **`ZoneScoped;` (a Tracy macro)** above it still executes inside `#if defined(TRACY_ENABLE)`.
  On top of that, `decode_forward` calls `ttnn.ReadDeviceProfiler` **on every decoded token** (~256 per caption)
  and the vision block loop calls it **on every block** (32 times) → tens of thousands of zones accumulated per caption.
- **Why env vars can't turn it off**: `TT_METAL_DEVICE_PROFILER` only disables the profiler's *operation*; Tracy is a
  **compile-time** switch and cannot be turned off at runtime.
- **Fix**: swap in the **default build** with no profiler — `./build_metal.sh` (default `ENABLE_TRACY=OFF`).
  No tt-metal source changes; the build directory is separate (`build_Release`) and only the symlink changes, so
  **rollback is a one-liner: `ln -nsf build_Release_tracy build`**. The build takes **7 minutes** on 24 cores.
- **Verification (re-measured under identical conditions)**: Tracy symbols in `_ttnn.so` went **705 → 0**.
  ```
              Tracy build              Default build
  RSS growth  +33MB/request (linear)   +0~1MB (flat)
  Latency     55.8 → 67.8s             36.0 → 36.0s (constant)
  Vision      84 → 50 it/s (falling)   84~185 it/s (steady)
  ```
  **The leak and the latency growth vanished completely.** Estimate for 272 captions: ~13 hours → **~2.9 hours**.
- **Lesson**: when investigating a performance anomaly, **first suspect that the environment is a profiling build**. Both causes
  (Inspector, Tracy) came down to "a debug tool was on by default or on purpose."

### Problem ⑨ Two prefills per request (2× waste in the vision encoder)
- **Discovery**: the server log printed `32 x Qwen2_5_VLVisionBlock` **twice** per caption.
- **Cause**: the original demo (`qwen25_vl/demo/vision_demo.py`) calls prefill twice **on purpose** —
  `profiler("compile_prefill")` warms the program cache, and `profiler("inference_prefill")` measures pure inference
  time. It is **a benchmarking device meant to report compile vs. inference ms separately**, and it is correct for a demo.
  But the server **only compiles on the first request** and is warm from then on, so the second pass was a pure duplicate forward.
  Worse, the first result went into `masks`, an unused variable in the VLM, and in the LLM (`generate`) it **wasn't even assigned to a variable**.
- **Fix**: keep only **one** pass (= the demo's `inference_prefill`) for both VLM and LLM. Measured: VisionBlock per caption 2× → 1×,
  ~76s → **~38s**.

### Problem ⑩ The optimal eval path crashes on the first question → infinite retries
- **Symptom**: `AttributeError: 'str' object has no attribute 'time'` (`text_memory.py`).
- **Cause**: the `optimal+` path in eval.py puts **an already-rendered string** (`qa_instance['context']`) into
  `TextMemory`. `get_working_memory()` returns the string as-is when it is a str, and `NonAgent` then passes it back
  into `memory_to_string()`, which **iterates the string character by character** → blows up at `doc.time`. An upstream bug.
- **Fix**: `memory_to_string` now passes strings straight through (= consistent with the documented intent of `get_working_memory`).

### Problem ⑪ (unresolved) The server ignores temperature → the eval retry loop never terminates
- **Symptom**: when the 3B answers a time question with `type='text', time=None`, `evaluate_output` crashes on
  `abs(answer - None)` → the `while True` in `answer_squad_question` retries forever.
- **Cause**: the original assumes **`temperature=0.7`** (eval.py's default), so a retry yields a different answer. But
  our `/v1/chat/completions` **never even reads temperature** and decodes only with `_masked_argmax` (greedy)
  → retries produce **the identical answer** → infinite loop. (This is a recurrence of the "greedy → identical retries"
  warning in `tt_server/README.md`; that time it was a JSON validity problem, this time **the JSON is valid but the content is wrong**.)
- **Options**: (A) have the server honor temperature (restoring the original intent) / (B) add a None guard in `evaluate_output` /
  (C) enforce per-type fields via JSON schema. **Direction undecided.**

### Problem ⑧ "The board is dead" misdiagnosis (device node name)
- **Cause**: diagnostics checked `/dev/tenstorrent` and `lspci 1e52` → always "not found" → repeated misdiagnosis of "fell off the bus". In reality it is a **BOS chip**, so the node is `/dev/bos/0`.
- **Lesson**: check the device with `fuser -v /dev/bos/0`, `lsmod | grep bos`, `/proc/<pid>/fd | grep bos`.

---

## 5. Remaining issues / next steps

### 3B LLM quality
- The 3B is "usable but borderline" for RAG. Given clean context it stays grounded, but when the context is empty or polluted it degenerates (repetition, foreign-language token leakage) — e.g. output like `有有`.
- Greedy (temperature=0) with no repetition penalty encourages the degeneration.
- **Options**
  1. **Switch to 8B** — DRAM constraints may make co-residency impossible → fall back to a configuration that time-separates the captioning and query stages (only one model loaded at a time).
  2. **Stay on 3B + tune** — add a repetition penalty to the server, reduce `TT_LLM_MAX_TOKENS`.

### Real sequence data (in progress)
- The pipeline for the original paper's data (CODa) is built: `preprocess_coda_rosfree.py` (ROS-free .pkl extraction) → `preprocess_captions.py --captioner tenstorrent` (checkpointed captioning) → `form_question_jsons.py` → `eval.py`. Details in [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md).
- **Fidelity deviation**: frames 6→1 (a compromise). Qwen supports multiple frames too, but we start at 1 because of first-compile cost and stability. To raise it, increase `--num-video-frames` (be prepared to wait out the first compile).
- For progress status see the [DEV_STATUS](DEV_STATUS.md) status board.

### Housekeeping
- Many PyMilvus/LangChain deprecation warnings (`connections.connect`, ORM-style APIs, etc.). Harmless functionally, but migrating to `MilvusClient` / `langchain-milvus` later is recommended.

---

## 6. How to run (summary)

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

Restarting the server is only necessary when server (`tt_server/`) code changes. Edits to the captioner/agent/smoke test (the remembr side) do not require a restart.

---

## 7. Files added/modified

| File | Role |
|---|---|
| `tt_server/tt_infer_server.py` | Unified NPU inference server (LLM+VLM, port 8100, constrained JSON decoding, clean shutdown, tensor deallocation) |
| `tt_server/run_server.sh` | Server launcher (env setup) |
| `remembr/captioners/tenstorrent_captioner.py` | Qwen2.5-VL captioner (HTTP client, `max_gen_len`, import fallback) |
| `remembr/agents/non_agent.py` | Llama-3.2-3B `tt/` backend, JSON schema constraints |
| `remembr/agents/remembr_agent.py` | `ChatNVIDIA` lazy import |
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa → .pkl extractor |
| `remembr/scripts/preprocess_captions.py` | Captioning (`--captioner tenstorrent`, checkpoint/resume, prompt neutralization) |
| `examples/tt_smoke_test.py` | end-to-end smoke test |

*(The tt-metal original is untouched; the added code is collected on the remembr side.)*
