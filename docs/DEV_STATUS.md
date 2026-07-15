# Development Status (DEV STATUS)

> **What this document is for**: the **living tracking document** for the ReMEmbR × NPU porting project.
> The single source of truth for "where we are, what's been verified, what's left, and why we decided things this way."
> Update the **① status board** and **⑥ changelog** below every time you work on this.
> For detailed procedures and design, see the individual documents in the [document map](#7-document-map).

_Last updated: 2026-07-15 · branch `feat/tenstorrent-npu-backend`_

---

## 1. Status board (at a glance)

**Goal**: run the ReMEmbR pipeline (captioning → vector memory → QA) end-to-end on the **BOS NPU** with 3B models
(Qwen2.5-VL captioner + Llama-3.2-3B LLM), and evaluate it on the paper's NaVQA/CODa.

| Milestone | Status |
|---|---|
| Unified NPU inference server (LLM+VLM, HTTP) | ✅ Done |
| NonAgent `tt/` backend + JSON-constrained decoding | ✅ Done |
| NPU captioner (Qwen2.5-VL) | ✅ Done |
| end-to-end smoke test | ✅ Done |
| Original paper data acquired (data.csv + CODa seq0) | ✅ Done |
| ROS-free CODa→.pkl extraction | ✅ Done (8213 frames) |
| Captioning pipeline (checkpoint/resume) | ✅ Done (resume verified on a full run) |
| Server stabilization (clean shutdown) | ✅ Done |
| Slowdown fix ① Inspector | ✅ Done (`TT_METAL_INSPECTOR=0`) — eliminated the 19→2.1 it/s decay |
| Slowdown fix ② Tracy build | ✅ Done — swapped in the default build (`ENABLE_TRACY=OFF`, Tracy symbols 705→0). Measured: RSS +33MB/request → **+0~1MB**, latency +1.0s/request → **flat 36s** |
| **Question JSON generation** (form_question_jsons) | ⚠️ 30 questions generated — **must be regenerated** after the captions are rebuilt (the optimal context is extracted from the captions) |
| Server temperature support | ✅ Done (`b2dab02`) — resolved eval's infinite retry loop. Measured: temp=0 identical / 0.7 differs every time |
| **Caption regeneration** (quality consistency) | 🔄 **109/272** — resumed after the Tracy rebuild, **flat 36s/caption** (zero leak) |
| **Evaluation run** (eval.py, optimal+tt/) | ⬜ After caption regeneration |
| Multi-frame captions (paper fidelity) | ⬜ Optional |

### Measured hardware specs (`tt_server/npu_info.py`)

| Item | Value |
|---|---|
| arch / devices | `BLACKHOLE` / 1 device (`/dev/bos/0`, module `bos` 2.3.0) |
| **DRAM** | **12 GiB** = 4 GiB × **3 banks** (`dram_grid_size (3,1)`) |
| Tensix compute grid | **5×4 = 20 cores** (harvesting mask `0x0` = all cores healthy) |
| L1 / worker core | 1.5 MB (eth cores 512 KB) |
| PCIe | `0000:01:00.0` Co-processor (Synopsys DWC) |

> The **12 GiB** figure is the basis for "3B+3B can be co-resident, 8B can't." Using 8B during evaluation
> would require loading a single model per stage (`TT_ENABLE_VLM=0`).

---

## 2. What works today (verified)

- **NPU server**: Llama-3.2-3B + Qwen2.5-VL-3B co-resident, single worker + queue serialization, `/health` `/caption` `/v1/chat/completions`.
- **JSON-constrained decoding**: NonAgent gets JSON responses that parse 100% of the time, no retry loop (~10–12s/query).
- **Captioning**: single CODa frame → proper caption (no refusals), stable across 3 consecutive requests.
- **Data**: `data.csv` (210 questions) + 8213 CODa seq0 `.pkl` files (`/home/user/coda_raw/coda_data/0/`).
- **Board recovery**: cleaning up zombie processes from a wedged state → restart recovered it successfully (without a reboot).

## 3. In progress / next steps

1. 🔄 **Waiting for caption regeneration to finish** — everything unified at 256 tokens (~2.9h). On completion it writes the final JSON and auto-deletes the partial.
2. **Regenerate the question JSONs** — the captions changed, so `form_question_jsons.py` has to be re-run to refresh the optimal context.
3. **Run the evaluation** — `eval.py --model optimal+tt/Llama-3.2-3B-Instruct --sequence_id 0 ...` (temperature defaults to 0.7).
4. (Optional) Improve paper fidelity with **multi-frame captions** — procedure in [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 4-1.

## 4. Known issues / gotchas

| Issue | Status / response |
|---|---|
| **Server slowdown** (two causes) | **① tt-metal Inspector** (debug, ON by default): a YAML `flush()` per workload → ~5,000 synchronous I/O calls per caption, accumulating 153MB / 5.34M lines → 19→2.1 it/s. Fixed with `TT_METAL_INSPECTOR=0` ([8be8bc6]).<br>**② Tracy profiling build**: surfaced once Inspector was off (84→50 it/s). Over 12 runs of the same image, **RSS +33MB/request and latency +1.0s/request, growing linearly**. `build -> build_Release_tracy` (`ENABLE_TRACY=ON`), so Tracy kept accumulating zones. **It's a compile-time switch, so env vars can't turn it off** → ✅ swapped in the default build (`./build_metal.sh`, TRACY OFF) ([458bf09], symbols 705→0). Measured: **flat 36s, RSS +0MB**. Rollback: `ln -nsf build_Release_tracy build`. |
| **Caption sentences truncated** | Truncated at the server's default max_gen_len=128. Fixed by making the captioner send 256 ([8caa7c9]). Takes effect after a restart. |
| **Board wedged** | Cause = force-killing during captioning/compilation (especially mashing Ctrl-C). Added shutdown cleanup logic ([f9bc7bf]). **Never use `-9`.** Recovery is [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) section 6. |
| **Reboot constraint** | This PC only detects the NPU if the NPU is powered on before boot. When wedged, try process cleanup + restart before rebooting. |
| **Device node = `/dev/bos/0`** | Not Tenstorrent (there's no `/dev/tenstorrent`). Check with `fuser -v /dev/bos/0`, `lsmod \| grep bos`. |
| ~~**eval infinite retry**~~ | ✅ **Fixed** ([b2dab02]). Cause = the server ignored temperature and ran greedy → eval's retries produced the same answer every time. Fixed so the server honors OpenAI's `temperature` (0 keeps greedy, >0 samples from the grammar-allowed tokens). The original trigger — "3B answers `time=None` for time questions" — is still there, but retries now make progress. |
| **3B quality** | JSON validity is guaranteed, but grounding hits 3B's limits — there are cases where it can't fill the field matching the question type (absorbed by retries). **See design decision 5 for why we're not moving up to 8B** (co-residency is required for robot deployment). |
| ~~**Mixed caption quality**~~ | 🔄 Being resolved by the regeneration in progress. (The old mixed set is kept as `*.MIXED-BACKUP.json`) |
| **Fidelity deviations** | Captioner VILA→Qwen, LLM 8B→3B (unavoidable for the port), frames 6→1 (a compromise). See the fidelity table in [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md). |
| **venv dependency trap** | If a new package pulls `langchain-core` up to 0.3 or `numpy` to 2.x, the stack breaks. Respect the pins ([RUNBOOK](../tt_server/RUNBOOK.md) 1-3). |

## 5. Key design decisions (why we did it this way)

- **Split the two venvs behind HTTP** — avoids version conflicts between ttnn/transformers (server) and langchain/pymilvus (remembr).
- **Adopted NonAgent** (not ReMEmbRAgent) — tool-calling is unreliable at 3B. A single JSON call suits the NPU.
- **Keeping the LLM at 3B (not going to 8B)** — if you only look at evaluation, the VLM isn't needed at eval time, so you
  could load 8B on its own with `TT_ENABLE_VLM=0` and get better quality (8B only fits in 12GiB of DRAM by itself). **But the end
  goal is robot deployment**, and while driving, caption generation and QA happen together, so **both models must be co-resident.**
  8B+VLM does not fit in 12GiB → **3B+3B is the only deployable configuration.** Time-separated loading is
  an eval-only optimization that can't be used in real deployment, so **we evaluate with the same 3B configuration we deploy** (keeping the evaluation representative).
- **The server honors temperature** — the original eval runs its retry loop on the premise that retrying at
  `temperature=0.7` (the default) yields a different answer. If the server is hard-wired to greedy, that contract breaks and the loop
  never ends. Instead of patching the eval code, we made **the server keep the contract** (temperature=0 stays greedy as before).
- **JSON-constrained decoding** — replaces Ollama's `format="json"`. Prevents 3B's infinite retries and nonsense output.
- **Captioner = HTTP client** — the VLM is not loaded directly into the remembr process (dependency / device-ownership conflicts).
- **Removed `<video>`, video→image prompt** — a VILA leftover. It triggered refusals on Qwen with still frames.
- **ROS-free extractor** — the ROS dependency in the original `preprocess_coda.py` is only for visualization. Plain-text pose + PNGs alone produce the same .pkl.
- **Checkpointing (JSONL append)** — long captioning runs (~70–95 min) resume without loss if interrupted.
- **Clean shutdown + ignore signals during teardown** — fixes device close being skipped because the worker was a daemon, and prevents wedging from mashed Ctrl-C.

## 6. Changelog

### 2026-07-15
- `458bf09` **Second slowdown cause = Tracy profiling build**: surfaced once Inspector was off (84→50 it/s).
  Over **12 runs of the same image**, RSS +33MB/request and latency +1.0s/request grew **linearly** (same input, so it's accumulation, not caching).
  `build -> build_Release_tracy` (`ENABLE_TRACY=ON`, 705 Tracy symbols), so Tracy kept collecting every zone
  without `TRACY_ON_DEMAND`. **It's a compile-time switch, so env vars can't turn it off** → rebuilt with `./build_metal.sh` (OFF by default, 7 min).
  **Verified**: Tracy symbols 705→0, latency **flat at 36.0s across 8 runs**, RSS **+0MB**. 272 captions ~13h → **~2.9h**.
  No tt-metal source changes; rollback is `ln -nsf build_Release_tracy build`.
- `b2dab02` **Server honors temperature → eval's infinite retry resolved**: `chat_completions` never read
  `temperature`; the JSON path used masked argmax and the non-JSON path used `SamplingParams(temperature=0.0)`, so it was **hard-wired to greedy**.
  The original eval assumes that retrying with `--temperature 0.7` produces a different answer, so under greedy the retries loop forever.
  temperature is now threaded through `chat_completions → generate() → sampler`, and `_masked_argmax` became `_masked_sample`
  (temp≤0 = greedy as before, >0 = sample from the grammar-allowed tokens). The grammar constraint is unchanged, so JSON validity is unaffected.
  **Measured**: temp=0 → 1 unique response across 3 requests; temp=0.7 → 3 (both schema-valid).
- 🔄 **Caption regeneration started**: regenerating everything to clear the mixed quality (128/256 tokens intermixed). The old set is kept as `*.MIXED-BACKUP.json`.
  Early measurements: consistent length around \~1150 characters, vision at **83\~182 it/s**, one VisionBlock per caption — confirms all the earlier fixes are working.
- `8aea788` **Fixed eval optimal-path bug**: `TextMemory.memory_to_string` iterated character by character over the optimal
  context (which is already a string) and blew up on `doc.time` (→ eval retry infinite loop). Now a string passes straight through.
- ✅ **Question JSON generation**: `human_qa.json` with 30 questions (position 12 / time 7 / text 5 / binary 4 / duration 2).
  eval skips text, so **25 questions get scored**. The remaining seqs have no captions and are skipped automatically.
- ⛔ **Evaluation started → blocked**: the server ignored temperature (greedy), so eval.py's retry loop ran forever. Waiting on a decision about direction.
- `9fe023c` **Fixed npu_info**: the device methods we'd guessed at don't actually exist, so everything came back `(n/a)` → now it uses only
  `open_mesh_device` and the real MeshDevice API. MeshDevice has no API for DRAM size, so that's read from the **SoC descriptor YAML**.
  ttnn's loguru + C++ logs are **suppressed via fd redirection** (visible with `-v`). → **Got real measured specs (DRAM 12GiB, etc.)**.
- `4f81c2e` **Removed duplicate prefill**: the demo (`vision_demo.py`) **deliberately prefills twice** via `compile_prefill` + `inference_prefill`
  (to measure compile vs. inference time separately = a benchmarking device), but on the server the cache is warm after the first request,
  so the second was pure duplication. The first result went into the unused `masks` for the VLM, and for the LLM it **wasn't even assigned to a variable**.
  Both VLM and LLM now prefill once. Measured: VisionBlock per caption 2→**1**, ~76s→**~38s**.
- ✅ **seq0 captioning finished**: 272 captions generated (`captions_tt-qwen25-vl-3b_3_secs.json`). Checkpoint resume verified in practice (interrupt, then continue → partial auto-deleted on completion).
- `8be8bc6` **Slowdown root-caused & fixed**: tt-metal's **Inspector** (debug, ON by default) does a YAML `flush()` per mesh workload → ~5,000 synchronous disk I/O calls per caption, accumulating 153MB / 5.34M lines in a single run → the bigger the file, the more the flush blocks → vision block **decays monotonically from 19 to 2.1 it/s**. Disabled via tt-metal's own switches in `run_server.sh` (`TT_METAL_INSPECTOR=0`, `_RPC=0`) — **no tt-metal source changes**. Measured: **holds steady at ~42 it/s**, no YAML produced. Also reverted the earlier misdiagnosis fix (`ttnn.deallocate` in `describe()`).
- `f9bc7bf` **Server stabilization**: join the worker on shutdown to guarantee `close_mesh_device()`, plus ignore SIGINT/SIGTERM during teardown (prevents wedging from mashed Ctrl-C).
- `804bcf6` **eval import**: lazy-import `ChatNVIDIA` (NIM) → eval runs without the NIM package.
- `8caa7c9` **Captioner**: import fallback (both the inner and outer paths), and sending `max_gen_len` (default 256) fixes the sentence truncation.
- `16a253e` **Captioning pipeline**: `--captioner {vila,tenstorrent}` selection + VILA lazy import, neutralized the prompt from video→image, JSONL checkpoint/resume, changed early-exit to key off "final JSON exists".
- `334c0b4` **ROS-free extractor**: `preprocess_coda_rosfree.py` — pose txt + cam0 PNG → .pkl.
- `c5cc220` **Docs**: added PAPER_EVAL_RUNBOOK; added the langgraph/opencv pins and shutdown procedure to RUNBOOK.
- (ops) Acquired the original paper data: downloaded `data.csv`; downloaded, extracted, and pulled 8213 .pkl files out of CODa seq0 (42GB).
- (ops) Tidied up venv dependencies: installed `langgraph==0.2.40`/`langsmith==0.1.147`, `numpy<2`/`opencv-python-headless<5` (avoiding pin conflicts).
- (diagnosis) Root-caused the wedged board: it's a **`/dev/bos`** chip, not `/dev/tenstorrent`. Force-killing during captioning was the cause.

### 2026-07-13
- `05fff77` Removed the VILA `<video>` token, per-frame captions. Strengthened the smoke test.
- `7d96cbc` LLM JSON-constrained (grammar) decoding.
- `165e846` Hardened NonAgent NPU end-to-end.
- `8f9db90` end-to-end smoke test.
- `bb169e2` Unified NPU inference server (Llama+Qwen).
- `d29661c`/`ef0729d` NonAgent `tt/` backend + NPU captioner.

---

## 7. Document map

| Document | Contents |
|---|---|
| **[DEV_STATUS.md](DEV_STATUS.md)** (this document) | Status, issues, decisions, changelog (start here) |
| [TENSTORRENT_INTEGRATION.md](TENSTORRENT_INTEGRATION.md) | Architecture, design decisions, debugging log |
| [PAPER_EVAL_RUNBOOK.md](PAPER_EVAL_RUNBOOK.md) | Full procedure for the paper's NaVQA/CODa evaluation + recovery |
| [../tt_server/README.md](../tt_server/README.md) | Server structure, requirements, env |
| [../tt_server/RUNBOOK.md](../tt_server/RUNBOOK.md) | Server startup, venv setup, troubleshooting, shutdown |
