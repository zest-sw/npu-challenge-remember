# Original Paper Data (NaVQA/CODa) Test Execution Guide — NPU

The full procedure for running the **NaVQA benchmark** from the ReMEmbR paper on our
Tenstorrent NPU stack (Qwen2.5-VL captioner + Llama-3.2-3B LLM). Data acquisition →
preprocessing → captioning → question generation → evaluation, all in one document.

> ⚠️ **Read this first — section 5, "NPU safety rules".** If you kill the server the wrong
> way during captioning, the board ends up wedged and **needs a reboot**. This has actually
> happened to us once.

---

## 0. The pipeline at a glance

```
CODa seq .zip ──(selective unzip)──► 2d_rect/cam0 + poses/dense
      │                                   │
      │                    preprocess_coda_rosfree.py (no ROS needed)
      ▼                                   ▼
   coda_data/{seq}/{ts}.pkl  ──► preprocess_captions.py --captioner tenstorrent
                                         │  (NPU Qwen2.5-VL, server required)
                                         ▼
                    data/captions/{seq}/captions/captions_<name>_<sec>_secs.json
                                         │
        data/navqa/data.csv ──► form_question_jsons.py
                                         ▼
                    data/questions/{seq}/human_qa.json
                                         │
                                         ▼
   eval.py --model optimal+tt/Llama-3.2-3B-Instruct  (NPU LLM, server required)
                                         ▼
                                    out/ (scored results)
```

---

## 1. Current acquisition status (as of 2026-07)

| Asset | Location | Status |
|---|---|---|
| NaVQA answer key `data.csv` (210 questions) | `remembr/data/navqa/data.csv` | ✅ Acquired |
| NaVQA question segments | `remembr/data/navqa/{0,3,4,6,16,21,22}/qa_unfilled.json` | ✅ Included in repo |
| CODa seq 0 original zip (42GB) | `/home/user/coda_raw/0.zip` | ✅ Downloaded & verified |
| 8213 cam0 images + pose | `/home/user/coda_raw/seq0/` | ✅ Extracted |
| 8213 `.pkl` frames | `/home/user/coda_raw/coda_data/0/` | ✅ Extracted |
| ROS-free extractor | `remembr/scripts/preprocess_coda_rosfree.py` | ✅ |
| **seq0 captions** | `remembr/data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json` | 🔄 Regenerating (quality consistency) |
| **seq0 30 questions** | `remembr/data/questions/0/human_qa.json` | ⚠️ **Must be regenerated** after captions are regenerated |

**What's left**: regenerate captions (quality consistency) → regenerate question JSON →
**4-3 evaluation**. The blocker caused by unsupported temperature is now resolved (the
server honors the OpenAI `temperature` parameter).

### Fidelity vs. the paper (what matches and what differs)

| Item | Paper original | Ours (NPU) | Fidelity |
|---|---|---|---|
| Dataset / answer key | NaVQA · CODa | Same (seq 0) | ✅ |
| `seconds_per_caption` | 3 seconds | 3 seconds | ✅ |
| Caption prompt | Campus prompt (video) | Wording preserved, only `video→image` neutralized | ✅ Nearly faithful |
| Text embedding | mxbai-embed-large-v1 | Same | ✅ |
| Evaluation & scoring (`eval.py`) | Same | Same routing (`tt/` backend) | ✅ |
| **Captioner model** | VILA 1.5 (13B) | Qwen2.5-VL 3B | ⚠️ Different (NPU port) |
| **Frames per caption** | ~6 frames (video window) | **1 frame (snapshot)** | ⚠️ Deviation (speed/stability trade-off) |
| **Inference LLM** | GPT-4o / llama3.1 8B | Llama-3.2-3B | ⚠️ Different (NPU port) |

> The captioner and LLM model differences are **inherent to the NPU port** (small local
> models), so they are unavoidable. The **frame count (6→1)**, on the other hand, is a
> trade-off we chose, so if you want to improve fidelity to the paper, start here (see 4-1).
> The prompts preserve the original text as closely as possible.

---

## 2. Data acquisition (reproduction procedure for adding other sequences)

seq 0 is already done. If you need other sequences (3, 4, 6, 16, 21, 22), repeat the steps below.

### 2-1. Answer key (once only)
```bash
curl -sL -o /home/user/remembr/remembr/data/navqa/data.csv \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/remembr/main/remembr/data/navqa/data.csv
```

### 2-2. Download CODa sequences (mind the size)
zip size per sequence you may need: **0=42GB, 3=72GB, 4=46GB, 6=68GB, 16=92GB, 21=111GB, 22=78GB** (total ≈509GB).
```bash
SEQ=0
curl -C - -L --retry 5 -o /home/user/coda_raw/${SEQ}.zip \
  https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/${SEQ}.zip
```

### 2-3. Selectively extract only what's needed (cam0 + pose)
The full zip is mostly LiDAR (3d_raw etc.), but **we only need cam0 + pose**.
```bash
cd /home/user/coda_raw
unzip -q -o ${SEQ}.zip "2d_rect/cam0/${SEQ}/*" "poses/dense/${SEQ}.txt" -d seq${SEQ}/
```

### 2-4. ROS-free `.pkl` extraction
The original `preprocess_coda.py` is heavy because it depends on ROS (rospy/cv_bridge/tf).
Our extractor reads only the plain-text pose txt + cam0 PNGs and produces identical `.pkl`
files (no ROS required).
```bash
cd /home/user/remembr
source ~/remembr_venv/bin/activate
python remembr/scripts/preprocess_coda_rosfree.py \
  --seq_id ${SEQ} \
  --coda_root /home/user/coda_raw/seq${SEQ} \
  --out_dir  /home/user/coda_raw/coda_data/${SEQ} \
  --frame_skip 1
```
- Output: `{ts}.pkl` = `{cam0(BGR), position, rotation(xyzw), timestamp}`.
- pose rows ↔ cam0 frames correspond 1:1 by index (verified).

---

## 3. Starting the NPU server (terminal A, tt-metal environment)

```bash
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 \
  bash /home/user/remembr/tt_server/run_server.sh
```
- `TT_LLM_MAX_SEQ_LEN=2048` is **required** (the NonAgent prompt pads up to 2048).
- Wait until `Application startup complete.` appears.
- Check: `curl -s http://localhost:8100/health` → `{"ready":true,"llm":true,"vlm":true}`
- **If the server fails to come up and you get `Device 0 init: failed to initialize FW! Try resetting the board.` → see the recovery procedure in section 6.**

---

## 4. Running the test (terminal B, remembr_venv, **inner** directory)

```bash
source ~/remembr_venv/bin/activate
cd /home/user/remembr/remembr           # inner: scripts use ./data relative paths
export OPENAI_API_KEY=local PYTHONPATH=/home/user/remembr
export TT_VLM_TIMEOUT=3600               # headroom for the first compile
```

### 4-1. NPU captioning
```bash
python scripts/preprocess_captions.py \
  --captioner tenstorrent \
  --seq_id 0 --seconds_per_caption 3 \
  --num-video-frames 1 \
  --captioner_name tt-qwen25-vl-3b \
  --data_path /home/user/coda_raw/coda_data \
  --out_path data/captions/0/captions
```
- **`--num-video-frames 1` = a stability/speed trade-off (⚠️ a fidelity deviation from the paper).**
  The paper captions each 3-second window from **~6 frames (video)**, whereas we caption each
  window from only **1 frame (snapshot)** → a **clear deviation** that reduces temporal
  information. On the other hand, the default (6 images) made the **first trace compile of the
  Qwen multi-image shape take 15+ minutes**, which caused client timeouts and a wedged board
  (this actually happened). We use 1 frame for reproducibility and stability, but **if you need
  paper-level fidelity**, raise the frame count (e.g. 3–6) and wait out the first compile (tens
  of minutes) — **never kill it partway through**.
- The caption prompt uses the paper's original text verbatim, neutralizing only the VILA video
  wording (remove `<video>`, "the few seconds of the video" → "the image"). Everything else
  ("Think step by step", the focus list) is exactly as in the paper.
- **Preventing truncated sentences**: the captioner sends `max_gen_len=256` (default, tunable via
  `TT_VLM_MAX_GEN_LEN`). With the server default of 128, verbose captions get cut off mid-way.
- **Checkpoint/resume**: every single caption is written immediately to
  `..._secs.json.partial.jsonl`. After an interruption (kill, server problem), **re-run the same
  command and it resumes** (already-done segments are skipped). On full completion it saves the
  final JSON and deletes the partial. To start over from scratch, delete the partial.
- **The progressive slowdown is fixed (there were two causes)**:
  **① the tt-metal Inspector** (debug, ON by default) flushed YAML on every mesh workload → ~5,000
  synchronous disk I/Os per caption, with the YAML growing to 153MB → vision dropped from 19 to
  2.1 it/s. `run_server.sh` disables it with `TT_METAL_INSPECTOR=0`.
  **② the Tracy profiler build** — revealed once ① was off (84→50 it/s). If tt-metal is built with
  `--enable-profiler`, Tracy keeps accumulating zones (+33MB RSS per request and +1.0s latency per
  request over 12 runs of the same image).
  **It can't be disabled via env, so you must rebuild with the default build** → [RUNBOOK 1-0](../tt_server/RUNBOOK.md).
  Measured after fixing both: **a steady 36 seconds/caption, zero RSS growth**.
  If it ever slows down again, the `32 x Qwen2_5_VLVisionBlock ... it/s` value in the server log is
  the best gauge.
- Result: `data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json`.
- Only the first caption is slow due to trace compilation; afterwards it's cached and fast.

### 4-2. Generating the question JSON (answer key + captions → human_qa.json)
```bash
python scripts/question_scripts/form_question_jsons.py \
  --caption_file captions_tt-qwen25-vl-3b_3_secs
```
- `./data/navqa/data.csv` + the captions above → `data/questions/0/human_qa.json`.
- Only sequences that have captions are processed; the rest are skipped automatically.
- ⚠️ **If you regenerate the captions, you must re-run this step too.** The question JSON embeds the
  "optimal context" extracted from the captions (the captions needed to answer), so when the
  captions change, that context has to be refreshed as well.

### 4-3. Evaluation
**Recommended — optimal (oracle) context + NonAgent** (fits within seq_len, no tool-calling needed):
```bash
python scripts/eval.py \
  --model optimal+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 \
  --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data \
  --postfix _0
```
**Full RAG (optional) — Milvus retrieval + agent** (3B tool-calling is unstable):
```bash
python scripts/eval.py \
  --model remembr+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data --postfix _0
```
- Results go to `./out/`. binary/position/time/duration are scored; text questions are skipped
  (original code behavior). For seq0, 5 text questions are excluded from the 30, leaving
  **25 questions scored**.
- ⚠️ **Avoid plain `tt/...` (without a prefix)** — TextMemory feeds in the captions for the entire
  window, which exceeds seq_len on long questions.
- **Beware the retry loop**: if the 3B answers a time question with `type='text', time=None`,
  `evaluate_output` crashes at `abs(answer - None)` and the `while True` in
  `answer_squad_question` retries. For that retry to make progress, **a different answer must come
  out each time**, so do not set `--temperature` to 0 (the default 0.7 is recommended). The server
  now honors the OpenAI `temperature` parameter (note that 0 means greedy, which turns this into an
  infinite loop).

### Memory modes by model
| `--model` | Memory | Context | Suitability for the 3B on NPU |
|---|---|---|---|
| `optimal+tt/...` | Oracle context | Small (answer captions only) | ✅ Recommended |
| `remembr+tt/...` | Milvus RAG top-k | Limited | △ Weak tool-calling |
| `tt/...` | Entire TextMemory | **Overflow risk** | ✗ |

---

## 5. NPU safety rules (mandatory)

If a process dies without teardown while the device is running or compiling, the core, NoC, and ARC
state is never cleaned up and you end up in a **hardware deadlock (wedged)**. On restart you get
`failed to initialize FW` and recovery is a hassle (this PC also has the constraint that on reboot
the NPU must be powered on first to be detected). The rules:

1. **Shut the server down with a single `Ctrl-C` (or SIGTERM), then wait.** The server joins its
   workers and cleanly closes the device via `close_mesh_device()`. It's done when you see these
   logs:
   ```
   Shutting down: releasing the NPU cleanly ...
   NPU released cleanly. Safe to exit.
   ```
   **Additional Ctrl-C presses during teardown are ignored**, so mashing the key is safe (the old
   bug is fixed).
2. **Never use `kill -9` (SIGKILL).** This is the one thing that skips teardown and wedges the
   board. It's the only genuinely dangerous way to terminate.
3. **Killing the client (captioning) is safe** and it resumes from the checkpoint. But killing the
   client does not kill the server, so don't hastily `-9` the server too.
4. **Don't kill the first compile (the slow stretch) — wait it out.** Use `--num-video-frames 1` to
   keep this window short, and `TT_VLM_TIMEOUT=3600` to give the client a generous timeout.
5. Check device occupancy: `fuser -v /dev/bos/0` (empty result = free), `lsmod | grep bos` (usage 0 = unused).

---

## 6. Recovering a wedged board (when FW init fails)

Symptom: on server startup
```
Device 0: Timeout waiting for physical cores to finish
Device 0 init: failed to initialize FW! Try resetting the board.
```

### Diagnosis (⚠️ the device node is `/dev/bos` — not Tenstorrent)
```bash
fuser -v /dev/bos/0            # who is holding the device (empty result = free)
lsmod | grep bos              # bos module usage count (0 = nobody is using it)
pgrep -af tt_infer_server     # is a server process still around
uptime -p                     # did it reboot
# Definitive test for whether the board is actually dead (after bringing the server down):
#   python3 -m ttnn.examples.usage.run_op_on_device
#   -> tensor output means the board is fine; an FW init error means wedged
```

### Recovery order
1. **First check for and clean up zombie servers.** If a hung server is **not holding the device**
   (it's absent from `fuser`, or an op can open the device), it is **safe** to clean it up with
   `kill -9 <pid>` (there's no device operation to interrupt).
   → After cleanup, restarting the server alone often fixes it (this has happened for real).
2. **If the board really is wedged (FW init failure)** → a warm reset is needed. Constraints in this
   environment: `tt-smi` is not installed, sudo requires a password, and `/sys/bus/pci/rescan` is
   root-only, so `sudo <bos reset tool>` is only possible if you have a sudo account.
3. **If sudo isn't available → reboot.** But **this PC only detects the NPU if the NPU is powered on
   first**, so a plain `sudo reboot` may not be enough — you may need the correct power sequence
   (cut power → NPU on → PC on). If it's badly wedged, a warm reboot won't clear it and a cold power
   cycle is required.
4. Confirm recovery: `fuser -v /dev/bos/0` (free) + `python3 -m ttnn.examples.usage.run_op_on_device`
   (op runs normally).

> Constraints in this environment: **the `bos` reset tool is not installed, sudo requires a password,
> and `/sys/bus/pci/rescan` is root-only**. So "reset just the NPU from the command line" is only
> possible with a sudo account. Most of the time, **cleaning up the zombie server and restarting**
> resolves it (unless the board is genuinely dead).

---

## 7. Related files

| File | Role |
|---|---|
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa→.pkl extractor |
| `remembr/scripts/preprocess_captions.py` | Captioning (supports `--captioner tenstorrent`) |
| `remembr/scripts/question_scripts/form_question_jsons.py` | data.csv + captions → question JSON |
| `remembr/scripts/eval.py` | Evaluation (`tt/` backend routing) |
| `remembr/tt_server/run_server.sh` | NPU server startup (includes disabling the Inspector) |
| `remembr/tt_server/npu_info.py` | NPU spec dump (arch/DRAM/Tensix/L1) |
| `remembr/scripts/viz.py` | (Optional) Visualizes evaluation results overlaid on the driving video — not required for the pipeline |
| `docs/TENSTORRENT_INTEGRATION.md` | Architecture & design decisions |
