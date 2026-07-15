# ReMEmbR on Tenstorrent NPU

All the glue code needed to run ReMEmbR on a Tenstorrent board lives in **this
repo**. The tt-metal SDK is used as-is (imported at runtime via `PYTHONPATH`);
nothing is copied into or modified inside the tt-metal source tree.

See also: [../docs/DEV_STATUS.md](../docs/DEV_STATUS.md) (진행 상태), [RUNBOOK.md](RUNBOOK.md)
(기동/venv/종료), [../docs/PAPER_EVAL_RUNBOOK.md](../docs/PAPER_EVAL_RUNBOOK.md) (논문 평가),
[../docs/TENSTORRENT_INTEGRATION.md](../docs/TENSTORRENT_INTEGRATION.md) (아키텍처).

## Layout of what was added

```
remembr/
├── tt_server/
│   ├── tt_infer_server.py   # unified NPU inference server (LLM + VLM)
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

> **HW note**: 이 장비는 **BOS 칩**(Tenstorrent 아님). device 노드 `/dev/bos/0`, 모듈 `bos`.
> 점유 확인: `fuser -v /dev/bos/0`, `lsmod | grep bos`.

## Architecture

```
 remembr (its own venv)                     tt_server/tt_infer_server.py (tt-metal venv, owns the NPU)
   NonAgent ── HTTP /v1/chat/completions ─────► Llama-3.2-3B  ┐  one process, device opened once,
   Captioner ─ HTTP /caption ─────────────────► Qwen2.5-VL-3B ┘  both models co-resident,
                                                                 all NPU work serialized by a queue
```

A single process owns the board (ttnn opens it exclusively) and hosts both
models. A single worker thread + queue serializes every prefill/decode, so
simultaneous `/caption` and `/v1/chat/completions` requests are safe.

## Requirements

- **Two venvs.** The server runs in tt-metal's `python_env` (ttnn, transformers).
  remembr runs in its own venv (langchain, pymilvus, sentence-transformers) —
  the versions conflict, which is why they are split behind HTTP.
  - remembr venv pins (newer langchain drops `ChatOllama`):
    `langchain-core==0.2.40 langchain-community==0.2.16 langchain-openai==0.1.25`
    plus `pillow requests pymilvus langchain-huggingface==0.0.3 sentence-transformers==3.1.0`.
  - server extra deps (installed into tt-metal venv): `fastapi uvicorn[standard] lm-format-enforcer`.
- **HF access.** `meta-llama/Llama-3.2-3B-Instruct` is gated — set `HF_TOKEN`
  (run_server.sh reads it from ~/.bashrc). `Qwen/Qwen2.5-VL-3B-Instruct` is open.
- **Milvus** running (docker) for the memory DB.

## Run

```bash
# 1) start the NPU server (tt-metal venv)
bash remembr/tt_server/run_server.sh
#    wait for: {"ready":true,"llm":true,"vlm":true}  at http://localhost:8100/health

# 2) run remembr against it (remembr venv)
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py path/to/frame.jpg
```

## Env vars (server)

| var | default | note |
|-----|---------|------|
| `TT_LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM (agent reasoning) |
| `TT_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM (captioning) |
| `TT_ENABLE_LLM` / `TT_ENABLE_VLM` | `1` / `1` | load one at a time for the fallback |
| `TT_LLM_MAX_SEQ_LEN` / `TT_VLM_MAX_SEQ_LEN` | `1024` | raise for longer context (uses more DRAM) |
| `TT_TRACE_REGION_SIZE` | `15000000` | per-model trace buffer |
| `TT_PORT` | `8100` | |

remembr side: `TT_LLM_BASE_URL` (default `http://localhost:8100/v1`),
`TT_VLM_URL` (default `http://localhost:8100/caption`),
`TT_VLM_TIMEOUT` (client timeout, default 900s — raise for the slow first compile),
`TT_VLM_MAX_GEN_LEN` (max caption tokens, default 256 — server default is 128 which truncates).

## JSON-constrained decoding (format="json" equivalent)

`NonAgent` needs strict JSON. Ollama guaranteed this with `format="json"`
(grammar-constrained decoding); free greedy decoding on the NPU does not, so a
small 3B would emit code fences / ramble and `NonAgent.query` looped forever
(greedy → identical retries).

The server replicates it. When a request sets OpenAI `response_format`, the LLM
runs **host-side grammar-constrained decoding** via `lm-format-enforcer`:
`decode_forward_text(sampling_params=None)` returns raw host logits, which are
masked to the grammar-allowed token ids before argmax (`_masked_argmax`).

- `response_format={"type":"json_object"}` → any valid JSON.
- `response_format={"type":"json_schema","json_schema":{"schema":{...}}}` →
  constrain to the schema. `NonAgent`'s `tt/` branch sends a schema with the
  required keys, `maxLength` on `text`, and `additionalProperties:false` — this
  forces the required keys, caps free text, and forbids a reasoning field so the
  3B can't ramble past the token budget. Result: **100% parseable JSON, no retry
  loop, grounded answers, ~10-12s/query** on Llama-3.2-3B.

Cost: JSON mode reads the full-vocab logits to host each step (vs. on-device
argmax), so it is slower than plain decoding — fine for short JSON answers.
Non-JSON requests keep the fast on-device greedy path.

## Notes / known limits

- **DRAM budget.** Measured via `npu_info.py`: **12 GiB = 4 GiB x 3 banks**
  (arch BLACKHOLE, Tensix 5x4 = 20 cores, L1 1.5MB/core). Llama-3.2-3B + Qwen-3B
  fit together at `max_seq_len=1024`. An 8B LLM does **not** co-fit — use the
  time-separated fallback (`TT_ENABLE_VLM=1 TT_ENABLE_LLM=0` for the caption
  phase, then swap) or a smaller model.
- **Inspector is off.** `run_server.sh` sets `TT_METAL_INSPECTOR=0` /
  `TT_METAL_INSPECTOR_RPC=0`. tt-metal's Inspector defaults to ON and flushes a
  YAML record per mesh workload (~5k synchronous disk flushes per caption), which
  dragged the VLM from 19 to 2.1 it/s over one run. Leave it off unless debugging.
- **Single prefill.** The upstream demo runs prefill twice (compile pass + timed
  pass) to report compile-vs-inference ms; the server keeps one, since its cache
  is warm after the first request.
- **3B answer quality.** JSON validity is now guaranteed; answer grounding still
  depends on the model. The 3B is mostly grounded but occasionally copies the
  prompt example — the 8B is stronger (use the DRAM fallback if needed).
- **Shutdown.** Stop with a single `Ctrl-C` (or SIGTERM) and wait — the server
  joins the worker so `close_mesh_device()` runs and the board is released
  cleanly, and it ignores extra SIGINT/SIGTERM during teardown (mashing Ctrl-C is
  safe). **Never `kill -9`** — SIGKILL skips teardown and wedges the board.
- **Per-caption slowdown.** `describe()` deallocates its per-request device
  tensors so latency doesn't creep up over a long captioning run. If a run still
  slows down, restart the server (resets device state) and resume from the
  captioner checkpoint. (The deallocation fix still wants a fresh-run re-check.)
