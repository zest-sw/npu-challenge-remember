# ReMEmbR on Tenstorrent NPU

All the glue code needed to run ReMEmbR on a Tenstorrent board lives in **this
repo**. The tt-metal SDK is used as-is (imported at runtime via `PYTHONPATH`);
nothing is copied into or modified inside the tt-metal source tree.

## Layout of what was added

```
remembr/
├── tt_server/
│   ├── tt_infer_server.py   # unified NPU inference server (LLM + VLM)
│   ├── run_server.sh        # launcher (sets env, runs in tt-metal venv)
│   └── README.md            # this file
├── remembr/agents/non_agent.py          # + 'tt/<model>' backend branch (ChatOpenAI -> server)
├── remembr/captioners/tenstorrent_captioner.py   # Captioner -> POST /caption
├── remembr/captioners/__init__.py       # captioner_selector() factory
└── examples/tt_smoke_test.py            # end-to-end: image -> caption -> memory -> query
```

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
`TT_VLM_URL` (default `http://localhost:8100/caption`).

## Notes / known limits

- **DRAM budget.** This board exposes only ~12GB DRAM (3 banks). Llama-3.2-3B +
  Qwen-3B fit together at `max_seq_len=1024`. An 8B LLM does **not** co-fit — use
  the time-separated fallback (`TT_ENABLE_VLM=1 TT_ENABLE_LLM=0` for the caption
  phase, then swap) or a smaller model.
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

- **DRAM budget.** This board exposes only ~12GB DRAM (3 banks). Llama-3.2-3B +
  Qwen-3B fit together at `max_seq_len=1024`. An 8B LLM does **not** co-fit — use
  the time-separated fallback (`TT_ENABLE_VLM=1 TT_ENABLE_LLM=0` for the caption
  phase, then swap) or a smaller model.
- **3B answer quality.** JSON validity is now guaranteed; answer grounding still
  depends on the model. The 3B is mostly grounded but occasionally copies the
  prompt example — the 8B is stronger (use the DRAM fallback if needed).
