<h1 align="center">ReMEmbR on an NPU</h1>

<p align="center">
<a href="docs/DEV_STATUS.md">Status</a> ·
<a href="docs/PAPER_EVAL_RUNBOOK.md">Paper eval</a> ·
<a href="tt_server/RUNBOOK.md">Run the server</a> ·
<a href="docs/TENSTORRENT_INTEGRATION.md">Architecture</a> ·
<a href="README.upstream.md">Upstream README</a>
</p>

A port of **[ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr)** (NVIDIA's long-horizon
spatio-temporal memory for robots) that runs the whole pipeline **on a local NPU** — no
cloud LLM, no discrete GPU.

Upstream captions with **VILA-1.5 (13B)** and reasons with **GPT-4o / llama3.1-8B**. Here
both models are small and local, co-resident on a single board:

| | Upstream | This fork |
|---|---|---|
| Captioner (VLM) | VILA 1.5 13B | **Qwen2.5-VL-3B** on the NPU |
| Reasoner (LLM) | GPT-4o / llama3.1 8B | **Llama-3.2-3B** on the NPU |
| Memory | Milvus + `mxbai-embed-large-v1` | same |
| Eval | NaVQA / CODa | same |

> The original project's README is preserved verbatim at
> **[README.upstream.md](README.upstream.md)** — setup, usage and evaluation as NVIDIA
> documents them. Everything below is specific to the NPU port.

## How it runs

```
 remembr (remembr_venv)                    tt_server/tt_infer_server.py (tt-metal python_env, owns the board)
   NonAgent ── HTTP /v1/chat/completions ────► Llama-3.2-3B  ┐  one process, device opened once,
   Captioner ─ HTTP /caption ────────────────► Qwen2.5-VL-3B ┘  both models co-resident,
                                                                every NPU op serialized by one worker queue
```

The two stacks need conflicting dependency pins (ttnn/transformers vs
langchain/pymilvus), and the device can only be held by one process, so the models live
behind a small HTTP server and remembr talks to it. See
[tt_server/README.md](tt_server/README.md) for the API and
[docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md) for why.

## Hardware

This board is a **BOS chip** (not Tenstorrent) driven through a tt-metal-compatible stack,
so logs say `BlackholeTTDevice` while the device node is `/dev/bos/0` and the kernel module
is `bos`. Measured with `python tt_server/npu_info.py`:

| | |
|---|---|
| arch | `BLACKHOLE`, 1 device |
| DRAM | **12 GiB** (4 GiB × 3 banks) |
| Tensix grid | 5×4 = 20 cores |
| L1 | 1.5 MB / worker core |

12 GiB is why the models are 3B: Llama-3.2-3B + Qwen2.5-VL-3B fit together, an 8B LLM
does not co-fit with the VLM — and the robot needs both resident at once.

## Quick start

```bash
# 1) NPU server (tt-metal python_env)
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 bash /home/user/remembr/tt_server/run_server.sh
curl -s localhost:8100/health          # {"ready":true,"llm":true,"vlm":true}

# 2) end-to-end smoke test (remembr_venv)
cd /home/user/remembr && source ~/remembr_venv/bin/activate
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py path/to/frame.jpg
```

First-time setup (venvs, pins, HF token, Milvus) and the **profiler-build check that
matters for performance**: [tt_server/RUNBOOK.md](tt_server/RUNBOOK.md).

> **Stop the server with a single `Ctrl-C` and wait** for `NPU released cleanly`.
> Never `kill -9` it — that skips the device teardown and wedges the board.

## Docs

| | |
|---|---|
| [docs/DEV_STATUS.md](docs/DEV_STATUS.md) | **Start here.** Status board, known issues, design decisions, changelog |
| [docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md) | Architecture, design decisions, debugging log |
| [docs/PAPER_EVAL_RUNBOOK.md](docs/PAPER_EVAL_RUNBOOK.md) | Reproducing the NaVQA/CODa evaluation on the NPU |
| [tt_server/README.md](tt_server/README.md) | Server layout, HTTP API, env vars |
| [tt_server/RUNBOOK.md](tt_server/RUNBOOK.md) | Setup, start/stop, troubleshooting, board recovery |

## What was added

```
tt_server/
├── tt_infer_server.py                    # unified NPU inference server (LLM + VLM)
├── run_server.sh                         # launcher (env, Inspector off)
└── npu_info.py                           # hardware/spec dump
remembr/agents/non_agent.py               # + 'tt/<model>' backend -> server
remembr/captioners/tenstorrent_captioner.py   # Captioner -> POST /caption
remembr/scripts/preprocess_coda_rosfree.py    # ROS-free CODa -> .pkl extractor
remembr/scripts/preprocess_captions.py    # + --captioner tenstorrent, checkpoint/resume
examples/tt_smoke_test.py                 # image -> caption -> memory -> query
```

The tt-metal SDK is used as-is (imported via `PYTHONPATH`); nothing is copied into or
modified inside its source tree.
