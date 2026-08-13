#!/usr/bin/env bash
# Launch the ReMEmbR NPU inference server (Llama-3.2-3B + Qwen2.5-VL-3B) on a
# Tenstorrent board. This file lives in the remembr repo but RUNS inside the
# tt-metal python_env and imports the tt-metal SDK via PYTHONPATH (the tt-metal
# source tree is left untouched — nothing is copied into remembr).
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/server_preflight.sh"
if tt_server_preflight; then
  : # no existing server/device holder; keep FD 9 locked through exec below
else
  preflight_status=$?
  if [[ "$preflight_status" -eq 10 ]]; then
    exit 0 # an existing server/launcher is healthy or already starting
  fi
  exit "$preflight_status"
fi

export TT_METAL_HOME="${TT_METAL_HOME:-/home/user/tt-metal}"
# env_set.sh does `export TT_METAL_HOME=$(pwd)`, so it must be sourced FROM the
# tt-metal root or it clobbers TT_METAL_HOME with the caller's cwd -> the SoC
# descriptor lookup then fails ("$TT_METAL_HOME/tt_metal/soc_descriptors/... does
# not exist"). cd there first so the server works regardless of where it's launched.
cd "$TT_METAL_HOME"
# env_set.sh for the environment variables (ARCH_NAME, PYTHONPATH, ...), then this
# repo's own venv for the interpreter. It activates one of tt-metal's venvs on the way
# through; the activation below replaces that.
#
# One venv now serves every mode. It used to be a choice -- TT_PYTHON_ENV=tf457 picked a
# venv with transformers 4.57 for qwen3_vl, unset picked 4.53 for everything else -- and
# each mode switch meant remembering which. 4.57 runs all four modes, so there is nothing
# left to choose. That venv also lived under tt-metal, which is not ours to write to.
# shellcheck disable=SC1091
source "$TT_METAL_HOME/env_set.sh"

REMEMBR_VENV="${REMEMBR_VENV:-$(cd "$SCRIPT_DIR/.." && pwd)/.venv}"
if [ ! -x "$REMEMBR_VENV/bin/python" ]; then
  echo "No venv at $REMEMBR_VENV. Create it:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  (see requirements.txt)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REMEMBR_VENV/bin/activate"

# Non-interactive shells skip ~/.bashrc, so pull the HF token from it explicitly
# (needed for gated meta-llama weights). HF_HOME is left at its default so the
# already-downloaded model cache (~/.cache/huggingface) is used.
eval "$(grep -E '^export (HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' ~/.bashrc 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

# Startup mode (this board exposes only ~12GB DRAM: 3 banks, so only one model
# combination fits at a time). Set TT_MODE to load a named combination directly
# instead of loading the legacy default and switching via POST /mode afterward:
#   TT_MODE=3b+3b      Llama-3.2-3B + Qwen2.5-VL-3B  (default combination)
#   TT_MODE=caption-3b Qwen2.5-VL-3B only (captioning, no LLM)
#   TT_MODE=llm-8b     Llama-3.1-8B only (no VLM)
#   TT_MODE=caption-8b Qwen3-VL-8B only (captioning, no LLM)
# An unknown TT_MODE fails immediately with a clear error, before the device is
# touched -- it does not hang or wedge the board.
# Leave TT_MODE unset to keep the old behavior: TT_ENABLE_LLM/VLM + TT_LLM_MODEL/
# TT_VLM_MODEL below pick the models individually (still the default combination).
export TT_MODE="${TT_MODE:-}"

# Models + memory-fit settings (used when TT_MODE is unset).
export TT_LLM_MODEL="${TT_LLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
export TT_VLM_MODEL="${TT_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export TT_LLM_MAX_SEQ_LEN="${TT_LLM_MAX_SEQ_LEN:-1024}"
export TT_VLM_MAX_SEQ_LEN="${TT_VLM_MAX_SEQ_LEN:-1024}"
export TT_TRACE_REGION_SIZE="${TT_TRACE_REGION_SIZE:-15000000}"
export TT_ENABLE_LLM="${TT_ENABLE_LLM:-1}"
export TT_ENABLE_VLM="${TT_ENABLE_VLM:-1}"

# tt-metal's Inspector is a DEBUG facility that defaults to ON (rtoptions.hpp:
# InspectorSettings::enabled = true). It appends a YAML record per mesh workload
# and fstream::flush()es on every one — ~5k synchronous disk flushes per caption
# here, growing generated/inspector/mesh_workloads_log.yaml to 153MB / 5.3M lines
# over one captioning run. As that file grows, each flush blocks longer and the
# VLM slows down progressively (measured: 19 -> 2.1 it/s on the vision blocks).
# It is pure debug tooling we never read, so turn it off for the server via
# tt-metal's own switches (no tt-metal source changes).
export TT_METAL_INSPECTOR="${TT_METAL_INSPECTOR:-0}"
export TT_METAL_INSPECTOR_RPC="${TT_METAL_INSPECTOR_RPC:-0}"

# tt-metal waits for the device by polling. With no timeout set (rtoptions.cpp:290
# defaults to 0) it takes the branch in system_memory_manager.cpp:69 that has no yield
# and no exit: `do { read_core(...); } while (ptr unchanged);`. Confirmed by gdb as
# where this board wedges -- in fetch_queue_reserve_back (:393), i.e. the host cannot
# even enqueue because the device's prefetcher stopped consuming. It burns 100% of a
# core forever holding the GIL, so /health goes silent and Ctrl-C is ignored, with no
# error ever printed. Setting this turns that into a catchable
# TT_THROW("TIMEOUT: device timeout, potential hang detected"). See #70.
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}"

exec python "$SCRIPT_DIR/tt_infer_server.py"
