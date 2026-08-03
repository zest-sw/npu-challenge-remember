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
# Pick the tt-metal venv: env_set.sh <suffix> activates python_env_<suffix>.
# TT_PYTHON_ENV=tf457 -> python_env_tf457 (transformers 4.57.1, needed for the
# qwen3_vl caption-8b backend); unset -> default python_env (4.53).
# shellcheck disable=SC1091
source "$TT_METAL_HOME/env_set.sh" ${TT_PYTHON_ENV:+"$TT_PYTHON_ENV"}   # activates venv, sets PYTHONPATH

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
#   TT_MODE=caption-8b Qwen3-VL-8B only -- needs TT_PYTHON_ENV=tf457 and the
#                      weights downloaded (not wired for the default 4.53 venv)
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

exec python "$SCRIPT_DIR/tt_infer_server.py"
