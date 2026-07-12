#!/usr/bin/env bash
# Launch the ReMEmbR NPU inference server (Llama-3.2-3B + Qwen2.5-VL-3B) on a
# Tenstorrent board. This file lives in the remembr repo but RUNS inside the
# tt-metal python_env and imports the tt-metal SDK via PYTHONPATH (the tt-metal
# source tree is left untouched — nothing is copied into remembr).
set -e

export TT_METAL_HOME="${TT_METAL_HOME:-/home/user/tt-metal}"
# shellcheck disable=SC1091
source "$TT_METAL_HOME/env_set.sh"          # activates python_env, sets PYTHONPATH

# Non-interactive shells skip ~/.bashrc, so pull the HF token from it explicitly
# (needed for gated meta-llama weights). HF_HOME is left at its default so the
# already-downloaded model cache (~/.cache/huggingface) is used.
eval "$(grep -E '^export (HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' ~/.bashrc 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

# Models + memory-fit settings (this board exposes only ~12GB DRAM: 3 banks).
export TT_LLM_MODEL="${TT_LLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
export TT_VLM_MODEL="${TT_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export TT_LLM_MAX_SEQ_LEN="${TT_LLM_MAX_SEQ_LEN:-1024}"
export TT_VLM_MAX_SEQ_LEN="${TT_VLM_MAX_SEQ_LEN:-1024}"
export TT_TRACE_REGION_SIZE="${TT_TRACE_REGION_SIZE:-15000000}"
export TT_ENABLE_LLM="${TT_ENABLE_LLM:-1}"
export TT_ENABLE_VLM="${TT_ENABLE_VLM:-1}"

exec python "$(dirname "$0")/tt_infer_server.py"
