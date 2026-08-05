#!/usr/bin/env bash
# Run an eval the same way every time.
#
# Every flag below was forgotten at least once in one session, and each omission cost a
# run or made a log unreadable:
#
#   repo root        eval.py must run from here or `remembr.*` imports fail with
#                    ModuleNotFoundError (#44). The runbook's `cd remembr` is wrong.
#   --data_dir       the default ./data/ does not exist from the repo root
#   python -u        without it print() is block-buffered, so a live log shows nothing
#                    for minutes and looks hung
#   REMEMBR_TRACE=1  otherwise the log is raw prints with no structure
#   temperature      eval defaults to 0.7 on purpose; 0 makes retries byte-identical and
#                    pointless (AGENTS.md). Pass 0 only for a paired before/after, and
#                    never compare a 0 number against a 0.7 one
#   header           branch, commit and dirty count, so a log says which code produced it
#                    (#45: editing files mid-run silently measures something else)
#
# Usage:
#   bash remembr/scripts/run_eval.sh retrieve+tt/Llama-3.1-8B-Instruct human_qa
#   bash remembr/scripts/run_eval.sh optimal+tt/Llama-3.1-8B-Instruct human_qa_SUBSET --temperature 0
#   DRY_RUN=1 bash remembr/scripts/run_eval.sh remembr+tt/Llama-3.1-8B-Instruct human_qa
#
# Watch it from anywhere:
#   tail -F out/logs/latest.log
set -euo pipefail

MODEL=${1:?"usage: run_eval.sh <model> <qa_file> [extra eval.py args]"}
QA_FILE=${2:?"usage: run_eval.sh <model> <qa_file> [extra eval.py args]"}
shift 2

CAPTIONS=${CAPTIONS:-captions_tt-qwen25-vl-3b_3_secs}
CODA_DIR=${CODA_DIR:-/home/user/coda_raw/coda_data}
SEQUENCE=${SEQUENCE:-0}
SERVER=${TT_LLM_BASE_URL:-http://localhost:8100/v1}
HEALTH=${SERVER%/v1}/health

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# The board is shared and one process holds it exclusively (AGENTS.md). A run that
# starts without a ready server just fails 25 times in a row.
if ! curl -sf --max-time 10 "$HEALTH" | grep -q '"ready":true'; then
    echo "서버가 준비되지 않았다: $HEALTH" >&2
    curl -s --max-time 10 "$HEALTH" >&2 || echo "  (응답 없음)" >&2
    echo "사람 사용자가 tmux에서 서버를 띄워야 한다 (AGENTS.md 규칙 0)" >&2
    exit 1
fi

DIRTY=$(git status --porcelain | wc -l)
STAMP=$(date '+%m%d-%H%M%S')
SLUG=$(echo "${MODEL}_${QA_FILE}" | tr '/+' '__')
LOG="out/logs/${SLUG}_${STAMP}.log"
mkdir -p out/logs

ARGS=(--model "$MODEL" --sequence_id "$SEQUENCE" --qa_file "$QA_FILE"
      --caption_file "$CAPTIONS" --data_dir remembr/data/ --coda_dir "$CODA_DIR"
      --postfix "_${STAMP}" "$@")

# Only default the temperature if the caller did not ask for one, so passing
# `--temperature 0` for a paired comparison still works.
if ! printf '%s\n' "$@" | grep -qx -- '--temperature'; then
    ARGS+=(--temperature 0.7)
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "log:  $LOG"
    echo "cmd:  python -u remembr/scripts/eval.py ${ARGS[*]}"
    exit 0
fi

{
    echo "=== $(date '+%F %H:%M:%S')  $MODEL  $QA_FILE ==="
    echo "브랜치 $(git branch --show-current)  커밋 $(git rev-parse --short HEAD)  dirty ${DIRTY}개"
    if [[ "$DIRTY" != "0" ]]; then
        echo "경고: 커밋되지 않은 변경 ${DIRTY}개. 이 결과가 어느 코드로 나온 것인지 재현할 수 없다 (#45)"
        git status --porcelain | sed 's/^/       /'
    fi
    echo "명령 python -u remembr/scripts/eval.py ${ARGS[*]}"
    echo
} > "$LOG"

ln -sfn "$REPO/$LOG" out/logs/latest.log
echo "로그: $LOG  (out/logs/latest.log 도 여기를 가리킨다)"

# shellcheck source=/dev/null
source remembr_venv/bin/activate
REMEMBR_TRACE=1 PYTHONWARNINGS=ignore python -u remembr/scripts/eval.py "${ARGS[@]}" >> "$LOG" 2>&1
echo "완료: $LOG"
