"""A readable, one-line-per-step view of how a question got answered.

Why this lives on the client and not in the server
--------------------------------------------------
The server sees an already-assembled prompt string, a schema, and the tokens it
generates. It cannot tell whether a request is the search-query call, a tool turn,
or the final answer, and the question and the retrieved context arrive as one blob.
The agent knows all of that. So the "why did it decide that" view belongs here; the
server's `TT_DEBUG_DECODE=1` covers the other layer, "what is it writing right now".

Off unless `REMEMBR_TRACE=1`, so it costs nothing in normal runs.

    REMEMBR_TRACE=1 python remembr/scripts/eval.py --model retrieve+tt/...

prints, per question:

    [ 3/25 position] Where is the nearest recycling bin?
      검색어      → 'recycling bin'
      검색        ← 메모리 4개
      유형        → position
      답변        → position=[-102.997, 161.855, -2.54]
      채점        27.4m  attempts=1  31.2s

and for the agent path, one line per tool turn, so a repeated search or a bad
argument is visible at a glance rather than buried in server logs.
"""
import os
import sys

ENABLED = os.environ.get("REMEMBR_TRACE", "0") == "1"

_LABEL_WIDTH = 10


def question(index, total, qtype, asked):
    """Header for one question. `index` is 1-based."""
    if not ENABLED:
        return
    print(f"\n[{index:2d}/{total} {qtype}] {asked}", flush=True)


def step(label, text, arrow="→"):
    """One line inside the current question.

    Use `→` for something the model produced and `←` for something handed back to
    it, so the direction of information is readable without parsing the words.
    """
    if not ENABLED:
        return
    print(f"  {label:<{_LABEL_WIDTH}} {arrow} {text}", flush=True)


def note(text):
    """An unlabelled line, for things that are neither input nor output."""
    if not ENABLED:
        return
    print(f"  {'':<{_LABEL_WIDTH}}   {text}", flush=True)


def warn(text):
    """Something went wrong. Printed even when tracing is off -- these are the lines
    worth seeing in a normal run too, and they used to be bare prints scattered
    around the agents."""
    print(f"  ! {text}", file=sys.stderr, flush=True)
