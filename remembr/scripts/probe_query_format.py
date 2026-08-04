"""Which part of the tool-calling format broke the planner's search queries?

Why this exists
---------------
In `remembr+` the planner invents its own search query as a tool argument, and on
this board it produces unrelated ones: asked when the robot left a building it
searched for "a green car driving down the road", elsewhere "rainbow" and even
"your available tools". Retrieval then returns unrelated memories and every later
stage reasons from the wrong evidence, so no amount of answer-side fixing helps.

`probe_query_generation.py` established the two ends of the problem:

  * retrieval is fine       - human-written queries land 3/4 position questions
                              within 5 m (0.0 m, 2.9 m, 0.0 m; one 21.2 m miss)
  * the model is fine       - asked the same thing as a plain question, with no
                              tool schema and no JSON, it also lands 3/4

So the tool-calling format is what breaks it. But the format changes four things
at once, and this script separates them:

  DESC   plain question + the real tool specs pasted in front
         -> if this degrades, nearby copyable text was the cause
  JSON1  question -> {"query": "..."} under a grammar, no tool specs
         -> if this degrades, constrained/structured output was the cause
  JSON4  question -> the 4-field agent shape (2 reasoning fields + tool +
         tool_input), no tool specs
         -> if only this degrades, the number of slots to fill was the cause

Results so far
--------------
    baseline (plain)  3/4
    DESC              2/4   <- valid; tool specs alone cost one question
    JSON1, JSON4      invalid, must be re-run

The JSON variants were measured against a poisoned server. Issue #21: the LLM
engine had two decode paths and tt-metal's trace cache did not distinguish them,
so whichever kind of request arrived first won and the other kind was broken for
the life of the process. This script calls DESC (plain) before the JSON variants
for each question, so its own first call poisoned every JSON call after it -- the
0/4 readings were the bug, not the model. Fixed by removing the device path; the
mixing is harmless now, and the ordering below is left as it is deliberately so a
rerun also re-checks that the fix holds.

    python remembr/scripts/probe_query_format.py --collection eval_memory_0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.utils.function_calling import convert_to_openai_function  # noqa: E402

from remembr.agents.remembr_agent import ReMEmbRAgent  # noqa: E402
from remembr.memory.milvus_memory import MilvusMemory  # noqa: E402
from remembr.scripts.probe_query_generation import (  # noqa: E402
    HIT_RADIUS_M,
    QUERY_SYSTEM_PROMPT,
    asked_line,
    call_chat,
    nearest_hit,
)

# One string field. Isolates "output is grammar-constrained" from everything else.
SCHEMA_ONE_SLOT = {
    "type": "object",
    "properties": {"query": {"type": "string", "maxLength": 60}},
    "required": ["query"],
    "additionalProperties": False,
}

# The shape ReMEmbRAgent actually asks for. Same grammar machinery as above, but
# the phrase is now the last of four fields the model has to fill.
SCHEMA_FOUR_SLOTS = {
    "type": "object",
    "properties": {
        "context_reasoning": {"type": "string", "maxLength": 200},
        "tool_reasoning": {"type": "string", "maxLength": 200},
        "tool": {"type": "string", "enum": ["retrieve_from_text"]},
        "tool_input": {
            "type": "object",
            "properties": {"x": {"type": "string", "maxLength": 60}},
            "required": ["x"],
            "additionalProperties": False,
        },
    },
    "required": ["context_reasoning", "tool_reasoning", "tool", "tool_input"],
    "additionalProperties": False,
}


def real_tool_specs():
    """The tool specs exactly as FunctionsWrapper puts them in front of the model."""

    class _StubMemory:
        def search_by_text(self, query):
            return ""

        def search_by_position(self, query):
            return ""

        def search_by_time(self, query):
            return ""

    agent = ReMEmbRAgent.__new__(ReMEmbRAgent)
    ReMEmbRAgent.create_tools(agent, _StubMemory())
    return json.dumps([convert_to_openai_function(tool) for tool in agent.tool_list], indent=2)


def variant_desc(base_url, asked, specs):
    return call_chat(
        base_url,
        [
            {"role": "system", "content": f"{QUERY_SYSTEM_PROMPT}\n\nFor reference, the retrieval tools are:\n{specs}"},
            {"role": "user", "content": asked},
        ],
        max_tokens=20,
    )


def variant_json_one_slot(base_url, asked, _specs):
    out = call_chat(
        base_url,
        [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT + '\nReply as {"query": "..."}.'},
            {"role": "user", "content": asked},
        ],
        max_tokens=40,
        schema=SCHEMA_ONE_SLOT,
    )
    return json.loads(out).get("query", "")


def variant_json_four_slots(base_url, asked, _specs):
    out = call_chat(
        base_url,
        [
            {
                "role": "system",
                "content": QUERY_SYSTEM_PROMPT + "\nReply with the tool-call object; put the phrase in tool_input.x.",
            },
            {"role": "user", "content": asked},
        ],
        max_tokens=220,
        schema=SCHEMA_FOUR_SLOTS,
    )
    return (json.loads(out).get("tool_input") or {}).get("x", "")


VARIANTS = (
    ("DESC", variant_desc),
    ("JSON1", variant_json_one_slot),
    ("JSON4", variant_json_four_slots),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="eval_memory_0")
    parser.add_argument("--qa_file", default="remembr/data/questions/0/human_qa_SUBSET.json")
    parser.add_argument("--base_url", default=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"))
    parser.add_argument("--db_ip", default="127.0.0.1")
    args = parser.parse_args()

    specs = real_tool_specs()
    questions = json.load(open(args.qa_file))["data"]
    memory = MilvusMemory(args.collection, db_ip=args.db_ip, time_offset=questions[0]["start_time"])

    rows = {name: [] for name, _ in VARIANTS}
    for question in questions:
        asked = asked_line(question["question"])
        truth = question["answers"].get("position")
        for name, run in VARIANTS:
            try:
                query = run(args.base_url, asked, specs)
            except Exception as error:  # noqa: BLE001
                # A variant that cannot even produce parseable output is a result,
                # not a crash -- record it and keep the other variants comparable.
                rows[name].append((question["type"], f"<{type(error).__name__}: {error}>", None))
                continue
            distance = nearest_hit(memory, query, truth)[0] if query else None
            rows[name].append((question["type"], query, distance))

    print(f"{'variant':7s} {'type':9s} {'hit':5s} {'dist':>8s}  query")
    print("-" * 88)
    for name, _ in VARIANTS:
        for qtype, query, distance in rows[name]:
            mark = ""
            if qtype == "position" and distance is not None:
                mark = "OK" if distance < HIT_RADIUS_M else "MISS"
            shown = "  n/a  " if distance is None else f"{distance:6.1f}m"
            print(f"{name:7s} {qtype:9s} {mark:5s} {shown:>8s}  {query[:52]}")
        print()

    print(f"position within {HIT_RADIUS_M:.0f}m (plain-text baseline = 3/4):")
    for name, _ in VARIANTS:
        position_rows = [row for row in rows[name] if row[0] == "position"]
        hits = sum(1 for row in position_rows if row[2] is not None and row[2] < HIT_RADIUS_M)
        print(f"  {name:6s} {hits}/{len(position_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
