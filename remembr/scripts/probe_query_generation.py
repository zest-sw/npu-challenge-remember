"""Does the LLM produce a usable visual search query, on its own, in one call?

Why this exists
---------------
In `remembr+` the planner has to invent its own search query as a tool argument,
and on this board it does not: asked when the robot left a building it searched
for "a green car driving down the road", and later for "rainbow" and even for
"your available tools". Retrieval then returns unrelated memories and every later
stage is working from the wrong evidence, so no amount of answer-side fixing helps.

Retrieval itself is not the broken part. Handed a query a human wrote, the same
captions/embeddings/Milvus find the answer: 3 of 4 position questions land within
5 m of ground truth (0.0 m, 2.9 m, 0.0 m; the miss was 21.2 m). So the failure is
specifically the model's query generation inside the tool-calling format.

The cheap hypothesis this probes: the format is the problem, not the model. Asking
"what should I search for?" as a plain question — no tool schema, no JSON, no
multi-step agent loop — is a much easier task than emitting a tool call. If the
model can do that, `remembr+` can be replaced by a fixed pipeline
(query -> retrieve -> answer) and the expensive work of wiring native
tool-calling into the server is unnecessary.

Result: it can. 3/4, matching the human ceiling, and on the police call pole it
beat the human (2.8 m vs 21.2 m) — which also killed the theory that the captions
simply do not cover that object. So the format is the blocker, native tool-calling
is not needed, and `probe_query_format.py` takes it from here and asks which part
of the format does the damage.

What it reports
---------------
Per question: the query the model generated, and how close the retrieved memories
land to the ground-truth coordinate. Compared against the human-written queries as
a ceiling. Needs a running server with an LLM and a populated Milvus collection;
uses ~20 generated tokens per question so it is cheap.

    python remembr/scripts/probe_query_generation.py --collection eval_memory_0
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.memory.milvus_memory import MilvusMemory  # noqa: E402

# Deliberately not a tool description: no schema, no sample values to copy (the
# model was observed lifting both), just the task and a length cap.
QUERY_SYSTEM_PROMPT = (
    "You turn a user's question into a short visual search phrase for searching a "
    "robot's camera memory.\n"
    "Reply with ONLY the phrase naming the object, place or activity the question is "
    "about. No explanation, no quotes, at most 6 words."
)

# A human's answer to the same task, as the ceiling to compare against. Keyed by the
# asked line of the question.
HUMAN_QUERIES = {
    "Where is the closest place to sit?": "bench seating place to sit",
    "Where is the yellow police call pole?": "yellow police call pole emergency phone",
    "Where is the statue in front of the Bill and Melinda Gates Computer Science building?":
        "statue sculpture in front of building",
    "Where is the nearest recycling bin?": "recycling bin trash can",
    "When did you leave the building?": "exiting leaving the building through doors",
    "When did you first enter the building?": "entering the building through doors",
    "How long were you in the building for?": "inside the building interior",
    "How long have you been outside?": "outdoors outside walking",
}

HIT_RADIUS_M = 5.0


def asked_line(question):
    """The actual question, without the 'current time / located at' preamble."""
    lines = [line.strip() for line in question.strip().splitlines() if line.strip()]
    return lines[-1] if lines else question.strip()


def post_choice(base_url, messages, max_tokens, schema=None, model=None, timeout=300):
    """One chat request. Returns (the server's first choice, the model it says it served).

    The whole choice, not just its text, because a probe that varies `max_tokens` needs
    `finish_reason` to tell a finished answer from a truncated one.

    temperature=0 throughout: these probes compare prompt shapes, and sampling
    variance at n=4 would swamp the difference being measured.
    """
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if model:
        payload["model"] = model
    if schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "probe", "schema": schema}}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body["choices"][0], body.get("model")


def post_chat(base_url, messages, max_tokens, schema=None, model=None):
    """post_choice, for the callers that only want the text."""
    choice, served = post_choice(base_url, messages, max_tokens, schema=schema, model=model)
    return choice["message"]["content"].strip(), served


def call_chat(base_url, messages, max_tokens, schema=None):
    """post_chat without the served-model name, for callers that don't report it."""
    return post_chat(base_url, messages, max_tokens, schema=schema)[0]


def generate_query(base_url, model, question, max_tokens=20):
    return post_chat(
        base_url,
        [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": asked_line(question)},
        ],
        max_tokens,
        model=model,
    )


def nearest_hit(memory, query, ground_truth):
    """Smallest distance from ground truth to any retrieved memory's position."""
    docs = memory.text_retriever.invoke(query)
    distances = []
    for doc in docs:
        position = (getattr(doc, "metadata", None) or {}).get("position")
        if position is None or ground_truth is None:
            continue
        try:
            distances.append(float(np.linalg.norm(np.array(ground_truth) - np.array(position))))
        except (TypeError, ValueError):
            continue
    top = docs[0].page_content if docs else ""
    return (min(distances) if distances else None), top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="eval_memory_0")
    parser.add_argument("--qa_file", default="remembr/data/questions/0/human_qa_SUBSET.json")
    parser.add_argument("--base_url", default=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"))
    parser.add_argument("--model", default="tt/probe")
    parser.add_argument("--db_ip", default="127.0.0.1")
    args = parser.parse_args()

    questions = json.load(open(args.qa_file))["data"]
    memory = MilvusMemory(args.collection, db_ip=args.db_ip, time_offset=questions[0]["start_time"])

    served = None
    rows = []
    for question in questions:
        asked = asked_line(question["question"])
        ground_truth = question["answers"].get("position")
        try:
            query, served = generate_query(args.base_url, args.model, question["question"])
        except (urllib.error.URLError, OSError) as error:
            print(f"  server unreachable ({type(error).__name__}); is it up and in an LLM mode?")
            return 1
        model_distance, top_caption = nearest_hit(memory, query, ground_truth)
        human_query = HUMAN_QUERIES.get(asked)
        human_distance = nearest_hit(memory, human_query, ground_truth)[0] if human_query else None
        rows.append((question["type"], asked, query, model_distance, human_query, human_distance, top_caption))

    # Report what actually served the requests, not what was asked for: the board
    # holds one model and ignores the requested name.
    print(f"served model: {served}\ncollection: {args.collection}  questions: {len(rows)}\n")
    for qtype, asked, query, model_distance, human_query, human_distance, top in rows:
        def show(distance):
            return "  n/a  " if distance is None else f"{distance:6.1f}m"

        mark = ""
        if qtype == "position" and model_distance is not None:
            mark = "OK " if model_distance < HIT_RADIUS_M else "MISS"
        print(f"[{qtype:8s}] {mark:4s} {asked[:52]}")
        print(f"    model : {show(model_distance)}  '{query}'")
        if human_query:
            print(f"    human : {show(human_distance)}  '{human_query}'")

    position_rows = [r for r in rows if r[0] == "position"]
    for label, index in (("model", 3), ("human", 5)):
        hits = sum(1 for r in position_rows if r[index] is not None and r[index] < HIT_RADIUS_M)
        print(f"\nposition within {HIT_RADIUS_M:.0f}m ({label}): {hits}/{len(position_rows)}")
    print("\nIf the model column matches the human column, the tool-calling format was the "
          "problem and a fixed query->retrieve->answer pipeline can replace the agent loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
