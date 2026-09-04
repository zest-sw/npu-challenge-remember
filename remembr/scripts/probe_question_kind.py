"""Can the 8B tell a "nearest X" question from a "where did you see X" question?

Why this exists
---------------
Retrieval ranks captions by what they say, which answers "where is the statue" but not
"where is the *nearest* bike rack": the discriminator there is distance from the robot, and
the retriever never looks at position. Measured on Q10, twenty-one captions in the window
mention a bike rack and the right one ranked sixth, so it missed the five shown to the
model -- while the answer sat 45.6 m away and four of the five shown were further.

Feeding the robot's own position into ranking needs to know which questions want it. A
regex on `nearest|closest` separates all twelve position questions correctly, but the twelve
are only eight distinct sentences and four of those carry the keyword, so the regex is
fitted to four examples and a real robot gets asked "where can I sit around here?".

So: ask the model. One extra call, but the answer is a single grammar-constrained digit --
measured at 0.7 s against the answer call's 4.8 s.

What is measured
----------------
Three groups of questions, each labelled with the class it belongs to:

    dataset     the twelve position questions as written, keyword and all
    paraphrase  the same intent with `nearest`/`closest` removed -- what a regex misses
    trap        distance-sounding wording that is not asking for the nearest thing

and several wordings of the classification question. Accuracy alone hides the failure
found first time out, so each row reports recall per class and the digits actually
generated: `nearest 10/10, named 0/8` with `1x18` is a constant, not a classifier.

Every wording is run twice, once with the digits reversed, because a two-value enum cannot
tell a preference for the first digit from an understanding of the first class.

Needs a running server with an LLM. No Milvus, no captions: 144 one-token generations,
roughly two minutes.

    python remembr/scripts/probe_question_kind.py
    python remembr/scripts/probe_question_kind.py --variant elsewhere --show
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.scripts.probe_query_generation import post_choice  # noqa: E402

BASE_URL = os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1")
MODEL = os.environ.get("TT_LLM_MODEL", "Llama-3.1-8B-Instruct")

NEAREST, NAMED = "nearest", "named"

# As written in human_qa.json. Duplicates dropped -- the repeats differ only in the robot's
# position, which is not part of the sentence being classified.
DATASET = [
    ("Where is the closest place to sit?", NEAREST),
    ("Where is the nearest recycling bin?", NEAREST),
    ("Where are the closest elevators?", NEAREST),
    ("Where is the nearest bike rack that is not empty?", NEAREST),
    ("Where is the yellow police call pole?", NAMED),
    ("Where is the statue in front of the Bill and Melinda Gates Computer Science building?",
     NAMED),
    ("Where did you see the sculpture with the hanging boats?", NAMED),
    ("Where did you see a red fire hydrant?", NAMED),
]

# Same intent, no `nearest` or `closest` anywhere. The group the regex cannot see.
PARAPHRASE = [
    ("Where can I sit around here?", NEAREST),
    ("I need to throw this bottle away, where do I go?", NEAREST),
    ("Which way to a lift from here?", NEAREST),
    ("Where do I leave my bike?", NEAREST),
    ("I want to sit down. Where should I head?", NEAREST),
    ("Where is somewhere nearby to park a bicycle?", NEAREST),
]

# Distance words that are not about distance from the robot. Classified as NEAREST, each
# would send retrieval after whatever happens to be underfoot.
TRAP = [
    ("Where is the bench closest to the fountain?", NAMED),
    ("Where did you see the recycling bin with the blue lid?", NAMED),
    ("Where is the building nearest the statue you passed?", NAMED),
    ("Where did you see a bike rack with no bikes in it?", NAMED),
]

GROUPS = (("데이터셋 원문", DATASET),
          ("의역", PARAPHRASE),
          ("함정", TRAP))
CASES = [case for _, group in GROUPS for case in group]


def two_class(a, b):
    """The first wording tried: name both classes and pick one.

    It answered one class for all eighteen questions, whichever digit that class was
    given. The two descriptions overlap -- "where did you see the fire hydrant?" is asking
    about one particular thing *and*, to a robot, about something near it.
    """
    return ("You are a robot deciding how to search your memory of what you saw.\n"
            f"Answer {a} if the question asks for whichever one is nearest to you now, so "
            "that the answer depends on where you are standing.\n"
            f"Answer {b} if the question asks about one particular thing, so that the "
            "answer is the same wherever you are standing.\n"
            "Respond with only the digit.")


def yes_no(a, b):
    """One condition, not two, so there is nothing for the classes to overlap on."""
    return ("You are a robot that remembers places you have been.\n"
            "Question: does answering it require knowing where you are standing right now?\n"
            f"Answer {a} for yes, {b} for no.\n"
            "Respond with only the digit.")


def elsewhere(a, b):
    """The same condition put as a counterfactual, which needs no notion of "nearest".

    "The nearest bench" is a different bench from the far side of campus; "the statue in
    front of the Gates building" is the same statue from anywhere.
    """
    return ("You are a robot that remembers places you have been. You will be asked where "
            "something is.\n"
            "Consider: if you were standing somewhere else, would the correct answer be a "
            "different place?\n"
            f"Answer {a} if the answer would change.\n"
            f"Answer {b} if the answer would be the same place either way.\n"
            "Respond with only the digit.")


def few_shot(a, b):
    """`two_class` plus worked examples, in case the definitions were the whole problem."""
    return (two_class(a, b) + "\n\nExamples:\n"
            f"Where is the nearest water fountain? -> {a}\n"
            f"Where can I find a toilet? -> {a}\n"
            f"Where is the clock tower? -> {b}\n"
            f"Where did you see the mural of a whale? -> {b}")


VARIANTS = {"two_class": two_class, "yes_no": yes_no,
            "elsewhere": elsewhere, "few_shot": few_shot}

REGEX = re.compile(r"\b(nearest|closest)\b", re.I)
calls = 0
served = set()


def classify(system, question, digits):
    """digits: {NEAREST: int, NAMED: int}. Returns the class the model chose."""
    global calls
    calls += 1
    # One digit, so the grammar leaves nothing else generable and nothing to trim.
    choice, model = post_choice(
        BASE_URL,
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        max_tokens=4, model=MODEL,
        schema={"type": "integer", "enum": sorted(digits.values())})
    served.add(model)
    answer = json.loads(choice["message"]["content"])
    return next(kind for kind, digit in digits.items() if digit == answer), answer


def score(build, digits, show=False):
    """Recall per class, so a constant answer cannot look like accuracy."""
    hits = {NEAREST: 0, NAMED: 0}
    seen = {}
    system = build(digits[NEAREST], digits[NAMED])
    for question, expected in CASES:
        chosen, digit = classify(system, question, digits)
        seen[digit] = seen.get(digit, 0) + 1
        hits[expected] += chosen == expected
        if show:
            print(f"    {'  ' if chosen == expected else '틀림'} {chosen:8s} {question}")
    return hits, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append")
    parser.add_argument("--show", action="store_true", help="질문별 결과를 모두 출력")
    args = parser.parse_args()

    totals = {NEAREST: sum(1 for _, k in CASES if k == NEAREST),
              NAMED: sum(1 for _, k in CASES if k == NAMED)}
    by_regex = {NEAREST: 0, NAMED: 0}
    for question, expected in CASES:
        guess = NEAREST if REGEX.search(question) else NAMED
        by_regex[expected] += guess == expected

    print(f"질문 {len(CASES)}개  (가까운 것 {totals[NEAREST]}, 특정한 것 {totals[NAMED]})\n")
    print(f"{'방식':14s}{'숫자':>8}{'가까운 것':>10}{'특정한 것':>10}{'합계':>8}   낸 숫자")
    print(f"{'정규식':14s}{'-':>8}{by_regex[NEAREST]:8d}/{totals[NEAREST]}"
          f"{by_regex[NAMED]:8d}/{totals[NAMED]}"
          f"{sum(by_regex.values()):6d}/{len(CASES)}")

    for name in (args.variant or sorted(VARIANTS)):
        for label, digits in (("1/2", {NEAREST: 1, NAMED: 2}),
                              ("2/1", {NEAREST: 2, NAMED: 1})):
            if args.show:
                print(f"\n  --- {name} {label} ---")
            hits, seen = score(VARIANTS[name], digits, args.show)
            spread = "  ".join(f"{d}x{n}" for d, n in sorted(seen.items()))
            print(f"{name:14s}{label:>8}{hits[NEAREST]:8d}/{totals[NEAREST]}"
                  f"{hits[NAMED]:8d}/{totals[NAMED]}"
                  f"{sum(hits.values()):6d}/{len(CASES)}   {spread}")
    print(f"\n콜 {calls}회   서버가 서빙한 모델: {sorted(served)}")


if __name__ == "__main__":
    main()
