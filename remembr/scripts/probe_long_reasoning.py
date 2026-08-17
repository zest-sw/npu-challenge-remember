"""Does the 8B lose the question when it writes a long answer?

Why this exists
---------------
`direct+ --reason` sends an unconstrained call to think in words before the
grammar-constrained pick. Raising its token budget on two real questions did not help and
twice made the reasoning wander off the question entirely:

    Q4, max_tokens=320  "the question is not explicitly stated, so I'll need to make
                         some educated guesses about what the question might be asking"
    Q4, max_tokens=480  "Memory 1 ... describes a hallway with a double door at the end,
                         which could be the exit or entrance the question is referring to"
    Q5, max_tokens=480  discussed memories 1-3 and never mentioned memory 5, then the
                         choosing call picked 5

The question sits after five captions of about a hundred words each, so there are two
candidate causes and they are separable:

    position   the question is ~500 words back, and long generation dilutes it
    length     the model conditions on its own growing output and drifts

So: synthetic memories, one unambiguous target, and one variable moved at a time. Graded
on things that are checkable rather than judged -- whether the generated reasoning still
contains the asked-about noun, whether it names the target's slot, and whether it emits a
drift marker ("the question is not stated", "unclear what is being asked").

Conditions
----------
    max_tokens       80 / 160 / 320 / 480
    question at      end (as `direct+` sends it) / start / both
    prompt size      5 memories / 12 memories

24 combinations, one call each. Needs a running server with an LLM; no Milvus, no
captions. Roughly fifteen minutes.

    python remembr/scripts/probe_long_reasoning.py
    python remembr/scripts/probe_long_reasoning.py --tokens 320 --show
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.direct_position_agent import reason_prompt  # noqa: E402
from remembr.scripts.probe_query_generation import post_choice  # noqa: E402

BASE_URL = os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1")
MODEL = os.environ.get("TT_LLM_MODEL", "Llama-3.1-8B-Instruct")

# Imported rather than copied: a copy cannot keep the guarantee that this measures the
# wording `direct+` actually sends.
INSTRUCTION = reason_prompt(has_distance=True)

ASKED = "Where did you see a red fire hydrant?"
NOUN = "hydrant"

TARGET = ("Concrete sidewalk runs alongside a paved road, bordered by a low brick wall. "
          "A red fire hydrant stands at the curb near the building's base. A white garage "
          "door is set into the lower level, flanked by narrow windows. The lot is empty "
          "except for a metal guardrail along the edge. The sky is overcast, casting a flat "
          "light across the scene.")

# Same register and length as the target, sharing its vocabulary with no hydrant in it.
FILLER = [
    "Pavement stretches forward in a herringbone pattern of tan and brown bricks, bordered "
    "by wide concrete sidewalks. Trees with green leaves line the street to the right, their "
    "trunks visible near lampposts. Buildings rise in the distance, brick and glass with "
    "windows reflecting the overcast sky. A few people walk far ahead on the sidewalk.",
    "Hallway with beige tiled floor and wood-panelled walls. Fluorescent lights hang from a "
    "white ceiling, illuminating posters along the wall. A metal door stands closed on the "
    "right, beside a chair with a woven seat. The corridor extends straight ahead, flanked "
    "by doors and framed artwork.",
    "Paved walkway made of interlocking bricks stretches forward, flanked by low stone "
    "retaining walls. Mature live oaks arch overhead, casting dappled shade on the path. "
    "Street lamps with white globes stand at intervals. Two people in dark clothing walk in "
    "the distance toward the campus centre.",
    "A spacious atrium with polished beige tile flooring and recessed lighting. Along the "
    "left wall, framed artwork hangs under warm accent lighting. To the right, glass doors "
    "lead to an adjacent room with tables and chairs. The ceiling is high with evenly spaced "
    "lights and the space feels open.",
    "Concrete driveway stretches between two tall buildings, flanked by concrete bollards "
    "and a red-painted curb. A white utility truck sits parked on the right near a loading "
    "dock. A low concrete wall runs along the far side, separating it from a walkway.",
    "Courtyard between brick and concrete buildings. Red-painted lines mark parking areas on "
    "the asphalt. A tall brick building with vertical metal panels stands behind, and a "
    "smaller structure is visible further back. No people are present.",
    "Glass doors with metal frames lead into a modern building. Behind them, wooden slat "
    "panels form a curved wall, casting shadows across the floor. A dark rug lies near the "
    "entrance. The corridor beyond extends into the distance under even lighting.",
    "Long, polished terrazzo floor stretches forward, flanked by glass doors and white "
    "walls. Metal handrails line the left side. Fluorescent lights run along the ceiling in "
    "unbroken rows. The corridor ends at a brighter area with daylight coming through.",
    "Pathway made of light grey concrete slabs stretches forward, bordered by trimmed green "
    "hedges on both sides. On the left, a lamppost with a rounded glass globe stands beside "
    "a low hedge. The path leads toward a distant building entrance with glass doors.",
    "Wide stone walkway of light-grey flagstones stretches between low hedges. Banners hang "
    "from poles at intervals along the edge. Brick buildings with arched windows stand back "
    "from the path. The ground is clear apart from a few scattered leaves.",
    "Service alley with painted yellow kerbs and a fire door set into the brick. Metal "
    "stairs descend to a lower level on the right. Windows line the upper floors, evenly "
    "spaced, reflecting a grey sky. A drainage grate sits near the centre.",
]

# Drift markers: the model saying it does not know what was asked.
LOST = re.compile(r"question (is|was) not|not explicitly stated|unclear what|"
                  r"question might be asking|educated guess", re.I)

TOKENS = (80, 160, 320, 480)
PLACES = ("end", "start", "both")
SIZES = (5, 12)
TARGET_SLOT = 3     # 1-based, kept away from both ends

calls = 0
served = set()


def build(place, size):
    """The prompt, with the target at TARGET_SLOT and `size` memories in total."""
    bodies = FILLER[:size - 1]
    bodies.insert(TARGET_SLOT - 1, TARGET)
    lines = [f"{n}. distance_from_you={10.0 * n:.1f}m. The robot saw the following: {b}"
             for n, b in enumerate(bodies, 1)]
    head = f"{INSTRUCTION}\n"
    if place in ("start", "both"):
        head += f"\nQuestion: {ASKED}\n"
    tail = f"\n\nQuestion: {ASKED}" if place in ("end", "both") else ""
    return f"{head}\n" + "\n".join(lines) + tail


def think(prompt, tokens):
    global calls
    calls += 1
    started = time.perf_counter()
    choice, model = post_choice(BASE_URL, [{"role": "user", "content": prompt}],
                                max_tokens=tokens, model=MODEL, timeout=600)
    served.add(model)
    return (choice["message"]["content"], choice.get("finish_reason"),
            time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, action="append")
    parser.add_argument("--show", action="store_true", help="생성된 추론을 전부 출력")
    args = parser.parse_args()

    print(f"목표: {NOUN!r} 를 담은 기억이 {TARGET_SLOT}번. 질문: {ASKED}\n")
    print(f"{'기억수':>5}{'질문위치':>9}{'max_tok':>8}{'생성':>7}{'끝':>7}"
          f"{'명사유지':>9}{'대상지목':>9}{'질문분실':>9}{'초':>7}")
    for size in SIZES:
        for place in PLACES:
            for tokens in (args.tokens or TOKENS):
                text, finish, seconds = think(build(place, size), tokens)
                kept = NOUN in text.lower()
                named = bool(re.search(rf"[Mm]emory {TARGET_SLOT}\b", text))
                lost = bool(LOST.search(text))
                print(f"{size:5d}{place:>9}{tokens:8d}{len(text.split()):7d}"
                      f"{('완결' if finish == 'stop' else '잘림'):>7}"
                      f"{('O' if kept else 'X'):>9}{('O' if named else 'X'):>9}"
                      f"{('O' if lost else '-'):>9}{seconds:7.1f}")
                if args.show:
                    for line in text.strip().splitlines():
                        if line.strip():
                            print(f"      {line.strip()}")
                    print()
    print(f"\n콜 {calls}회   서버가 서빙한 모델: {sorted(served)}")


if __name__ == "__main__":
    main()
