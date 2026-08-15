"""Which part of "pick the memory that answers this" can the 8B actually do?

Why this exists
---------------
`direct+` hands the model the retrieved memories and asks which one answers the question.
It scored 26.9 m average error on twelve position questions where picking the best of what
it was shown would have scored 7.1 m, so the selection step looked like the bottleneck --
and three attempts to fix it by prompting all failed: naming the selection criterion in the
system prompt, making the model commit to a memory number instead of typing a coordinate,
and re-rendering the memories caption-first. The last one made things *worse* (26.9 -> 39.2 m).

Every one of those measurements confounded three things at once: real captions, real
retrieval, real questions. This separates them. The memories here are synthetic, so the
right answer is known, and one variable moves at a time.

What it found
-------------
Identification is not the problem. 60/60, four different target objects per condition:

    target at slot 1 / 2 / 3 / 4 / 5 (k=5)          4/4 each
    k = 2 / 3 / 5 / 10 / 20, target in the middle   4/4 each
    distractors from the same scene type            4/4
    target buried in four filler sentences          4/4
    target paraphrased, question's noun absent      4/4

So it reads the whole list, at any length up to 20, and matches on meaning rather than on
the question's words. The one thing it cannot do is arithmetic on coordinates:

    candidates at  8 / 75 / 210 m   (10x apart)     11/12   picks the nearest
    candidates at  8 / 12 / 210 m   (50% apart)     11/12
    candidates at 97.1 / 104.5 / 122.2 / 123.0 m     2/5    <- the real geometry
      same list, distances given as numbers          4/5

That last pair is the finding. Asked to compare coordinates when the answers are within
7-27% of each other, it stops discriminating and answers with the *last* memory whatever
the arrangement -- the same collapse that showed up on real data at k=2 and k=3, where it
picked the last of the list 9 times in 12.

Two other things worth knowing:

  * with nothing in the list matching, the grammar still forces a pick and it picks
    arbitrarily (3, 5, 5, 1 across four questions). It cannot say "none of these".
  * dropping "closest" from the question scatters the picks, so the hits above are the
    model responding to the qualifier, not to position.

`DirectPositionAgent` acts on this by computing each distance and showing it as a number,
which moved the twelve-question run from 26.9 m to 20.6 m.

Needs a running server with an LLM. No Milvus, no captions, no NPU beyond the LLM itself;
about 140 one-token generations, roughly three minutes.

    python remembr/scripts/probe_memory_selection.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

URL = os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1") + "/chat/completions"
MODEL = os.environ.get("TT_LLM_MODEL", "Llama-3.1-8B-Instruct")

# Deliberately the agent's own wording, so this measures the deployed configuration.
SYSTEM = ("You are a robot looking back over your memory of what you saw. Each numbered "
          "memory below records where you were and what you saw there. Pick the memory "
          "whose description mentions the thing the question asks about, and respond with "
          "only that memory's number.")
SYSTEM_DISTANCE = SYSTEM.replace("records where you were and what you saw there",
                                 "records what you saw and how far it is from you now")

TARGETS = [
    {"question": "Where is the closest place to sit?",
     "plain": "A wooden bench sits against the left wall beside a potted plant.",
     "paraphrase": "A place to rest yourself is fixed to the left wall, wooden and waist high.",
     "variants": ["A wooden bench sits against the left wall beside a potted plant.",
                  "A metal bench stands under the window, empty.",
                  "A stone bench is set into the low wall by the entrance."]},
    {"question": "Where is the nearest recycling bin?",
     "plain": "A blue recycling bin stands beside the doorway.",
     "paraphrase": "A blue container for bottles and cans stands beside the doorway.",
     "variants": ["A blue recycling bin stands beside the doorway.",
                  "A recycling bin with a round lid sits at the corner of the corridor.",
                  "Two recycling bins are lined up against the railing."]},
    {"question": "Where are the closest elevators?",
     "plain": "A bank of elevators faces the corridor, doors closed.",
     "paraphrase": "Two sliding doors with call buttons face the corridor.",
     "variants": ["A bank of elevators faces the corridor, doors closed.",
                  "An elevator lobby opens to the right, call buttons lit.",
                  "A single elevator door is set into the panelled wall."]},
    {"question": "Where did you see a red fire hydrant?",
     "plain": "A red fire hydrant sits at the edge of the curb.",
     "paraphrase": "A squat red cast-iron water outlet for firefighters sits at the curb edge.",
     "variants": ["A red fire hydrant sits at the edge of the curb.",
                  "A fire hydrant painted red stands on the grass verge.",
                  "A red hydrant is bolted to the pavement by the crossing."]},
]

FAR = [
    "Pavement stretches forward in a herringbone pattern of tan bricks.",
    "Concrete sidewalk runs alongside a paved road under a gray sky.",
    "A wide stone walkway of light-gray flagstones stretches between low hedges.",
    "Paved courtyard bordered by mature live oaks and a metal lattice fence.",
    "Gravel path curves past a bed of dry leaves toward a parking area.",
    "Asphalt road surface with faded white lane markings and a manhole cover.",
    "Grass verge beside a low brick retaining wall and a drainage grate.",
    "Brick plaza with a shallow step down toward a row of bollards.",
    "Wet pavement reflecting overhead lights near a loading dock.",
    "Dirt track running between two rows of young trees.",
    "Tiled forecourt in front of a glass revolving door.",
    "Long ramp of poured concrete flanked by steel handrails.",
    "Cobbled lane narrowing between two brick walls.",
    "Open square paved in gray slabs with a drain in the centre.",
    "Boardwalk of weathered planks over a shallow ditch.",
    "Median strip of mown grass dividing two carriageways.",
    "Service alley with painted yellow kerbs and a fire door.",
    "Footpath edged with granite setts beside a hedge.",
    "Terrace of poured concrete overlooking a lower courtyard.",
]

# Same indoor scene as the bench, sharing its vocabulary with nothing to sit on.
NEAR = [
    "A spacious atrium with polished beige tile flooring and recessed lighting.",
    "Hallway with polished terrazzo floor, flanked by glass doors and white walls.",
    "Interior corridor with cream walls and large framed posters along one side.",
    "Lobby with a potted tree in a white ceramic planter beside a reception desk.",
    "Indoor walkway with wood-panelled walls and fluorescent ceiling lights.",
]

FILLER = [
    "The ceiling is fitted with recessed lights spaced evenly along its length.",
    "Sunlight enters through tall windows on the far side of the space.",
    "The floor surface is clean and reflects the lighting overhead.",
    "A doorway is visible at the far end, propped slightly open.",
]

HERE = [-55.7, 86.3, -2.1]  # an arbitrary pose, as the real questions have
calls = 0


def coordinate(distance, index):
    """A point that far from HERE, in a direction that varies so it is not a giveaway."""
    heading = [(0.6, 0.8), (-0.8, 0.6), (0.8, -0.6), (-0.6, -0.8)][index % 4]
    return [round(HERE[0] + heading[0] * distance, 3),
            round(HERE[1] + heading[1] * distance, 3), -2.4]


def ask(question, entries, show_distance=False):
    """entries: list of (caption, distance). Returns the chosen 1-based index."""
    global calls
    if show_distance:
        body = "\n".join(f"{n}. distance_from_you={d:.1f}m. The robot saw the following: {c}"
                         for n, (c, d) in enumerate(entries, 1))
    else:
        body = "\n".join(
            f"{n}. At time=2023-01-16 07:5{n}:00, the robot was at an average position of "
            f"{coordinate(d, n)}.The robot saw the following: {c}"
            for n, (c, d) in enumerate(entries, 1))
    prompt = (f"{SYSTEM_DISTANCE if show_distance else SYSTEM}\n\n"
              f"Your current situation: You started moving at 07:58:58. The current time is "
              f"08:10:00 and you are located at {HERE}.\n"
              f"(This is where and when you are now. It is not the answer.)\n\n"
              f"{body}\n\nQuestion: {question}")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "memory_index",
            "schema": {"type": "integer", "enum": list(range(1, len(entries) + 1))}}},
    }
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    calls += 1
    try:
        with urllib.request.urlopen(request, timeout=180) as reply:
            return json.loads(json.loads(reply.read())["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as error:
        print(f"    HTTP {error.code}: {error.read().decode()[:160]}", file=sys.stderr)
        return None


def place(target, count, slot, pool=FAR, distance=50.0):
    """`count` memories with `target` at 1-based `slot`, the rest drawn from `pool`."""
    entries = [(pool[i % len(pool)], 100.0 + i) for i in range(count - 1)]
    entries.insert(slot - 1, (target, distance))
    return entries


def report(label, cases, show_distance=False):
    """cases: list of (question, entries, expected_slot)."""
    picks = [ask(q, e, show_distance) for q, e, _ in cases]
    hits = sum(p == expected for p, (_, _, expected) in zip(picks, cases))
    print(f"  {label:38s} {hits}/{len(cases)}   골랐다 {picks}")
    return hits, len(cases)


def main():
    start = time.perf_counter()

    print("=== A. 정답 위치 (k=5) ===")
    for slot in range(1, 6):
        report(f"정답이 {slot}번",
               [(t["question"], place(t["plain"], 5, slot), slot) for t in TARGETS])

    print("\n=== B. 후보 개수 (정답은 가운데) ===")
    for count in (2, 3, 5, 10, 20):
        slot = (count + 1) // 2
        report(f"k={count:<2d} (정답 {slot}번)",
               [(t["question"], place(t["plain"], count, slot), slot) for t in TARGETS])

    print("\n=== C. 방해자 유사도 (k=5, 정답 3번) ===")
    report("방해자가 딴판", [(t["question"], place(t["plain"], 5, 3, FAR), 3) for t in TARGETS])
    report("방해자가 같은 장면 유형",
           [(t["question"], place(t["plain"], 5, 3, NEAR), 3) for t in TARGETS])

    print("\n=== D. 캡션 길이 (k=5, 정답 3번) ===")
    report("정답이 한 문장", [(t["question"], place(t["plain"], 5, 3), 3) for t in TARGETS])
    report("정답이 5문장 중 하나로 파묻힘",
           [(t["question"],
             place(" ".join([FILLER[0], FILLER[1], t["plain"], FILLER[2], FILLER[3]]), 5, 3), 3)
            for t in TARGETS])

    print("\n=== E. 어휘 일치 (k=5, 정답 3번) ===")
    report("질문의 명사가 캡션에 그대로",
           [(t["question"], place(t["plain"], 5, 3), 3) for t in TARGETS])
    report("의역 (질문 명사 없음)",
           [(t["question"], place(t["paraphrase"], 5, 3), 3) for t in TARGETS])

    print("\n=== F. 정답이 아예 없을 때 (k=5) ===")
    absent = [ask(t["question"], [(c, 100.0) for c in FAR[:5]]) for t in TARGETS]
    print(f"  {'무엇을 고르나':38s}         골랐다 {absent}  분포 {dict(sorted(Counter(absent).items()))}")

    print("\n=== G. 여러 개가 조건에 맞을 때 가장 가까운 것을 고르나 ===")
    for label, distances in (("멀리 떨어짐  8 / 75 / 210 m", [8.0, 75.0, 210.0]),
                             ("가깝게 붙음  8 / 12 / 210 m", [8.0, 12.0, 210.0])):
        print(f"  -- {label}")
        for slot in (1, 3, 5):
            cases = []
            for target in TARGETS:
                matches = list(zip(target["variants"], distances))
                entries = [matches[1], (FAR[0], 150.0), (FAR[1], 190.0), matches[2]]
                entries.insert(slot - 1, matches[0])
                cases.append((target["question"], entries[:5], slot))
            report(f"     가장 가까운 것이 {slot}번", cases)

    print("\n=== H. 실제 기하 (97.1 / 104.5 / 122.2 / 123.0 m) ===")
    real = [97.1, 104.5, 122.2, 123.0]
    for show_distance, label in ((False, "좌표를 주고 비교시킴"), (True, "거리를 숫자로 줌")):
        cases = []
        for slot in range(1, 6):
            target = TARGETS[0]
            matches = [(target["variants"][i % 3] + f" ({i})", real[i]) for i in range(4)]
            entries = [matches[1], matches[2], (FAR[0], 160.0), matches[3]]
            entries.insert(slot - 1, matches[0])
            cases.append((target["question"], entries[:5], slot))
        report(label, cases, show_distance)

    print(f"\n총 {calls}콜, {time.perf_counter() - start:.0f}초")


if __name__ == "__main__":
    main()
