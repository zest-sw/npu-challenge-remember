"""Question -> vector search with the question itself -> a bare [x, y, z].

No query-generation call, no answer-type call, no JSON object: the grammar
allows only a three-number array, so every generated token is answer, none is
punctuation. A position-only probe -- every question gets a coordinate, and
non-position questions are knowingly wrong.

No retries. Greedy decoding repeats itself byte-for-byte, so a second try buys
nothing; a parse failure raises and eval's own attempt cap records it.

    python remembr/scripts/eval.py --model direct+tt/Llama-3.1-8B-Instruct ...
"""
import json
import os

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory
from remembr.utils import trace

# The whole answer contract. Number grammar is finite, so no maxLength is
# needed, and with no string branch the anyOf/null rules in agent.py do not
# apply here.
POSITION_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

SYSTEM_PROMPT = (
    "You are a robot looking back over your memory of what you saw. "
    "From the memory context, answer where the thing the question asks about "
    "is. Respond with only a JSON array of three numbers: [x, y, z]."
)


def position_output(parsed) -> AgentOutput:
    """A bare [x, y, z] as the AgentOutput eval scores.

    bool is an int subclass, so `[True, 1, 2]` must be rejected explicitly.
    """
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in parsed)
    ):
        raise ValueError(f"expected [x, y, z], got: {parsed!r}")
    return AgentOutput(type="position", text="", binary=None,
                       position=[float(n) for n in parsed],
                       orientation=None, duration=None, time=None)
