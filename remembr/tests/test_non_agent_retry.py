"""Self-check for NonAgent's answer-parsing retry cap.

This loop used to be `while True`, which hangs outright when the model cannot
produce parseable output — at temperature=0 every retry is byte-identical, so one
bad generation repeats forever. Measured on the board: 975s in 26 identical
retries of the same truncated JSON, stalling a 30-question eval on question 10.

The point of these tests is that they hang (not just fail) if the cap is removed,
so run them with a timeout in CI. No NPU, no Milvus, no server:

    python remembr/tests/test_non_agent_retry.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import AgentOutput  # noqa: E402
from remembr.agents.non_agent import NonAgent  # noqa: E402


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChain:
    """Stands in for ChatOpenAI. Always returns the same thing, like greedy does."""

    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, _inputs):
        self.calls += 1
        return FakeReply(self.content)


class FakeMemory:
    def get_working_memory(self):
        return []

    def memory_to_string(self, _memory):
        return "At time=2023-01-16 07:58:58, the robot was at [1.0, 2.0, 0.0]."


def make_agent(reply):
    """Build a NonAgent without touching __init__ (which would build an LLM)."""
    agent = NonAgent.__new__(NonAgent)
    agent.llm_type = "tt/fake"
    agent.chain = FakeChain(reply)
    agent.memory = FakeMemory()
    agent.prompt = "Context: {context}\nQuestion: {question}\nFormat: {output_format}"
    # No grammar, so no answer-type pinning: query() keeps the permissive path, which
    # is the one these tests are about.
    agent.answer_schema = None
    return agent


# The exact payload observed on the board: `position` ran away until max_tokens
# truncated the JSON mid-string.
TRUNCATED = (
    '{ "type": "text", "text": "I am not sure what you are asking about.", '
    '"position": "[-55.74, 86.31, -2.11], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0'
)


def test_unparseable_answer_raises_instead_of_hanging():
    agent = make_agent(TRUNCATED)
    try:
        agent.query("When did you leave the building?")
    except ValueError as error:
        assert "parseable" in str(error), error
    else:
        raise AssertionError("expected a ValueError once retries were exhausted")
    # Bounded: it must not keep calling the model forever.
    assert agent.chain.calls == 5, f"expected 5 attempts, made {agent.chain.calls}"


def test_missing_required_key_is_also_bounded():
    agent = make_agent('{"type": "time", "text": "5 minutes ago"}')  # no position/binary
    try:
        agent.query("When did you leave the building?")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError once retries were exhausted")
    assert agent.chain.calls == 5, f"expected 5 attempts, made {agent.chain.calls}"


def test_good_answer_returns_on_first_call():
    good = ('{"type": "time", "text": "4 minutes ago", "binary": null, '
            '"position": null, "orientation": null, "time": 4.07, "duration": null}')
    agent = make_agent(good)
    result = agent.query("When did you leave the building?")
    assert isinstance(result, AgentOutput)
    assert result.time == 4.07
    assert agent.chain.calls == 1, "a valid answer must not be retried"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
