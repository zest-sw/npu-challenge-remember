"""Self-check for the direct+ pipeline: question -> search -> a bare [x, y, z].

No NPU, no Milvus, no server:

    python remembr/tests/test_direct_position_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import AgentOutput  # noqa: E402
from remembr.agents.direct_position_agent import (  # noqa: E402
    POSITION_SCHEMA,
    position_output,
)


def test_schema_is_exactly_three_numbers():
    """The grammar is the whole contract: three numbers, nothing else generable."""
    assert POSITION_SCHEMA == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }, POSITION_SCHEMA


def test_a_bare_array_becomes_a_position_answer():
    out = position_output([1, 2.5, -3])
    assert isinstance(out, AgentOutput)
    assert out.type == "position"
    assert out.position == [1.0, 2.5, -3.0]
    assert out.time is None and out.duration is None and out.binary is None


def test_wrong_shapes_are_rejected():
    # Shapes the old loose-string field let through and eval then mis-scored.
    for bad in ([1, 2], [[1, 2, 3]], "at [1,2,3]", [1, 2, "a"], [True, 1, 2], None):
        try:
            position_output(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


from remembr.agents.direct_position_agent import (  # noqa: E402
    DirectPositionAgent,
    split_question,
)


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChain:
    """Stands in for ChatOpenAI. Counts calls so the no-retry rule is testable."""

    def __init__(self, content):
        self.content = content
        self.calls = 0
        self.prompts = []

    def invoke(self, inputs):
        self.calls += 1
        self.prompts.append(inputs)
        return FakeReply(self.content)


class FakeMemory:
    """Records what was searched for, so the question-as-query rule is testable."""

    def __init__(self):
        self.searched = []

    def search_by_text(self, query):
        self.searched.append(query)
        return []

    def get_working_memory(self):
        return []

    def memory_to_string(self, _memory):
        return "At time=2023-01-16 07:58:58, the robot was at [1.0, 2.0, 0.0]."


def make_agent(reply):
    """Build the agent without __init__ (which would build an LLM client)."""
    agent = DirectPositionAgent.__new__(DirectPositionAgent)
    agent.llm_type = "tt/fake"
    agent.chain = FakeChain(reply)
    agent.memory = FakeMemory()
    return agent


# The shape form_question_jsons.py:214 actually produces: preamble sentences, a
# newline, then the asked line.
PREAMBLE = (
    "You started moving at 07:58:58. The current time is 08:10:00 and you are "
    "located at [0.0, 0.0, 0.0]."
)
ASKED = "Where is the yellow police call pole?"
PREAMBLE_QUESTION = f"{PREAMBLE} \n {ASKED}"


def test_split_question_separates_the_preamble_from_the_asked_line():
    assert split_question(PREAMBLE_QUESTION) == (PREAMBLE, ASKED)


def test_split_question_handles_a_question_with_no_preamble():
    # Nothing to hoist; the whole thing is what was asked.
    assert split_question(ASKED) == ("", ASKED)
    assert split_question("  ") == ("", "")


def test_query_searches_with_the_asked_line_only_and_calls_once():
    agent = make_agent("[1.0, 2.0, 3.0]")
    out = agent.query(PREAMBLE_QUESTION)
    # The preamble names the robot's own position and time. The retriever has no use
    # for either, and they only dilute the embedding, so they never reach it.
    assert agent.memory.searched == [ASKED], agent.memory.searched
    assert agent.chain.calls == 1, f"expected 1 call, made {agent.chain.calls}"
    assert out.position == [1.0, 2.0, 3.0]


def test_the_preamble_is_kept_but_out_of_the_question_sentence():
    """Measured: a coordinate inside the question sentence becomes the answer.

    Asked the whole question, the 8B answered with its own current position and was
    205 m out; with the preamble hoisted to its own labelled block it was exact. The
    preamble is not deleted, because "nearest X" and "how long ago" need it.
    """
    agent = make_agent("[1.0, 2.0, 3.0]")
    agent.query(PREAMBLE_QUESTION)
    prompt = agent.chain.prompts[0]

    # Still available to the model...
    assert PREAMBLE in prompt, prompt
    assert "Your current situation:" in prompt, prompt
    # ...but the question the model reads is only the asked line.
    assert prompt.rstrip().endswith(f"Question: {ASKED}"), prompt
    assert f"Question: {PREAMBLE}" not in prompt, prompt


def test_a_question_with_no_preamble_gets_no_situation_block():
    agent = make_agent("[1.0, 2.0, 3.0]")
    agent.query(ASKED)
    assert "Your current situation:" not in agent.chain.prompts[0], agent.chain.prompts[0]


def test_a_parse_failure_raises_immediately_no_retry():
    agent = make_agent("[1.0, 2.0")  # truncated -- json.loads fails
    try:
        agent.query(PREAMBLE_QUESTION)
    except Exception:
        pass
    else:
        raise AssertionError("expected a raise on unparseable output")
    assert agent.chain.calls == 1, f"no retries allowed, made {agent.chain.calls}"


def test_a_wrong_shape_also_raises_immediately():
    agent = make_agent("null")  # valid JSON, wrong shape
    try:
        agent.query(PREAMBLE_QUESTION)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a non-array answer")
    assert agent.chain.calls == 1


def test_non_tt_backend_is_refused():
    try:
        DirectPositionAgent(llm_type="llama3")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a non-tt/ backend")


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
