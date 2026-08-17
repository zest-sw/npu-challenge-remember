"""Self-check for the direct+ pipeline: question -> search -> which memory answers it.

No NPU, no Milvus, no server:

    python remembr/tests/test_direct_position_agent.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import AgentOutput  # noqa: E402
from remembr.agents.direct_position_agent import (  # noqa: E402
    combined_queries,
    memory_index_output,
    memory_index_schema,
    query_expansion_schema,
)


class FakeDoc:
    """A retrieved memory, as far as this agent is concerned: a caption and a position."""

    def __init__(self, position, caption="a caption"):
        self.metadata = {"position": position}
        self.page_content = caption


MEMORIES = [FakeDoc([1.0, 2.0, 3.0], "a bench by the window"),
            FakeDoc([-4.0, 5.5, -6.0], "a paved walkway"),
            FakeDoc([7, 8, 9], "a bike rack")]


def test_schema_offers_exactly_the_memories_on_hand():
    """The grammar is the whole contract: one of these numbers, nothing else generable."""
    assert memory_index_schema(3) == {"type": "integer", "enum": [1, 2, 3]}
    assert memory_index_schema(1) == {"type": "integer", "enum": [1]}


def test_a_memory_number_becomes_that_memory_s_position():
    out = memory_index_output(2, MEMORIES)
    assert isinstance(out, AgentOutput)
    assert out.type == "position"
    assert out.position == [-4.0, 5.5, -6.0]
    assert out.time is None and out.duration is None and out.binary is None


def test_ints_are_coerced_to_floats():
    # eval does numpy arithmetic on this, and MemoryItem declares a list of numbers.
    assert memory_index_output(3, MEMORIES).position == [7.0, 8.0, 9.0]


def test_answers_outside_the_offer_are_rejected():
    # The grammar should prevent all of these, but it only binds while the tt server is
    # enforcing it -- a bad number must fail loudly, not index some other memory.
    for bad in (0, 4, -1, 2.0, "2", None, True, [2]):
        try:
            memory_index_output(bad, MEMORIES)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_a_memory_without_a_coordinate_is_rejected():
    try:
        memory_index_output(1, [FakeDoc([1.0, 2.0])])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a 2-element position")


from remembr.agents.direct_position_agent import (  # noqa: E402
    DirectPositionAgent,
    current_position,
    split_question,
)


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChain:
    """Stands in for ChatOpenAI. Counts calls so the no-retry rule is testable, and keeps
    the bound schema so the grammar can be asserted on."""

    def __init__(self, content):
        self.content = content
        self.calls = 0
        self.prompts = []
        self.schemas = []

    def bind(self, **kwargs):
        self.schemas.append(kwargs["response_format"]["json_schema"]["schema"])
        return self

    def invoke(self, inputs):
        self.calls += 1
        self.prompts.append(inputs)
        return FakeReply(self.content)


class FakeMemory:
    """Records what was searched for, and hands back memories to choose between."""

    def __init__(self, memories=MEMORIES):
        self.searched = []
        self.memories = memories

    def search_by_text(self, query):
        self.searched.append(query)
        return []

    def search_by_text_multi(self, queries):
        self.searched.append(list(queries))
        return []

    def get_working_memory(self):
        return self.memories

    def memory_to_string(self, memory_list):
        # The real one formats a list and joins with blank lines; the agent passes one
        # doc at a time so it can number them.
        return "\n\n".join(
            f"At time=2023-01-16 07:58:58, the robot was at an average position of "
            f"{doc.metadata['position']}.The robot saw the following: a caption\n"
            for doc in memory_list)


def make_agent(reply, memories=MEMORIES):
    """Build the agent without __init__ (which would build an LLM client)."""
    agent = DirectPositionAgent.__new__(DirectPositionAgent)
    agent.llm_type = "tt/fake"
    agent.text_query_mode = "direct"
    agent.chain = FakeChain(reply)
    agent.query_chain = None
    agent.memory = FakeMemory(memories)
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
    agent = make_agent("2")
    out = agent.query(PREAMBLE_QUESTION)
    # The preamble names the robot's own position and time. The retriever has no use
    # for either, and they only dilute the embedding, so they never reach it.
    assert agent.memory.searched == [ASKED], agent.memory.searched
    assert agent.chain.calls == 1, f"expected 1 call, made {agent.chain.calls}"
    # Memory 2's coordinate, looked up here rather than typed by the model.
    assert out.position == [-4.0, 5.5, -6.0]


def test_multi_query_keeps_original_and_searches_fused_candidates():
    agent = make_agent("2")
    agent.text_query_mode = "multi"
    agent.query_chain = FakeChain(json.dumps({
        "target_query": "yellow emergency call pole help station post",
        "visual_query": "A yellow emergency call post is visible.",
    }))

    out = agent.query(PREAMBLE_QUESTION)

    assert agent.memory.searched == [[
        ASKED,
        "yellow emergency call pole help station post",
        "A yellow emergency call post is visible",
    ]]
    assert agent.query_chain.calls == 1
    assert agent.chain.calls == 1
    assert out.position == [-4.0, 5.5, -6.0]


def test_multi_query_schema_bounds_every_generated_string():
    schema = query_expansion_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"target_query", "visual_query"}
    for field in schema["required"]:
        assert schema["properties"][field]["type"] == "string"
        assert schema["properties"][field]["maxLength"] == 180


def test_combined_queries_removes_packaging_blanks_and_duplicates():
    assert combined_queries("Where is the bench?", {
        "target_query": '  "empty bench". ',
        "visual_query": "Where is the bench?",
    }) == ["Where is the bench?", "empty bench"]


def test_combined_queries_compacts_aliases_and_rejects_hallucinated_visual_facts():
    queries = combined_queries("Where is the yellow police call pole?", {
        "target_query": (
            "police call pole, yellow pole, traffic pole, street pole, pole, "
            "the pole, extra sixth name"
        ),
        "visual_query": "A blue police car is beside the yellow police call pole.",
    })
    assert queries == [
        "Where is the yellow police call pole?",
        "police call pole, yellow pole, traffic pole, street pole, pole",
    ]


def test_combined_queries_keeps_a_visual_rewrite_grounded_in_question_and_aliases():
    queries = combined_queries("Where is the statue in front of Gates Building?", {
        "target_query": "statue, monument, sculpture, public art",
        "visual_query": "The sculpture is in front of Gates Building.",
    })
    assert queries[-1] == "The sculpture is in front of Gates Building"


def test_multi_query_generation_failure_falls_back_to_original_search():
    agent = make_agent("1")
    agent.text_query_mode = "multi"
    agent.query_chain = FakeChain("not JSON")

    agent.query(PREAMBLE_QUESTION)

    assert agent.memory.searched == [[ASKED]]
    assert agent.query_chain.calls == 1


def test_the_grammar_offers_only_the_memories_that_came_back():
    agent = make_agent("1", memories=MEMORIES[:2])
    agent.query(PREAMBLE_QUESTION)
    assert agent.chain.schemas == [{"type": "integer", "enum": [1, 2]}], agent.chain.schemas


def test_memories_are_numbered_and_carry_their_distance():
    """Numbered so the answer can name one, measured so "nearest" needs no arithmetic.

    Measured on the real geometry -- four candidates at 97.1/104.5/122.2/123.0 m -- the
    model scored 1/5 comparing coordinates and 4/5 reading distances, and when it failed
    it answered with the last memory whatever the arrangement.
    """
    agent = make_agent("1")
    agent.query(PREAMBLE_QUESTION)
    prompt = agent.chain.prompts[0]

    # The preamble puts the robot at the origin, so each distance is the memory's own norm.
    for number, expected in ((1, "3.7"), (2, "9.1"), (3, "13.9")):
        assert f"{number}. distance_from_you={expected}m." in prompt, prompt
    # The coordinate itself is gone -- it is what the model kept mis-reading.
    assert "[-4.0, 5.5, -6.0]" not in prompt, prompt
    assert "how far it is from you now" in prompt, prompt


def test_without_a_stated_position_it_falls_back_to_coordinates():
    """No preamble means no distance to give, so do not claim to give one."""
    agent = make_agent("1")
    agent.query(ASKED)
    prompt = agent.chain.prompts[0]
    assert "distance_from_you" not in prompt, prompt
    assert "1. At time=" in prompt, prompt
    assert "how far it is from you now" not in prompt, prompt


def test_an_unreadable_position_falls_back_rather_than_raising():
    """`current_position` promises None when the preamble does not give a usable point.

    A trailing empty field still matches the bracket, and float('') would otherwise raise
    out of prompt building -- turning a cosmetic preamble typo into an unanswered question.
    """
    for preamble in ("you are located at [1.0, 2.0, ].",
                     "you are located at [1.0, 2.0].",
                     "you are somewhere out there."):
        assert current_position(preamble) is None, preamble
    assert current_position("you are located at [1.0, 2.0, 3.0].") == [1.0, 2.0, 3.0]


def test_a_memory_without_a_coordinate_stops_the_distance_line():
    """The same invariant `memory_index_output` enforces, checked on the way in too.

    `zip` would otherwise drop a dimension silently and print a too-small distance, which
    is worse than failing: it steers the model onto exactly that memory.
    """
    agent = make_agent("1", memories=[FakeDoc([1.0, 2.0], "a bench")])
    try:
        agent.query(PREAMBLE_QUESTION)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a 2-element position")
    assert agent.chain.calls == 0, "should fail before spending an LLM call"


def test_the_preamble_is_kept_but_out_of_the_question_sentence():
    """Measured: a coordinate inside the question sentence becomes the answer.

    Asked the whole question, the 8B answered with its own current position and was
    205 m out; with the preamble hoisted to its own labelled block it was exact. The
    preamble is not deleted, because "nearest X" and "how long ago" need it.
    """
    agent = make_agent("1")
    agent.query(PREAMBLE_QUESTION)
    prompt = agent.chain.prompts[0]

    # Still available to the model...
    assert PREAMBLE in prompt, prompt
    assert "Your current situation:" in prompt, prompt
    # ...but the question the model reads is only the asked line.
    assert prompt.rstrip().endswith(f"Question: {ASKED}"), prompt
    assert f"Question: {PREAMBLE}" not in prompt, prompt


def test_a_question_with_no_preamble_gets_no_situation_block():
    agent = make_agent("1")
    agent.query(ASKED)
    assert "Your current situation:" not in agent.chain.prompts[0], agent.chain.prompts[0]


def test_a_parse_failure_raises_immediately_no_retry():
    agent = make_agent("2,")  # trailing comma -- json.loads fails
    try:
        agent.query(PREAMBLE_QUESTION)
    except Exception:
        pass
    else:
        raise AssertionError("expected a raise on unparseable output")
    assert agent.chain.calls == 1, f"no retries allowed, made {agent.chain.calls}"


def test_a_number_outside_the_offer_raises_immediately():
    agent = make_agent("9")  # valid JSON, no such memory
    try:
        agent.query(PREAMBLE_QUESTION)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a memory that does not exist")
    assert agent.chain.calls == 1


def test_no_memories_means_no_answer():
    """An empty retrieval leaves nothing to name, so it must not reach the model."""
    agent = make_agent("1", memories=[])
    try:
        agent.query(PREAMBLE_QUESTION)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError when retrieval came back empty")
    assert agent.chain.calls == 0, "no memories, so no LLM call should have been made"


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
