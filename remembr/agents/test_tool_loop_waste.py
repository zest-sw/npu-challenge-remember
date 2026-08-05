"""Self-check: the tool loop cannot spend a turn on a search it already ran, and cannot
ask for a time outside the recording.

Both were measured on the board with a 7.5-minute recording and a 3-turn budget:

    turn1 retrieve_from_time(27)   t-1170s, outside -> window boundary
    turn2 retrieve_from_time(8)    t-30s,   outside -> same four memories
    turn3 retrieve_from_time(8)    identical to turn 2

    turn1 retrieve_from_time(48)   outside
    turn2 retrieve_from_time(8)    inside
    turn3 retrieve_from_time(48)   identical to turn 1

`previous_tool_requests` does tell the planner what it already used -- that was #48's fix
and it landed -- and it repeats anyway. Prose does not hold; the grammar and the loop do.
So the out-of-range number is made ungenerable and the repeat is dropped in code.

No board, no server, no Milvus:

    python remembr/agents/test_tool_loop_waste.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.remembr_agent import ReMEmbRAgent  # noqa: E402


class StubMemory:
    """The first SUBSET question's clock anchor."""

    now_clock = "2023-01-16 08:03:02"
    now_epoch = 1673884983.0

    def search_by_text(self, query):
        return ""

    def search_by_position(self, query):
        return ""

    def search_by_time(self, query):
        return ""


def built_agent(memory=None):
    agent = ReMEmbRAgent.__new__(ReMEmbRAgent)
    agent.tool_calls_made = set()
    agent.previous_tool_requests = ""
    agent.agent_call_count = 0
    ReMEmbRAgent.create_tools(agent, memory or StubMemory())
    return agent


AGENT_PROMPT = os.path.join(os.path.dirname(__file__), "..", "prompts",
                            "agent_system_prompt.txt")


def agent_prompt_text():
    return open(AGENT_PROMPT).read()


def tool_names(schema):
    return [o["properties"]["tool"].get("const") for o in schema["items"]["oneOf"]]


def test_with_nothing_retrieved_only_text_search_is_offered():
    """Searching by a time or a position needs a value, and the only honest source is a
    retrieved memory -- the question's own values are ruled out (its position is where you
    stand now, its time is the present). With nothing retrieved there is no value to search
    by.

    Enforced rather than asked, because it was measured both ways. With the rewritten
    prompt saying "Start with retrieve_from_text" in as many words, the planner still
    called retrieve_from_time 16 times against retrieve_from_text 9 times over 9 questions
    -- on a dataset where 0 of 30 questions name a clock time."""
    allowed = tool_names(built_agent().no_memory_yet_schema)
    assert set(allowed) == {"retrieve_from_text", "__conversational_response"}, allowed


def test_once_a_memory_exists_every_tool_is_offered():
    """Following up on a time or position a memory gave you is what those tools are for."""
    allowed = tool_names(built_agent().turn_schema)
    assert {"retrieve_from_text", "retrieve_from_time", "retrieve_from_position",
            "__conversational_response"} == set(allowed), allowed


def test_the_exit_signal_survives_in_both_schemas_but_is_capped():
    """It is the loop's only way out; removing it was measured at 270s -> 389s."""
    agent = built_agent()
    for schema in (agent.turn_schema, agent.no_memory_yet_schema):
        assert "__conversational_response" in tool_names(schema), tool_names(schema)
    for option in agent.turn_schema["items"]["oneOf"]:
        if option["properties"]["tool"].get("const") == "__conversational_response":
            cap = option["properties"]["tool_input"]["properties"]["response"]["maxLength"]
            assert cap == 40, cap


def test_the_schema_variants_are_copies():
    agent = built_agent()
    agent.turn_schema["items"]["oneOf"].clear()
    assert agent.no_memory_yet_schema["items"]["oneOf"], "the variants share a list"
    assert agent.tool_call_schema["items"]["oneOf"], "a variant aliases the base schema"


def test_the_prompt_tells_it_to_start_with_text():
    """Now that this is instruction rather than grammar, the instruction has to be there."""
    assert "Start with retrieve_from_text" in agent_prompt_text()


def test_the_prompt_and_the_tool_agree_on_the_time_format():
    """They did not for an hour: the tool took minutes-ago while the prompt still said
    H:M:S. The model was given two different instructions. Both are clock times now."""
    text = agent_prompt_text()
    assert "HH:MM:SS clock time" in text, text
    described = [t for t in built_agent().tool_list if t.name == "retrieve_from_time"][0]
    assert "HH:MM:SS" in described.args_schema.__fields__["x"].field_info.description


def test_the_prompt_does_not_let_position_search_copy_the_question():
    """Turn 1 searched by the question's own current position, 77.2 m from the answer.
    The description used to permit exactly that: "copied from a coordinate the question
    or a retrieved memory gave you"."""
    for text, where in ((agent_prompt_text(), "agent_system_prompt.txt"),
                        (open(os.path.join(os.path.dirname(__file__),
                                           "remembr_agent.py")).read(), "remembr_agent.py")):
        assert "coordinate the question" not in text, where
        assert "RETRIEVED MEMORY" in text, where


def test_the_prompt_does_not_demand_extra_tool_calls():
    """"Call at least two tools ... better to have more information than less" is why
    whole 3-turn budgets were spent on searches whose results went unused."""
    text = agent_prompt_text()
    assert "at least two tools" not in text, text
    assert "better to have more information than less" not in text, text


def test_the_prompt_says_what_to_do_when_information_is_missing():
    """Two rules used to say the same thing ("if you are able to answer" / "if you do have
    enough information"), so nothing covered the case where information is missing -- the
    only case where a retrieval tool is the right move."""
    text = agent_prompt_text()
    assert "If it does not" in text, text[:200]
    assert "If it answers the question" in text, text[:200]


class FakeResponse:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.content = ""


def dedupe(agent, calls):
    """The part of agent() that filters repeats, without running an LLM."""
    response = FakeResponse(list(calls))
    fresh = []
    for tool_call in response.tool_calls:
        signature = (tool_call["name"], repr(tool_call["args"]))
        if signature in agent.tool_calls_made:
            continue
        agent.tool_calls_made.add(signature)
        fresh.append(tool_call)
    if len(fresh) != len(response.tool_calls):
        response.tool_calls = fresh
    return response


def call(name="retrieve_from_time", **args):
    return {"name": name, "args": args or {"x": 8}}


def test_a_repeated_search_is_dropped():
    agent = built_agent()
    first = dedupe(agent, [call(x=8)])
    assert len(first.tool_calls) == 1

    again = dedupe(agent, [call(x=8)])
    assert again.tool_calls == [], again.tool_calls


def test_an_all_repeat_turn_leaves_no_tool_calls_so_the_loop_ends():
    """`should_continue` routes a message with no tool_calls to generate, so a fully
    repeated turn becomes an early exit instead of a wasted one."""
    from remembr.agents.remembr_agent import should_continue

    agent = built_agent()
    dedupe(agent, [call(x=8), call(x=3)])
    repeat = dedupe(agent, [call(x=8), call(x=3)])
    assert repeat.tool_calls == []
    assert should_continue({"messages": [repeat]}) == "end"


def test_a_different_argument_still_runs():
    agent = built_agent()
    dedupe(agent, [call(x=8)])
    other = dedupe(agent, [call(x=3)])
    assert len(other.tool_calls) == 1, other.tool_calls


def test_the_same_argument_on_a_different_tool_still_runs():
    agent = built_agent()
    dedupe(agent, [call("retrieve_from_time", x=8)])
    other = dedupe(agent, [call("retrieve_from_text", x=8)])
    assert len(other.tool_calls) == 1, other.tool_calls


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
