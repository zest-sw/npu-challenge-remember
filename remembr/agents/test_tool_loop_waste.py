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
    """A 450-second recording, like the first SUBSET question."""

    now_offset = 450.0

    def search_by_text(self, query):
        return ""

    def search_by_position(self, query):
        return ""

    def search_by_time(self, query):
        return ""


class StreamingMemory(StubMemory):
    """A live robot has no end time."""

    now_offset = None


def built_agent(memory=None):
    agent = ReMEmbRAgent.__new__(ReMEmbRAgent)
    agent.tool_calls_made = set()
    agent.previous_tool_requests = ""
    agent.agent_call_count = 0
    ReMEmbRAgent.create_tools(agent, memory or StubMemory())
    return agent


def tool_names(schema):
    return [o["properties"]["tool"].get("const") for o in schema["items"]["oneOf"]]


def test_the_first_turn_can_only_search_by_text():
    """To answer anything about an event you must find the event first, and you can only
    search by time if you already know the time. 0 of 30 questions in this dataset name a
    clock time, yet whole 3-turn budgets went to retrieve_from_time."""
    allowed = tool_names(built_agent().first_turn_schema)
    assert "retrieve_from_text" in allowed, allowed
    assert "retrieve_from_time" not in allowed, allowed
    assert "retrieve_from_position" not in allowed, allowed


def test_later_turns_get_every_tool():
    """Following up on a time or position a retrieved memory gave you is what they are for."""
    allowed = tool_names(built_agent().later_turn_schema)
    assert {"retrieve_from_text", "retrieve_from_time", "retrieve_from_position"} <= set(allowed)


def test_every_turn_keeps_the_exit_signal_but_caps_it():
    """It is the loop's only way out; removing it was measured at 270s -> 389s."""
    agent = built_agent()
    for schema, name in ((agent.first_turn_schema, "first"), (agent.later_turn_schema, "later")):
        assert "__conversational_response" in tool_names(schema), name
        for option in schema["items"]["oneOf"]:
            if option["properties"]["tool"].get("const") == "__conversational_response":
                cap = option["properties"]["tool_input"]["properties"]["response"]["maxLength"]
                assert cap == 40, (name, cap)


def test_the_per_turn_schemas_are_copies():
    agent = built_agent()
    agent.first_turn_schema["items"]["oneOf"].clear()
    assert agent.later_turn_schema["items"]["oneOf"], "the variants share a list"
    assert agent.tool_call_schema["items"]["oneOf"], "a variant aliases the base schema"


def time_argument_schema(agent):
    for option in agent.tool_call_schema["items"]["oneOf"]:
        if option["properties"]["tool"].get("const") == "retrieve_from_time":
            return option["properties"]["tool_input"]["properties"]["x"]
    raise AssertionError("retrieve_from_time is missing from the schema")


def test_the_window_is_read_from_the_memory():
    assert built_agent().window_minutes == 8  # 450s rounds to 8 minutes


def test_a_time_outside_the_recording_is_ungenerable():
    schema = time_argument_schema(built_agent())
    assert schema["minimum"] == 0, schema
    assert schema["maximum"] == 8, schema
    # 48 and 27, the values actually observed, are now outside the grammar.
    for observed in (27, 48, 40):
        assert observed > schema["maximum"], observed


def test_the_tool_description_states_the_limit():
    agent = built_agent()
    described = [t for t in agent.tool_list if t.name == "retrieve_from_time"][0]
    text = described.args_schema.__fields__["x"].field_info.description
    assert "8 minutes" in text, text


def test_a_streaming_robot_gets_no_upper_bound():
    """Live, there is no end time and no window to bound by."""
    agent = built_agent(StreamingMemory())
    assert agent.window_minutes is None
    assert time_argument_schema(agent) == {"type": "number"}


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
