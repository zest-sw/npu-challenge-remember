"""Self-check: the grammar names the answer vocabulary, and the planner is told what
it already searched for.

Both of these looked like the model being weak and were not -- our own prompt and a
regex caused them.

1. The tt/ prompt said only "type: the answer category" and never named the five
   values, so the model invented `"image"` and `"yes/no"` (the latter lifted from the
   `binary` line right below it). Nothing in the schema stopped it either.
2. It said "time: a number or string for minutes-ago answers", so the model wrote
   `"11 minutes ago"`. eval() cannot parse that, so the answer scored as unanswered
   even though the model knew it was 11 minutes.
3. `previous_tool_requests` was built with `re.sub("\\{.*?\\}", "", str(args))`, which
   deletes the whole dict because its repr is brace-delimited. The planner was told
   "I previously used retrieve_from_text with the arguments: ." and repeated the
   identical search on the next turn.

    python remembr/agents/test_answer_vocabulary.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import ANSWER_SCHEMA, ANSWER_TYPES  # noqa: E402
from remembr.agents.remembr_agent import GENERATE_SCHEMA_PROMPT, ReMEmbRAgent  # noqa: E402


class StubMemory:
    def search_by_text(self, query):
        return ""

    def search_by_position(self, query):
        return ""

    def search_by_time(self, query):
        return ""


def built_agent():
    agent = ReMEmbRAgent.__new__(ReMEmbRAgent)
    ReMEmbRAgent.create_tools(agent, StubMemory())
    return agent


def test_type_is_an_enum_of_the_five_categories():
    assert ANSWER_SCHEMA["properties"]["type"] == {"enum": list(ANSWER_TYPES)}
    assert set(ANSWER_TYPES) == {"position", "time", "duration", "binary", "text"}


def test_time_and_duration_take_numbers_only():
    """A prose string here scores as unanswered, so make it ungenerable."""
    for name in ("time", "duration"):
        kinds = {branch.get("type") for branch in ANSWER_SCHEMA["properties"][name]["anyOf"]}
        assert kinds == {"number", "null"}, (name, ANSWER_SCHEMA["properties"][name])


def test_the_prompt_names_the_valid_types():
    """The schema makes them ungenerable; the prompt should still say what they are."""
    for answer_type in ANSWER_TYPES:
        assert answer_type in GENERATE_SCHEMA_PROMPT, answer_type


def test_the_prompt_no_longer_invites_a_string_time():
    assert "a number or string" not in GENERATE_SCHEMA_PROMPT, GENERATE_SCHEMA_PROMPT


def conversational_cap(schema):
    for option in schema["items"]["oneOf"]:
        if option["properties"]["tool"].get("const") == "__conversational_response":
            return option["properties"]["tool_input"]["properties"]["response"]["maxLength"]
    return None


def test_the_conversational_option_is_never_removed():
    """It is the loop's only exit signal.

    `should_continue` ends when a message has no tool_calls, and FunctionsWrapper makes
    such a message only for `__conversational_response`. Removing it was measured on the
    board and made things worse: every question then burned all 3 turns before the budget
    cap forced an answer, 270s -> 389s per question.
    """
    agent = built_agent()
    for schema, name in ((agent.tool_call_schema, "tool_call_schema"),
                         (agent.retrieval_only_tool_call_schema, "per-turn schema")):
        options = [option["properties"]["tool"].get("const")
                   for option in schema["items"]["oneOf"]]
        assert "__conversational_response" in options, (name, options)


def test_the_per_turn_schema_caps_the_conversational_reply():
    """Its length is the waste, not its existence: the reply is discarded when it
    arrives next to a real tool call (#35) and we still pay ~125 tokens to write it."""
    agent = built_agent()
    assert conversational_cap(agent.tool_call_schema) == 500
    assert conversational_cap(agent.retrieval_only_tool_call_schema) == 40


def test_the_per_turn_schema_keeps_every_retrieval_tool():
    agent = built_agent()
    full = {option["properties"]["tool"].get("const")
            for option in agent.tool_call_schema["items"]["oneOf"]}
    per_turn = {option["properties"]["tool"].get("const")
                for option in agent.retrieval_only_tool_call_schema["items"]["oneOf"]}
    assert per_turn == full, (per_turn, full)


def test_the_per_turn_schema_is_a_copy_not_a_view():
    """Editing one must not edit the other -- the cap is set on the copy only."""
    agent = built_agent()
    agent.retrieval_only_tool_call_schema["items"]["oneOf"].clear()
    assert agent.tool_call_schema["items"]["oneOf"], "the two share a list"


def test_tool_arguments_survive_the_brace_stripping():
    """The planner has to see what it already searched for, or it repeats the search."""
    args = {"x": "yellow police call pole"}
    kept = str(args).replace("{", "").replace("}", "")
    assert "yellow police call pole" in kept, kept
    assert "{" not in kept and "}" not in kept, kept
    # The old spelling, for contrast: it erased everything.
    assert re.sub(r"\{.*?\}", "", str(args)) == "", "the old regex kept something now?"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
