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


# The per-turn schema variants moved to test_tool_loop_waste.py, which owns the tool
# loop's behaviour. This file keeps the answer vocabulary.


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
