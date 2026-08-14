"""Self-check: the grammar names the answer vocabulary, and the planner is told what it
already searched for. Both failures came from our prompt and a regex, not the model:
it invented `"image"` for an unnamed `type`, wrote `"11 minutes ago"` into a string
`time`, and was handed "with the arguments: ." after a regex ate the dict.

    python remembr/tests/test_answer_vocabulary.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import ANSWER_SCHEMA, ANSWER_TYPES  # noqa: E402
from remembr.agents.remembr_agent import GENERATE_SCHEMA_PROMPT  # noqa: E402


def read(relative):
    return open(os.path.join(os.path.dirname(__file__), "..", "..", relative)).read()




def test_type_is_an_enum_of_the_five_categories():
    assert ANSWER_SCHEMA["properties"]["type"] == {"enum": list(ANSWER_TYPES)}
    assert set(ANSWER_TYPES) == {"position", "time", "duration", "binary", "text"}


def test_time_and_duration_take_numbers_only():
    """A prose string here scores as unanswered, so make it ungenerable."""
    for name in ("time", "duration"):
        kinds = {branch.get("type") for branch in ANSWER_SCHEMA["properties"][name]["anyOf"]}
        assert kinds == {"number", "null"}, (name, ANSWER_SCHEMA["properties"][name])



def test_the_prompt_no_longer_invites_a_string_time():
    assert "a number or string" not in GENERATE_SCHEMA_PROMPT, GENERATE_SCHEMA_PROMPT





def test_the_planner_is_told_which_arguments_it_used():
    """Guards a revert to the regex, which took the dict with the braces.

    A source check because the line sits inside `agent()`, which needs a live LLM."""
    source = read("remembr/agents/remembr_agent.py")
    call = source.split("previous_tool_requests +=")[0].rsplit("args =", 1)[1]
    assert "re.sub" not in call, call
    assert ".replace('{', '').replace('}', '')" in call, call


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
