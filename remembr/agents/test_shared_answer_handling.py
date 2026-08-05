"""Self-check: the answer paths share their logic instead of each keeping a copy.

An audit found six things that should have been common and were not, and the same class
of bug kept coming back because a fix landed in one place:

    parse_json            4 copies, and eval.py's used eval() instead of json.loads --
                          running model output as Python
    required-key check    NonAgent and VLMNonAgent checked 4 keys, ReMEmbRAgent 5 (it
                          included `duration`), so one malformed answer passed on one
                          path and failed on another
    position shape check  only ReMEmbRAgent verified three numbers
    answer-type pinning   only NonAgent. ReMEmbRAgent answered `type="position"` for a
                          time question, scored unanswered, and eval retried the whole
                          3-turn tool loop five times -- up to 20 LLM calls per question
    answer prompts        three near-identical files; a fix landed in one
    ANSWER_SCHEMA         two copies (fixed earlier, guarded in test_schema_constraints)

These tests fail if any of them splits again. No board, no server, no Milvus:

    python remembr/agents/test_shared_answer_handling.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents import agent as agent_module  # noqa: E402
from remembr.agents.agent import (  # noqa: E402
    ANSWER_SCHEMA,
    ANSWER_TYPES,
    TYPE_FIELD,
    schema_for_type,
    validate_answer,
)

AGENT_FILES = (
    "remembr/agents/non_agent.py",
    "remembr/agents/remembr_agent.py",
    "remembr/agents/vlm_non_agent.py",
    "remembr/scripts/eval.py",
)

ANSWER_PROMPTS = (
    "remembr/prompts/non_agent_system_prompt.txt",
    "remembr/prompts/generate_system_prompt.txt",
    "remembr/prompts/vlm_non_agent_system_prompt.txt",
)

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def read(relative):
    return open(os.path.join(REPO, relative)).read()


def test_parse_json_is_defined_once():
    for path in AGENT_FILES:
        assert "def parse_json" not in read(path), (
            f"{path} defines its own parse_json; import it from remembr.utils.util"
        )


def test_nobody_evals_model_output():
    """eval.py used to run the model's JSON through eval(). That is arbitrary code."""
    for path in AGENT_FILES:
        assert "return eval(" not in read(path), f"{path} eval()s model output"


def test_the_required_key_check_is_not_reimplemented():
    for path in AGENT_FILES:
        assert "keys_to_check_for" not in read(path), (
            f"{path} has its own required-key list; use validate_answer()"
        )


def test_validate_answer_catches_what_each_path_used_to_check_separately():
    complete = {name: None for name in ANSWER_SCHEMA["required"]}
    validate_answer(complete)  # all keys present, position null -> fine

    missing = {"text": "x"}
    try:
        validate_answer(missing)
    except ValueError as error:
        assert "missing" in str(error), error
    else:
        raise AssertionError("a missing key should raise")

    # Only ReMEmbRAgent used to check this.
    try:
        validate_answer(dict(complete, position=[1.0, 2.0]))
    except ValueError as error:
        assert "three numbers" in str(error), error
    else:
        raise AssertionError("a two-element position should raise")


def test_both_answer_paths_pin_the_settled_type():
    """The gap that cost up to 20 LLM calls per remembr+ question."""
    for path in ("remembr/agents/non_agent.py", "remembr/agents/remembr_agent.py"):
        source = read(path)
        assert "settle_answer_type" in source, f"{path} does not settle the answer type"
        assert "schema_for_type" in source, f"{path} does not pin the type's field"


def test_schema_for_type_pins_the_type_for_every_type():
    """Every type in ANSWER_TYPES, not just the four in TYPE_FIELD. This test used to
    iterate TYPE_FIELD, which is exactly the set the bug did not cover: `text` is not in
    it, so `schema_for_type("text")` returned the shared permissive schema and the model
    answered `type="time", time=5` to a "where" question whose type had settled to text."""
    for answer_type in ANSWER_TYPES:
        schema = schema_for_type(answer_type)
        assert schema["properties"]["type"] == {"const": answer_type}, answer_type
        assert schema is not ANSWER_SCHEMA, answer_type


def test_schema_for_type_pins_the_typed_field_non_null():
    for answer_type, field in TYPE_FIELD.items():
        schema = schema_for_type(answer_type)
        branches = schema["properties"][field].get("anyOf", [schema["properties"][field]])
        assert not any(b.get("type") == "null" for b in branches), (answer_type, branches)


def test_the_type_is_settled_at_temperature_zero():
    """Choosing among five words is classification, not generation. At the eval default
    of 0.7 it sampled `text` for "Where is X" against instructions that say that is
    always position -- and the whole answer shape follows from this one word."""
    source = read("remembr/agents/agent.py")
    settle = source.split("def settle_answer_type")[1].split("\ndef ")[0]
    assert "temperature=0" in settle, settle


def test_schema_for_type_leaves_the_shared_schema_alone():
    before = repr(ANSWER_SCHEMA)
    for answer_type in ANSWER_TYPES:
        schema_for_type(answer_type)
    assert repr(ANSWER_SCHEMA) == before, "schema_for_type mutated the shared schema"


def test_every_answer_prompt_names_all_five_types():
    """One prompt listed four and omitted duration, for which questions exist."""
    for path in ANSWER_PROMPTS:
        source = read(path)
        assert "position, binary, time, duration, or text" in source, path


def test_every_answer_prompt_forbids_answering_with_the_current_position():
    """The dominant cause of position error: 5 of 7 wrong optimal+ answers were the
    question's own current position copied verbatim (#23)."""
    for path in ANSWER_PROMPTS:
        assert "never the answer" in read(path), (
            f"{path} still lets the model fall back to its current position"
        )


def test_no_answer_prompt_shows_time_or_orientation_as_a_string():
    """A prose time scores as unanswered; a string orientation was the runaway field."""
    for path in ANSWER_PROMPTS:
        source = read(path)
        assert '"time": "5.3"' not in source, path
        assert "euler_rotation" not in source, path
        assert '"[-.92]"' not in source, path


def test_the_trace_summary_is_not_reimplemented():
    """It existed twice: a helper in non_agent.py and an inline join in remembr_agent.py."""
    for path in ("remembr/agents/non_agent.py", "remembr/agents/remembr_agent.py"):
        source = read(path)
        assert "answer_summary(" in source, f"{path} does not use the shared summary"
        assert "def _answer_summary" not in source, f"{path} kept its own copy"


def test_answer_summary_takes_both_shapes():
    """NonAgent holds an AgentOutput at that point, ReMEmbRAgent a dict."""
    from remembr.agents.agent import AgentOutput, answer_summary

    output = AgentOutput(type="position", text="x", binary=None,
                         position=[1.0, 2.0, 0.0], orientation=None, duration=None, time=None)
    assert "position=[1.0, 2.0, 0.0]" in answer_summary(output)
    assert "time=4.07" in answer_summary({"type": "time", "time": 4.07, "position": None})
    # Empty fields stay out of the line.
    assert "binary" not in answer_summary(output)


def test_the_helpers_live_next_to_the_schema():
    """Answer shape, answer vocabulary and answer validation are one concern."""
    for name in ("ANSWER_SCHEMA", "ANSWER_TYPES", "TYPE_FIELD", "NON_NULL_FIELD",
                 "validate_answer", "schema_for_type", "settle_answer_type",
                 "answer_summary", "asked_line"):
        assert hasattr(agent_module, name), f"agent.py is missing {name}"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
