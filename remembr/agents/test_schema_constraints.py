"""Self-check: the answer schema is enforceable, and it is the only one.

Two failures this guards, both of which already happened:

1. **A constraint the grammar ignores.** lm-format-enforcer silently drops the sibling
   keywords on a JSON-Schema type union. Measured with the real Llama-3.1 tokenizer,
   walking a string past its cap:

       {'type': 'string', 'maxLength': 40}                            stops at 40
       {'anyOf': [{'type':'string','maxLength':40},{'type':'null'}]}   stops at 40
       {'type': ['string', 'null'], 'maxLength': 40}                  60+ and counting

   `orientation` was the union form and ran away on the board until it ate max_tokens,
   truncating the JSON mid-string and costing the whole answer.

2. **Two full-answer copies drifting.** The full schema used to be written separately
   in each agent, and they disagreed twice. ReMEmbRAgent still uses that shared contract;
   NonAgent now deliberately has a smaller position-only response contract, guarded by
   `test_position_only_schema.py`.

Needs neither lm-format-enforcer nor a board: it asserts the *shape* of the schema,
which is the property the enforcer depends on.

    python remembr/agents/test_schema_constraints.py
"""
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import ANSWER_SCHEMA, AgentOutput  # noqa: E402


def branches(field):
    """The alternatives a field may take, whether spelled anyOf or as a bare type."""
    return field["anyOf"] if "anyOf" in field else [field]


def string_branches(field):
    for branch in branches(field):
        types = branch.get("type")
        types = [types] if isinstance(types, str) else (types or [])
        if "string" in types and "enum" not in branch:
            yield branch


def check(schema, label):
    for name, field in schema["properties"].items():
        assert not isinstance(field.get("type"), list), (
            f"{label}.{name} uses a type union; lm-format-enforcer drops its "
            f"constraints. Write it as anyOf instead."
        )
        for branch in string_branches(field):
            assert "maxLength" in branch, (
                f"{label}.{name} has a string branch with no maxLength, so the model "
                f"can run to max_tokens inside it and truncate the JSON."
            )


def test_answer_schema_is_enforceable():
    check(ANSWER_SCHEMA, "ANSWER_SCHEMA")


def test_schema_covers_exactly_agent_output():
    """The drift guard: the grammar and the dataclass must describe the same answer."""
    declared = {field.name for field in fields(AgentOutput)}
    in_schema = set(ANSWER_SCHEMA["properties"])
    assert in_schema == declared, (
        f"schema and AgentOutput disagree: only in schema {in_schema - declared}, "
        f"only in AgentOutput {declared - in_schema}"
    )
    assert set(ANSWER_SCHEMA["required"]) == declared, ANSWER_SCHEMA["required"]
    assert ANSWER_SCHEMA["additionalProperties"] is False


def test_remembr_agent_uses_the_shared_full_answer_schema():
    """The remaining full-answer agent must not grow a second copy."""
    from remembr.agents import remembr_agent

    assert remembr_agent.ANSWER_SCHEMA is ANSWER_SCHEMA


def test_orientation_is_a_number_like_agent_output_declares():
    field = ANSWER_SCHEMA["properties"]["orientation"]
    kinds = {branch.get("type") for branch in branches(field)}
    assert kinds == {"number", "null"}, field


def test_position_is_exactly_three_numbers():
    array = next(b for b in branches(ANSWER_SCHEMA["properties"]["position"])
                 if b.get("type") == "array")
    assert array["items"] == {"type": "number"}, array
    assert array["minItems"] == array["maxItems"] == 3, array


def test_the_check_catches_the_union_form():
    """The guard is worthless if it passes the broken spelling."""
    broken = {"properties": {"orientation": {"type": ["string", "null"], "maxLength": 40}}}
    try:
        check(broken, "broken")
    except AssertionError:
        return
    raise AssertionError("check() accepted a type union, which is what it exists to reject")


def test_the_check_catches_an_unbounded_string_branch():
    broken = {"properties": {"time": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
    try:
        check(broken, "broken")
    except AssertionError:
        return
    raise AssertionError("check() accepted an unbounded string branch")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
