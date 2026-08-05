"""Self-check: every constrained answer field must be constrainable by the grammar.

lm-format-enforcer silently drops the sibling keywords on a JSON-Schema type union.
Measured with the real Llama-3.1 tokenizer, walking a string past its cap:

    {'type': 'string', 'maxLength': 40}                        stops at 40
    {'anyOf': [{'type':'string','maxLength':40},{'type':'null'}]}  stops at 40
    {'type': ['string', 'null'], 'maxLength': 40}              60+ and counting

So a nullable field written the union way is unbounded in practice. On the board that
showed up as `orientation` emitting
`"[-0.92, 0.21, 0.12, 0.12, 0.21, -0.92, 0.12, ...` until it ate max_tokens, which
truncates the JSON mid-string and costs the whole answer. `time`/`duration` were
unions with no maxLength at all.

This test needs neither lm-format-enforcer nor a board: it asserts the *shape* of the
schemas, which is the property the enforcer depends on.

    python remembr/agents/test_schema_constraints.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.remembr_agent import ANSWER_SCHEMA  # noqa: E402

# NonAgent builds its schema inside __init__, which would open a network client, so
# lift it the same way the module does without constructing the agent.
NON_AGENT_SOURCE = os.path.join(os.path.dirname(__file__), "non_agent.py")


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


def test_agent_answer_schema_is_enforceable():
    check(ANSWER_SCHEMA, "ANSWER_SCHEMA")


def test_non_agent_answer_schema_is_enforceable():
    # The schema is a literal inside __init__; eval it out of the source rather than
    # instantiate NonAgent (which builds a ChatOpenAI client).
    source = open(NON_AGENT_SOURCE).read()
    start = source.index("answer_schema = {")
    end = source.index("\n            }", start) + len("\n            }")
    schema = eval(source[start + len("answer_schema = ") : end])  # noqa: S307
    check(schema, "NonAgent.answer_schema")


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
