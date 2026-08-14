"""Self-check: once the answer type is settled, the grammar cannot leave its field null.

Measured on the board, asked where a statue was, the model replied

    type="text", position=null,
    text="I'm not sure where the statue ... is, but ... Based on my previous location
          at [-60.197, 196.541, -1.097]"

and that coordinate matches ground truth to three decimals. The answer was right; the
field was empty, and eval scores an empty field as unanswered. The prompt already says
to fill position for a "where" question, and the model does not break that rule -- it
escapes by choosing type="text", for which a null position is consistent.

So the type gets settled in its own call and the answer schema pins that type's field.
These tests cover the schema surgery and the fallbacks; whether the model classifies
correctly is a board measurement, not a unit test.

No NPU, no Milvus, no server:

    python remembr/tests/test_answer_type_schema.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.non_agent import (  # noqa: E402
    ANSWER_TYPES,
    NON_NULL_FIELD,
    TYPE_FIELD,
    NonAgent,
)

PERMISSIVE = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "maxLength": 20},
        "text": {"type": "string", "maxLength": 300},
        "binary": {"anyOf": [{"enum": ["yes", "no"]}, {"type": "null"}]},
        "position": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                {"type": "null"},
            ]
        },
        "orientation": {"anyOf": [{"type": "string", "maxLength": 40}, {"type": "null"}]},
        "time": {"anyOf": [{"type": "number"}, {"type": "string", "maxLength": 24}, {"type": "null"}]},
        "duration": {"anyOf": [{"type": "number"}, {"type": "string", "maxLength": 24}, {"type": "null"}]},
    },
    "required": ["type", "text", "binary", "position", "orientation", "time", "duration"],
    "additionalProperties": False,
}


def agent(schema=PERMISSIVE):
    """A NonAgent with just the attributes the schema logic touches."""
    made = NonAgent.__new__(NonAgent)
    made.answer_schema = schema
    return made


def allows_null(field):
    return any(branch.get("type") == "null" for branch in field.get("anyOf", []))


def test_the_settled_types_field_cannot_be_null():
    for answer_type, key in TYPE_FIELD.items():
        schema = agent().schema_for(answer_type)
        assert not allows_null(schema["properties"][key]), (answer_type, schema["properties"][key])
        assert schema["properties"][key] == NON_NULL_FIELD[answer_type], answer_type


def test_the_type_field_is_pinned_so_the_model_cannot_switch_out():
    schema = agent().schema_for("position")
    assert schema["properties"]["type"] == {"const": "position"}, schema["properties"]["type"]


def test_other_fields_stay_nullable():
    schema = agent().schema_for("position")
    for key in ("binary", "time", "duration", "orientation"):
        assert allows_null(schema["properties"][key]), key


def test_the_permissive_schema_is_not_mutated():
    before = repr(PERMISSIVE)
    agent().schema_for("position")
    agent().schema_for("duration")
    assert repr(PERMISSIVE) == before, "schema_for edited the shared schema in place"


def test_text_and_unknown_types_fall_back_to_permissive():
    # 'text' needs no pinning (it is already required and non-nullable), and anything
    # unrecognised must not silently produce a broken schema.
    for answer_type in ("text", None, "banana"):
        assert agent().schema_for(answer_type) is PERMISSIVE, answer_type


def test_no_schema_means_no_pinning():
    # The Ollama path has no grammar; it must keep working untouched.
    assert agent(schema=None).schema_for("position") is None
    assert agent(schema=None).answer_type_for("Where is the bench?") is None


def test_every_declared_type_is_handled():
    assert set(TYPE_FIELD) | {"text"} == set(ANSWER_TYPES), (set(TYPE_FIELD), set(ANSWER_TYPES))
    assert set(NON_NULL_FIELD) == set(TYPE_FIELD)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
