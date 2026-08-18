"""Self-check for NonAgent's position-only model response contract.

No NPU, Milvus, or server:

    python remembr/agents/test_position_only_schema.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.non_agent import POSITION_ONLY_SCHEMA  # noqa: E402


def test_only_position_can_be_generated():
    assert set(POSITION_ONLY_SCHEMA["properties"]) == {"position"}
    assert POSITION_ONLY_SCHEMA["required"] == ["position"]
    assert POSITION_ONLY_SCHEMA["additionalProperties"] is False


def test_position_is_exactly_three_numbers():
    field = POSITION_ONLY_SCHEMA["properties"]["position"]
    assert field == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }


def test_prompt_requests_the_same_shape():
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "non_agent_system_prompt.txt"
    )
    prompt = open(prompt_path).read()
    assert '{{"position": [1.0, 2.0, 0.0]}}' in prompt
    for stale_field in ('"text"', '"binary"', '"orientation"', '"time"', '"duration"'):
        assert stale_field not in prompt, stale_field


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
