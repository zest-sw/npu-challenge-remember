"""Self-check: a response that asks to retrieve AND to reply keeps the retrieval.

The loop in `_generate` returned on the first `__conversational_response` it met
and discarded every real tool call alongside it. Measured on the board
(Llama-3.1-8B, remembr+), 2 of 3 questions asked for `retrieve_from_text` next to
a conversational reply and got no retrieval at all, so the agent answered "I do
not have enough information" from an empty memory.

No NPU, no Milvus, no server:

    python remembr/tests/test_functions_wrapper_tool_calls.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.tools.functions_wrapper import FunctionsWrapper  # noqa: E402

RETRIEVE = {"name": "retrieve_from_text", "description": "search memory", "parameters": {}}
REPLY = {"tool": "__conversational_response", "tool_input": {"response": "I do not have enough information"}}
SEARCH = {"tool": "retrieve_from_text", "tool_input": {"x": "place to sit"}}


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _messages, **_kwargs):
        return FakeReply(json.dumps(self.payload))


def generate(payload):
    wrapper = FunctionsWrapper(FakeLLM(payload))
    result = wrapper._generate([], functions=[dict(RETRIEVE)])
    return result.generations[0].message


def test_retrieval_survives_a_reply_in_the_same_response():
    message = generate([SEARCH, REPLY])
    names = [call["name"] for call in message.tool_calls]
    assert names == ["retrieve_from_text"], names
    assert message.tool_calls[0]["args"] == {"x": "place to sit"}, message.tool_calls


def test_order_does_not_matter():
    message = generate([REPLY, SEARCH])
    names = [call["name"] for call in message.tool_calls]
    assert names == ["retrieve_from_text"], names


def test_a_reply_on_its_own_is_still_the_answer():
    message = generate([REPLY])
    assert message.content == "I do not have enough information", message.content
    assert not message.tool_calls, message.tool_calls


def test_a_lone_tool_call_is_unchanged():
    message = generate([SEARCH])
    names = [call["name"] for call in message.tool_calls]
    assert names == ["retrieve_from_text"], names


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
