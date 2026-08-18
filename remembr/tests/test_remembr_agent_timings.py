"""Offline checks for remembr+ latency collection.

The interesting part is not arithmetic but plumbing: langchain_openai drops unknown
top-level response fields, so these tests pin that the httpx hook still sees the tt
server's `timings` while ChatOpenAI parses the reply as usual. A throwaway localhost
server stands in for the NPU -- no board involved.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from remembr.agents.remembr_agent import _latency_summary, _timing_client

TIMINGS = {
    "prefill_s": 1.5,
    "device_decode_s": 2.5,
    "constraint_decoding_s": 0.25,
    "generation_s": 2.75,
    "generated_tokens": 7,
}


@pytest.fixture
def fake_server():
    body = json.dumps({
        "id": "x", "object": "chat.completion", "model": "fake",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "timings": TIMINGS,
    }).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_hook_captures_server_timings_langchain_drops(fake_server):
    from langchain_openai import ChatOpenAI

    calls = []
    llm = ChatOpenAI(model="fake", base_url=fake_server, api_key="local",
                     http_client=_timing_client(calls))

    reply = llm.invoke("first")
    llm.invoke("second")

    # The reply itself must still parse: reading the body in the hook may not consume it.
    assert reply.content == "hi"
    # Which is the whole reason the hook exists.
    assert "timings" not in reply.response_metadata
    assert calls == [TIMINGS, TIMINGS]


def test_hook_ignores_errors_and_non_json(fake_server):
    calls = []
    client = _timing_client(calls)
    hook = client.event_hooks["response"][0]

    hook(httpx.Response(500, json={"timings": TIMINGS}))
    hook(httpx.Response(200, text="not json"))
    hook(httpx.Response(200, json={"choices": []}))

    assert calls == []


def test_summary_sums_every_call_the_planner_made():
    summary = _latency_summary([TIMINGS, TIMINGS], pipeline_total_s=12.0)

    assert summary["llm_calls"] == 2
    assert summary["pipeline_total_s"] == 12.0
    assert summary["prefill_s"] == 3.0
    assert summary["generated_tokens"] == 14
    assert summary["per_call"] == [TIMINGS, TIMINGS]


def test_summary_omits_server_split_when_no_timings_came_back():
    summary = _latency_summary([], pipeline_total_s=4.0)

    # Zeros would read as "the server spent no time", not "this backend is not tt/".
    assert summary == {"llm_calls": 0, "pipeline_total_s": 4.0}
