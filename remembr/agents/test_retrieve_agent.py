"""Self-check for RetrieveAgent's query -> retrieve -> answer wiring.

The thing worth protecting is the order: the search has to happen before the
answer stage reads working memory, and it has to search for the phrase the model
gave rather than the raw question (the question's preamble states the current time
and position, and sending it pulled the model toward searching for those).

No NPU, no Milvus, no server:

    python remembr/agents/test_retrieve_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.retrieve_agent import RetrieveAgent  # noqa: E402

QUESTION = (
    "The current time is 08/05/2026 09:15:00.\n"
    "You are located at [1.0, 2.0, 0.0].\n"
    "Where is the closest place to sit?"
)

ANSWER_JSON = (
    '{"type": "position", "text": "A bench.", "binary": null, '
    '"position": [3.0, 4.0, 0.0], "orientation": null, "time": null, "duration": null}'
)


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeReply(self.content)


class FakeMemory:
    """Records searches and only reveals memories that were searched for."""

    def __init__(self):
        self.searched = []
        self.working = []

    def search_by_text(self, query):
        self.searched.append(query)
        self.working.append(f"caption for {query}")

    def get_working_memory(self):
        return list(self.working)

    def memory_to_string(self, memory_list):
        return "\n".join(memory_list)


def build(query_reply, answer_reply=ANSWER_JSON):
    """A RetrieveAgent with both LLM calls faked, skipping __init__'s network work."""
    agent = RetrieveAgent.__new__(RetrieveAgent)
    agent.prompt = "Context: {context}\nQuestion: {question}"
    agent.query_prompt = "turn the question into a search phrase"
    agent.query_chain = FakeChain(query_reply)
    agent.chain = FakeChain(answer_reply)
    agent.memory = FakeMemory()
    # No grammar, so NonAgent.query skips answer-type pinning. These tests are about
    # the query -> retrieve -> answer order, not the answer schema.
    agent.answer_schema = None
    return agent


def test_searches_for_the_generated_phrase_not_the_question():
    agent = build("bench seating place to sit")
    agent.query("Where is the closest place to sit?")
    assert agent.memory.searched == ["bench seating place to sit"], agent.memory.searched


def test_only_the_asked_line_is_sent_to_the_query_model():
    agent = build("bench")
    agent.query(QUESTION)
    sent = agent.query_chain.prompts[0]
    asked = sent[-1].content
    assert asked == "Where is the closest place to sit?", asked
    assert "09:15:00" not in asked, "the time preamble leaked into the search query prompt"
    assert "[1.0, 2.0, 0.0]" not in asked, "the position preamble leaked in"


def test_retrieval_happens_before_the_answer_stage_reads_memory():
    agent = build("bench")
    output = agent.query(QUESTION)
    # The fake memory is empty until searched, so the answer prompt containing the
    # retrieved caption is proof the search ran first.
    answer_prompt = agent.chain.prompts[0]
    assert "caption for bench" in answer_prompt, answer_prompt
    assert output.position == [3.0, 4.0, 0.0], output


def test_packaging_around_the_phrase_is_stripped():
    for reply, expected in (
        ('"recycling bin"', "recycling bin"),
        ("recycling bin.", "recycling bin"),
        ("  'yellow police call pole'  ", "yellow police call pole"),
    ):
        agent = build(reply)
        agent.query(QUESTION)
        assert agent.memory.searched == [expected], (reply, agent.memory.searched)


def test_empty_query_does_not_search_and_still_answers():
    agent = build("   ")
    output = agent.query(QUESTION)
    assert agent.memory.searched == [], agent.memory.searched
    # An honest answer from no context beats crashing the whole eval run.
    assert output.position == [3.0, 4.0, 0.0], output


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
