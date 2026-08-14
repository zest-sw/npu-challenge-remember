"""Question -> vector search with the question itself -> a bare [x, y, z].

No query-generation call, no answer-type call, no JSON object: the grammar
allows only a three-number array, so every generated token is answer, none is
punctuation. A position-only probe -- every question gets a coordinate, and
non-position questions are knowingly wrong.

No retries. Greedy decoding repeats itself byte-for-byte, so a second try buys
nothing; a parse failure raises and eval's own attempt cap records it.

    python remembr/scripts/eval.py --model direct+tt/Llama-3.1-8B-Instruct ...
"""
import json
import os

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory
from remembr.utils import trace

# The whole answer contract. Number grammar is finite, so no maxLength is
# needed, and with no string branch the anyOf/null rules in agent.py do not
# apply here.
POSITION_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

SYSTEM_PROMPT = (
    "You are a robot looking back over your memory of what you saw. "
    "From the memory context, answer where the thing the question asks about "
    "is. Respond with only a JSON array of three numbers: [x, y, z]."
)


def position_output(parsed) -> AgentOutput:
    """A bare [x, y, z] as the AgentOutput eval scores.

    bool is an int subclass, so `[True, 1, 2]` must be rejected explicitly.
    """
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in parsed)
    ):
        raise ValueError(f"expected [x, y, z], got: {parsed!r}")
    return AgentOutput(type="position", text="", binary=None,
                       position=[float(n) for n in parsed],
                       orientation=None, duration=None, time=None)


class DirectPositionAgent(Agent):
    def __init__(self, llm_type="tt/Llama-3.1-8B-Instruct", num_ctx=8192, temperature=0):
        # num_ctx: accepted so eval.py can construct every agent the same way; the
        # tt server owns its own context budget, so it is unused here.
        if not llm_type.startswith("tt/"):
            raise ValueError(
                "DirectPositionAgent needs the tt/ server's grammar-constrained "
                f"decoding; got llm_type={llm_type!r}"
            )
        from langchain_openai import ChatOpenAI  # lazy: optional dep

        self.llm_type = llm_type
        self.chain = ChatOpenAI(
            model=llm_type[len("tt/"):],
            base_url=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"),
            api_key="local",
            temperature=temperature,
            # Three numbers and two brackets; the grammar cannot ramble past that.
            max_tokens=48,
            model_kwargs={"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "position", "schema": POSITION_SCHEMA},
            }},
        )

    def set_memory(self, memory: Memory):
        self.memory = memory

    def query(self, question: str) -> AgentOutput:
        # The question itself is the search query -- no trimming, no query call.
        trace.step("검색어", repr(question))
        self.memory.search_by_text(question)
        context = self.memory.memory_to_string(self.memory.get_working_memory())

        # Exactly one LLM call. No retry: greedy decoding would repeat the same
        # bytes, and eval.answer_squad_question already caps attempts at 5.
        reply = self.chain.invoke(f"{SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {question}")
        output = position_output(json.loads(reply.content))
        trace.step("답변", f"position={output.position!r}")
        return output
