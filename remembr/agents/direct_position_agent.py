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


def split_question(question: str):
    """The question's preamble and the line actually asked, separately.

    Every question arrives with a preamble stating when and where the robot is now
    (`form_question_jsons.py`: "You started moving at ... The current time is ... and
    you are located at [x, y, z]. \\n <asked>"). Same last-line rule as
    `RetrieveAgent.asked_line`, spelled here because this needs both halves, not just
    the asked one.

    Both halves are load-bearing, in different places -- measured on the board, one
    question, greedy, identical context:

        preamble inside the question sentence   [0.097, -4.321, 0.003]   209.72 m off
        + prompt saying it is not the answer    [0.0, 0.0, 0.0]          205.56 m off
        preamble deleted entirely               [-60.197, 196.541, -1.097]  0.00 m
        preamble hoisted to a labelled field    [-60.197, 196.541, -1.097]  0.00 m

    Asked for a coordinate with a coordinate sitting in the question sentence, the
    model answers with that one -- its own current position -- and saying "that is not
    the answer" does not stop it. Deleting the preamble fixes the coordinate but throws
    away what half the dataset needs: "Where is the *nearest* recycling bin?" cannot be
    disambiguated without the current position, and "How long have you been outside?"
    cannot be answered without the current time. So keep it, out of the sentence.
    """
    lines = [line.strip() for line in question.strip().splitlines() if line.strip()]
    if not lines:
        return "", question.strip()
    return "\n".join(lines[:-1]), lines[-1]


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

    def prompt_for(self, question: str, context: str) -> str:
        """The one prompt this agent sends, with the preamble out of the question.

        See `split_question` for the measurements behind the shape. The preamble keeps
        its own labelled block so the current time and position stay available to
        "nearest"/"how long ago" questions without sitting in the sentence that asks
        for a coordinate.
        """
        preamble, asked = split_question(question)
        situation = (
            f"Your current situation: {preamble}\n"
            "(This is where and when you are now. It is not the answer.)\n\n"
            if preamble else ""
        )
        return f"{SYSTEM_PROMPT}\n\n{situation}{context}\n\nQuestion: {asked}"

    def query(self, question: str) -> AgentOutput:
        # The asked line is the search query -- no query-generation call. The preamble
        # is dropped here rather than hoisted: it names the robot's own position and
        # time, which the retriever has no use for and which only dilute the embedding.
        _, asked = split_question(question)
        trace.step("검색어", repr(asked))
        self.memory.search_by_text(asked)
        context = self.memory.memory_to_string(self.memory.get_working_memory())

        # Exactly one LLM call. No retry: greedy decoding would repeat the same
        # bytes, and eval.answer_squad_question already caps attempts at 5.
        reply = self.chain.invoke(self.prompt_for(question, context))
        output = position_output(json.loads(reply.content))
        trace.step("답변", f"position={output.position!r}")
        return output
