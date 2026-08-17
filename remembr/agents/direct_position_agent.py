"""Question -> vector search with the question itself -> which memory answers it.

No query-generation call, no answer-type call, one LLM call per question. The model
answers with a memory's *number*, not a coordinate, and this module reads the coordinate
off that memory. A position-only probe -- every question gets a coordinate, and
non-position questions are knowingly wrong.

No retries. Greedy decoding repeats itself byte-for-byte, so a second try buys
nothing; a parse failure raises and eval's own attempt cap records it.

    python remembr/scripts/eval.py --model direct+tt/Llama-3.1-8B-Instruct ...
"""
import json
import os
import re

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory
from remembr.utils import trace

# Cheap because the answer is one token, and short because the grammar cannot open with
# whitespace the way an object's can.
ANSWER_MAX_TOKENS = 8

# Room for the optional thinking call. Long enough for a sentence about each of the five
# memories, short enough that a rambling reply cannot dominate the turn.
REASON_MAX_TOKENS = 160

# The answer that names no memory. Without it the grammar forces a pick even when no
# retrieved caption names the thing asked about, and the model picks anyway.
ABSTAIN = 0

def memory_index_schema(count: int, abstain: bool = False) -> dict:
    """The whole answer contract: one of the memory numbers on offer.

    An answer is always one of the retrieved memories' own coordinates, never a computed
    one -- ground truth for a position question is where the robot stood when it saw the
    thing, and the caption was written from that same pose. Checked against the data: in
    all 12 position questions some caption in the window sits at the answer to nine
    decimal places. So naming a memory loses no answer this eval can score, and it takes
    away the two ways the coordinate form went wrong -- inventing a coordinate, and
    copying the wrong memory's. Which number maps to which coordinate is this module's
    job, not the model's.

    An enum rather than minimum/maximum: `agent.py` records lm-format-enforcer dropping
    sibling keywords on some forms, and an enum makes anything else ungenerable instead
    of merely out of range. (Both forms were measured working on the board; the enum is
    the one that cannot quietly stop constraining.)
    """
    offered = ([ABSTAIN] if abstain else []) + list(range(1, count + 1))
    return {"type": "integer", "enum": offered}


def what_memories_say(has_distance: bool) -> str:
    """The sentence both calls open with, describing what the memory lines carry.

    One copy, because saying "how far it is from you now" while the lines carry
    coordinates is a lie the model has to work around -- and because the thinking call
    and the choosing call must describe the same lines the same way.
    """
    carries = ("records what you saw and how far it is from you now" if has_distance
               else "records where you were and what you saw there")
    return ("You are a robot looking back over your memory of what you saw. Each numbered "
            f"memory below {carries}.")


def system_prompt(has_distance: bool, abstain: bool = False) -> str:
    """The instruction for the call that answers."""
    way_out = (f" If none of the memories mentions it, answer {ABSTAIN}."
               if abstain else "")
    return (f"{what_memories_say(has_distance)} Pick the memory whose description mentions "
            "the thing the question asks about, and respond with only that memory's "
            f"number.{way_out}")


def reason_prompt(has_distance: bool) -> str:
    """The instruction for the thinking call, which deliberately does not ask for an answer.

    Given the answer contract this call answers instead of reasoning, and the second call
    is left with a bare number to agree with. A separate call is what puts the words
    first: asked for {reason, memory} in one, the server generated `memory` first whatever
    order the schema declared, so the reasoning only justified a choice already made.
    """
    return (f"{what_memories_say(has_distance)}\n"
            "Think about which memories could be the thing the question asks about, and "
            "which cannot. Two or three sentences. Do not give a final answer yet.")


def position_of(doc) -> list[float]:
    """A retrieved memory's coordinate, or a raise if it has no usable one.

    Both readers need the same guard: `zip` in the distance line would quietly drop a
    dimension and print a too-small distance, and a too-small distance is exactly what
    steers the model onto that memory.
    """
    where = [float(value) for value in doc.metadata["position"]]
    if len(where) != 3:
        raise ValueError(f"memory has no [x, y, z]: {where!r}")
    return where


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


def current_position(preamble: str):
    """The robot's own coordinate, out of the preamble, or None if it does not say.

    What `numbered_memories` measures each candidate against. Returns None rather than
    guessing, so a question without a preamble falls back to showing coordinates instead
    of quietly reporting distances from the origin.
    """
    found = re.search(r"located at \[([-\d.,\s]+)\]", preamble)
    if not found:
        return None
    try:
        values = [float(v) for v in found.group(1).split(",")]
    except ValueError:
        # The bracket matched but a field is not a number ("[1.0, 2.0, ]"). Same answer
        # as no bracket at all: say nothing rather than raise out of prompt building.
        return None
    return values if len(values) == 3 else None


def memory_index_output(index, memories, text="") -> AgentOutput:
    """The chosen memory's own coordinate, as the AgentOutput eval scores.

    Validated even though the grammar already restricts the answer: the grammar only
    binds when the tt server is enforcing it, and a wrong number must fail loudly rather
    than silently score against whatever memory it lands on. bool is an int subclass, so
    `True` would otherwise index memory 1.

    `text` is unused by the scoring, so it carries how the answer was reached into
    eval's saved JSON -- otherwise an abstention is indistinguishable from a confident
    pick once the run is over.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"expected a memory number, got: {index!r}")
    if not 1 <= index <= len(memories):
        raise ValueError(f"memory {index} is outside 1..{len(memories)}")

    return AgentOutput(type="position", text=text, binary=None,
                       position=position_of(memories[index - 1]),
                       orientation=None, duration=None, time=None)


class DirectPositionAgent(Agent):
    def __init__(self, llm_type="tt/Llama-3.1-8B-Instruct", num_ctx=8192, temperature=0,
                 abstain=False, reason=False):
        # num_ctx: accepted so eval.py can construct every agent the same way; the
        # tt server owns its own context budget, so it is unused here.
        if not llm_type.startswith("tt/"):
            raise ValueError(
                "DirectPositionAgent needs the tt/ server's grammar-constrained "
                f"decoding; got llm_type={llm_type!r}"
            )
        from langchain_openai import ChatOpenAI  # lazy: optional dep

        self.llm_type = llm_type
        self.abstain = abstain
        self.reason = reason
        # The grammar is bound per call, not here: it enumerates the memory numbers on
        # offer, and how many came back is only known once the search has run.
        self.chain = ChatOpenAI(
            model=llm_type[len("tt/"):],
            base_url=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"),
            api_key="local",
            temperature=temperature,
            max_tokens=ANSWER_MAX_TOKENS,
        )

    def set_memory(self, memory: Memory):
        self.memory = memory

    def numbered_memories(self, memories, here=None) -> str:
        """The retrieved memories, numbered so the answer can name one.

        Each one carries how far it is from where the robot is now, worked out here rather
        than left as a coordinate for the model to work out. Half the dataset's questions
        turn on that distance -- "the *closest* place to sit", "the *nearest* recycling
        bin" -- and comparing 3D coordinates is the one part of this job the model cannot
        do. See `probe_memory_selection.py` for the controlled measurement.

        Falls back to `Memory.memory_to_string` when the question does not say where the
        robot is, since then there is no distance to give.
        """
        if here is None:
            return "\n".join(f"{number}. {self.memory.memory_to_string([doc]).strip()}"
                             for number, doc in enumerate(memories, 1))

        lines = []
        for number, doc in enumerate(memories, 1):
            where = position_of(doc)
            gap = sum((a - b) ** 2 for a, b in zip(where, here)) ** 0.5
            # "The robot saw the following: " is `MilvusMemory.memory_to_string`'s own
            # wording, repeated here because that rendering also carries the coordinate
            # this line exists to replace. Keep the two in step by hand.
            caption = " ".join(doc.page_content.split())
            lines.append(f"{number}. distance_from_you={gap:.1f}m. "
                         f"The robot saw the following: {caption}")
        return "\n".join(lines)

    def think(self, asked: str, memories, here) -> str:
        """One unconstrained call whose words become context for the choosing call.

        With one token to answer in, the model reads the list but falls back to an end of
        it rather than discriminating. This buys it somewhere to do the discriminating.
        """
        prompt = (f"{reason_prompt(here is not None)}\n\n"
                  f"{self.numbered_memories(memories, here)}\n\nQuestion: {asked}")
        thought = self.chain.bind(max_tokens=REASON_MAX_TOKENS).invoke(prompt).content
        trace.step("추론", trace.clip(" ".join(thought.split()), 120))
        return thought

    def prompt_for(self, preamble: str, asked: str, memories, here, thought=None) -> str:
        """The one prompt this agent sends, with the preamble out of the question.

        See `split_question` for the measurements behind the shape. The preamble keeps its
        own labelled block so the current time stays available, and its position is what
        `numbered_memories` measures each candidate against.
        """
        situation = (
            f"Your current situation: {preamble}\n"
            "(This is where and when you are now. It is not the answer.)\n\n"
            if preamble else ""
        )
        thinking = f"Your earlier thinking about these memories:\n{thought}\n\n" if thought else ""
        return (f"{system_prompt(here is not None, self.abstain)}\n\n{situation}"
                f"{self.numbered_memories(memories, here)}\n\n{thinking}Question: {asked}")

    def query(self, question: str) -> AgentOutput:
        # The asked line is the search query -- no query-generation call. The preamble
        # is dropped here rather than hoisted: it names the robot's own position and
        # time, which the retriever has no use for and which only dilute the embedding.
        preamble, asked = split_question(question)
        trace.step("검색어", repr(asked))
        self.memory.search_by_text(asked)
        here = current_position(preamble)
        memories = self.memory.get_working_memory()
        if not memories:
            # Nothing to name, so there is no answer to give. Raising lets eval record an
            # unanswered attempt instead of scoring a guess.
            raise ValueError("retrieval returned no memories; nothing to choose from")

        thought = self.think(asked, memories, here) if self.reason else None

        # One LLM call, or two with --reason. No retry: greedy decoding would repeat the
        # same bytes, and eval.answer_squad_question already caps attempts at 5.
        reply = self.chain.bind(response_format={
            "type": "json_schema",
            "json_schema": {"name": "memory_index",
                            "schema": memory_index_schema(len(memories), self.abstain)},
        }).invoke(self.prompt_for(preamble, asked, memories, here, thought))

        index = json.loads(reply.content)
        if self.abstain and index == ABSTAIN:
            # No memory names the thing asked about, so the search's own first hit stands
            # -- taken from retrieval order, which `ordered` only changes the display of.
            output = memory_index_output(1, memories, text="abstained")
            trace.step("답변", f"기권 -> 검색 1위  position={output.position!r}")
            return output

        output = memory_index_output(index, memories)
        trace.step("답변", f"memory={index}/{len(memories)}  position={output.position!r}")
        return output
