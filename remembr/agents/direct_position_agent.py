"""Question -> vector search -> which memory answers it.

The default path searches with the question itself and makes one LLM selection call.
The optional multi-query path asks the LLM for one copied target noun phrase and one alias,
then fuses the original-question ranking with one compact keyword ranking. In both paths
the selector answers with a memory's *number*, not a coordinate, and this module reads
the coordinate off that memory. A position-only probe -- every question gets a
coordinate, and non-position questions are knowingly wrong.

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
QUERY_MAX_TOKENS = 64
TARGET_MAX_LENGTH = 60
ALIAS_MAX_LENGTH = 40
ALIAS_LIMIT = 1

QUERY_EXPANSION_PROMPT = (
    "Copy the noun phrase naming the requested object exactly and contiguously from the "
    "question. Keep essential visual modifiers and attached identifying features, such as "
    "yellow police call pole, red fire hydrant, or sculpture with hanging boats. Exclude "
    "location or relation clauses and proper-place descriptions, such as in front of the "
    "Bill and Melinda Gates Computer Science building; for that question the target is "
    "statue. Keep target_phrase under 60 characters. Return exactly one short alias for "
    "the same complete object category. The alias must not be a component, feature, "
    "superclass, surrounding object, or guessed interpretation. Do not invent scene facts."
)


def query_expansion_schema() -> dict:
    """Bound the copied target and one alias so constrained decoding cannot run away."""
    return {
        "type": "object",
        "properties": {
            "target_phrase": {"type": "string", "maxLength": TARGET_MAX_LENGTH},
            "aliases": {
                "type": "array",
                "items": {"type": "string", "maxLength": ALIAS_MAX_LENGTH},
                "maxItems": ALIAS_LIMIT,
            },
        },
        "required": ["target_phrase", "aliases"],
        "additionalProperties": False,
    }


def combined_queries(original: str, expansion: dict) -> list[str]:
    """Return the original question plus one target-and-alias keyword query."""
    target = expansion.get("target_phrase", "")
    aliases = expansion.get("aliases", [])
    if not isinstance(target, str) or not isinstance(aliases, list):
        raise ValueError(f"generated target or aliases have wrong types: {expansion!r}")

    original = " ".join(original.split()).strip()
    target = " ".join(target.split()).strip('"\' .')
    if not target:
        return [original]
    target_pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(target).replace(r"\ ", r"\s+")
        + r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    if not target_pattern.search(original):
        raise ValueError(f"target phrase was not copied from the question: {target!r}")

    alias = " ".join(aliases[0].split()).strip('"\' .,;:?') if aliases else ""
    if alias and alias.casefold() != target.casefold() and len(alias.split()) <= 4 \
            and "," not in alias and ";" not in alias:
        return [original, f"{target}, {alias}"]
    return [original, target]


def memory_index_schema(count: int) -> dict:
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
    return {"type": "integer", "enum": list(range(1, count + 1))}


def system_prompt(has_distance: bool) -> str:
    """The instruction, matched to what the memory lines actually carry.

    One string with one varying clause rather than two, so the two cannot drift apart:
    saying "how far it is from you now" when the lines carry coordinates would be a lie
    the model has to work around.
    """
    carries = ("records what you saw and how far it is from you now" if has_distance
               else "records where you were and what you saw there")
    return ("You are a robot looking back over your memory of what you saw. Each numbered "
            f"memory below {carries}. Pick the memory whose description mentions the thing "
            "the question asks about, and respond with only that memory's number.")


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


def memory_index_output(index, memories) -> AgentOutput:
    """The chosen memory's own coordinate, as the AgentOutput eval scores.

    Validated even though the grammar already restricts the answer: the grammar only
    binds when the tt server is enforcing it, and a wrong number must fail loudly rather
    than silently score against whatever memory it lands on. bool is an int subclass, so
    `True` would otherwise index memory 1.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(f"expected a memory number, got: {index!r}")
    if not 1 <= index <= len(memories):
        raise ValueError(f"memory {index} is outside 1..{len(memories)}")

    position = memories[index - 1].metadata["position"]
    if len(position) != 3:
        raise ValueError(f"memory {index} has no [x, y, z]: {position!r}")
    return AgentOutput(type="position", text="", binary=None,
                       position=[float(value) for value in position],
                       orientation=None, duration=None, time=None)


class DirectPositionAgent(Agent):
    def __init__(self, llm_type="tt/Llama-3.1-8B-Instruct", num_ctx=8192, temperature=0,
                 text_query_mode="direct"):
        # num_ctx: accepted so eval.py can construct every agent the same way; the
        # tt server owns its own context budget, so it is unused here.
        if not llm_type.startswith("tt/"):
            raise ValueError(
                "DirectPositionAgent needs the tt/ server's grammar-constrained "
                f"decoding; got llm_type={llm_type!r}"
            )
        from langchain_openai import ChatOpenAI  # lazy: optional dep

        self.llm_type = llm_type
        if text_query_mode not in {"direct", "multi"}:
            raise ValueError(f"unknown text_query_mode: {text_query_mode!r}")
        self.text_query_mode = text_query_mode
        self.query_cache = {}
        # The grammar is bound per call, not here: it enumerates the memory numbers on
        # offer, and how many came back is only known once the search has run.
        self.chain = ChatOpenAI(
            model=llm_type[len("tt/"):],
            base_url=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"),
            api_key="local",
            temperature=temperature,
            max_tokens=ANSWER_MAX_TOKENS,
        )
        self.query_chain = None
        if text_query_mode == "multi":
            self.query_chain = ChatOpenAI(
                model=llm_type[len("tt/"):],
                base_url=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"),
                api_key="local",
                temperature=temperature,
                max_tokens=QUERY_MAX_TOKENS,
            )

    def set_memory(self, memory: Memory):
        self.memory = memory

    def numbered_memories(self, memories, here=None) -> str:
        """The retrieved memories, numbered so the answer can name one.

        Each one carries how far it is from where the robot is now, worked out here rather
        than left as a coordinate for the model to work out. Half the dataset's questions
        turn on that distance -- "the *closest* place to sit", "the *nearest* recycling
        bin" -- and comparing 3D coordinates is the one part of this job the model cannot
        do. Measured on the real geometry, four benches at 97.1 / 104.5 / 122.2 / 123.0 m
        from the robot, the target moved through every slot:

            coordinates, model compares   1/5   -- answered "5" whatever the arrangement
            distances given as numbers    4/5

        It is not a reading problem. Given the same lists with an unambiguous target the
        model scores 64/64: any slot, up to 20 candidates, same-scene distractors, the
        target buried in filler sentences, and paraphrased so the question's noun never
        appears. It fails only when it has to do the arithmetic, and when it fails it
        answers with the last memory. Over the twelve position questions, `--temperature 0`,
        this swap moved the run from 42.0 m to 21.6 m.

        Falls back to `Memory.memory_to_string` when the question does not say where the
        robot is, since then there is no distance to give.
        """
        if here is None:
            return "\n".join(f"{number}. {self.memory.memory_to_string([doc]).strip()}"
                             for number, doc in enumerate(memories, 1))

        lines = []
        for number, doc in enumerate(memories, 1):
            where = [float(value) for value in doc.metadata["position"]]
            if len(where) != 3:
                # Same invariant `memory_index_output` enforces. Without this, `zip`
                # would quietly drop a dimension and print a too-small distance, which
                # is worse than failing: it steers the model onto that memory.
                raise ValueError(f"memory {number} has no [x, y, z]: {where!r}")
            gap = sum((a - b) ** 2 for a, b in zip(where, here)) ** 0.5
            # "The robot saw the following: " is `MilvusMemory.memory_to_string`'s own
            # wording, repeated here because that rendering also carries the coordinate
            # this line exists to replace. Keep the two in step by hand.
            caption = " ".join(doc.page_content.split())
            lines.append(f"{number}. distance_from_you={gap:.1f}m. "
                         f"The robot saw the following: {caption}")
        return "\n".join(lines)

    def prompt_for(self, question: str, memories) -> str:
        """The one prompt this agent sends, with the preamble out of the question.

        See `split_question` for the measurements behind the shape. The preamble keeps its
        own labelled block so the current time stays available, and its position is what
        `numbered_memories` measures each candidate against.
        """
        preamble, asked = split_question(question)
        situation = (
            f"Your current situation: {preamble}\n"
            "(This is where and when you are now. It is not the answer.)\n\n"
            if preamble else ""
        )
        here = current_position(preamble)
        return (f"{system_prompt(here is not None)}\n\n{situation}"
                f"{self.numbered_memories(memories, here)}\n\nQuestion: {asked}")

    def search_queries_for(self, asked: str) -> list[str]:
        """Return original + one target keyword bundle, falling back on errors.

        The original preserves proper names, colors, state, distance, and relations. The
        second query contains only the copied target and one alias, rather than rewrites of
        the whole question. Repeated asked lines reuse their expansion because
        several eval questions differ only in the time window and current pose.
        """
        if self.text_query_mode == "direct":
            return [asked]
        if asked in self.query_cache:
            return self.query_cache[asked]
        try:
            reply = self.query_chain.bind(response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "visual_target_alias",
                    "schema": query_expansion_schema(),
                },
            }).invoke(f"{QUERY_EXPANSION_PROMPT}\n\nQuestion: {asked}")
            queries = combined_queries(asked, json.loads(reply.content))
            if len(queries) == 1:
                trace.warn("multi-query가 원문과 다른 검색어를 만들지 못해 원문만 사용한다")
            self.query_cache[asked] = queries
            return queries
        except Exception as error:  # noqa: BLE001 - retrieval has a safe direct fallback
            trace.warn(f"multi-query 생성 실패, 원문 검색으로 대체: {error!r}")
            return [asked]

    def query(self, question: str) -> AgentOutput:
        # The asked line is always one search query and is also the input to optional
        # expansion. The preamble is dropped: it names the robot's own position and time,
        # which the retriever has no use for and which only dilute the embedding.
        _, asked = split_question(question)
        queries = self.search_queries_for(asked)
        trace.step("검색어", repr(queries if self.text_query_mode == "multi" else asked))
        if self.text_query_mode == "multi":
            self.memory.search_by_text_multi(queries)
        else:
            self.memory.search_by_text(asked)
        memories = self.memory.get_working_memory()
        if not memories:
            # Nothing to name, so there is no answer to give. Raising lets eval record an
            # unanswered attempt instead of scoring a guess.
            raise ValueError("retrieval returned no memories; nothing to choose from")

        # Exactly one LLM call. No retry: greedy decoding would repeat the same
        # bytes, and eval.answer_squad_question already caps attempts at 5.
        reply = self.chain.bind(response_format={
            "type": "json_schema",
            "json_schema": {"name": "memory_index",
                            "schema": memory_index_schema(len(memories))},
        }).invoke(self.prompt_for(question, memories))

        index = json.loads(reply.content)
        output = memory_index_output(index, memories)
        trace.step("답변", f"memory={index}/{len(memories)}  position={output.position!r}")
        return output
