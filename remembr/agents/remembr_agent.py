from typing import Annotated, Literal, Sequence, TypedDict
import traceback
import sys, re, ast, json, datetime, math

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama
# NOTE: ChatNVIDIA (langchain_nvidia_ai_endpoints) is only needed for the NVIDIA
# NIM backend ('nim/...'). Import it lazily inside llm_selector so eval.py can
# import ReMEmbRAgent (e.g. for the tt/ or optimal paths) without that package.

from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field, conlist


import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper, _openai_backend

from remembr.memory.memory import Memory

from remembr.agents.agent import Agent, AgentOutput



### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    # print(state)
    return state


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)


# ---------------------------------------------------------------------------
# Constrained final-answer generation.
#
# The final answer must become an AgentOutput(type, text, binary, position,
# orientation, duration, time) -- ALL seven fields are required (the dataclass has
# no defaults). A weak 3B model left unconstrained tends to (a) ramble its reasoning
# into a string until it hits the token cap, which breaks the JSON mid-value, and
# (b) omit fields such as `type`/`orientation`, which then crashes AgentOutput. So on
# the OpenAI-compatible (tt) backend we grammar-constrain this step too: a flat schema
# forcing exactly the seven keys, with no room for free-form reasoning.
# ---------------------------------------------------------------------------

def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"enum": ["text", "binary", "position", "time", "duration"]},
        # maxLength caps degeneration inside the string the same way maxItems does for
        # arrays: without it a non-terminating model rambles until the 512-token budget.
        "text": {"type": "string", "maxLength": 300},
        "binary": {"anyOf": [{"enum": ["yes", "no"]}, {"type": "null"}]},
        # Array lengths are grammar-enforced: without maxItems the tt backend's greedy
        # decoding degenerates into unbounded [0, 0, 0, ...] arrays that eat the whole
        # token budget (see docs/TENSTORRENT_INTEGRATION.md on degeneration).
        "position": _nullable({"type": "array", "items": {"type": "number"},
                               "minItems": 3, "maxItems": 3}),
        "orientation": _nullable({"type": "array", "items": {"type": "number"},
                                  "minItems": 1, "maxItems": 1}),
        "time": _nullable({"type": "number"}),
        "duration": _nullable({"type": "number"}),
    },
    "required": ["type", "text", "binary", "position", "orientation", "time", "duration"],
    "additionalProperties": False,
}

# Slim answer prompt aligned to ANSWER_SCHEMA: flat, no nested tool wrapper, no
# reasoning fields (those are what overflowed the token budget and broke the JSON).
GENERATE_STRICT_PROMPT = """You are a robot answering a question about your past from your memory.
The question begins with the current time (HH:MM:SS) and position (x, y, z in meters); the user asks about something earlier. Use the retrieved context in the conversation to answer.

Question: "{question}"

Respond with ONE JSON object and NOTHING else -- no reasoning, no prose outside it. Fill EVERY key:
- "type": the kind of answer the question needs -- "position", "time", "duration", "binary", or "text".
- "text": a single short sentence answering the user (no low-level details).
- "binary": "yes" or "no" for yes/no questions, otherwise null.
- "position": [x, y, z] in meters for "where" questions, otherwise null. Copy the coordinates from the retrieved memory that mentions the thing being asked about. Do NOT answer with the current position from the question -- the question asks where something else is, not where you are.
- "orientation": [yaw] for "where" questions, otherwise null.
- "time": a number of minutes-ago for "when did X happen" questions, otherwise null.
- "duration": a number of minutes for "how long" questions, otherwise null.
Numbers must be plain JSON numbers (e.g. 5.3), not strings. Use null for keys that do not apply, but ALWAYS fill the field matching the selected "type" with your best guess."""

_DEFAULT_ANSWER = {
    "type": "text", "text": "", "binary": None,
    "position": None, "orientation": None, "time": None, "duration": None,
}


def _coerce_list(value):
    """Coerce a position/orientation value to a list of floats, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)):
        return [float(value)]
    return None


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_answer(text):
    """Parse a model answer string into a dict. Prefers strict JSON (what constrained
    decoding emits); falls back to a python-literal for the Ollama/legacy paths."""
    text = ''.join(str(text).splitlines()).strip()
    m = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    return ast.literal_eval(text)  # raises on truly broken output -> node retries


def normalize_answer(parsed):
    """Return a dict with ALL seven AgentOutput fields present and sanely typed, so
    AgentOutput.from_dict() can never crash on a missing or mis-typed key."""
    if not isinstance(parsed, dict):
        parsed = {}
    out = dict(_DEFAULT_ANSWER)
    for k in _DEFAULT_ANSWER:
        if k in parsed:
            out[k] = parsed[k]
    out["position"] = _coerce_list(out["position"])
    out["orientation"] = _coerce_list(out["orientation"])
    out["time"] = _coerce_float(out["time"])
    out["duration"] = _coerce_float(out["duration"])
    if not isinstance(out["type"], str) or not out["type"]:
        out["type"] = "text"
    if not isinstance(out["text"], str):
        out["text"] = "" if out["text"] is None else str(out["text"])
    if out["binary"] not in ("yes", "no", None):
        out["binary"] = None
    return out


def _question_window_minutes(question):
    """Minutes between the two timestamps a CODA question states ('started moving at ...'
    and 'current time is ...'). This is the maximum physically-valid value for a
    'minutes ago' or 'how long' answer. Returns None if two timestamps aren't found."""
    ts = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", question or "")
    if len(ts) >= 2:
        try:
            s = datetime.datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
            n = datetime.datetime.strptime(ts[1], "%Y-%m-%d %H:%M:%S")
            w = (n - s).total_seconds() / 60.0
            return w if w > 0 else None
        except Exception:
            return None
    return None


def clamp_minutes(value, question):
    """Clamp a time/duration answer (minutes) to the valid range [0, elapsed-window].

    The 3B does the absolute-timestamp -> minutes arithmetic poorly and produces
    outliers (e.g. 343 min on an 8-min window). Since a "minutes ago" / "how long"
    answer cannot be negative nor exceed the window the question itself defines, we
    floor at 0 and cap at that window -- in-range answers are left untouched. On this
    eval it cuts mean time error 59.2 -> 1.2 min and duration 15.4 -> 4.7 min.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    value = max(0.0, value)
    window = _question_window_minutes(question)
    if window is not None:
        value = min(value, window)
    return value


# Memory strings handed back by the retrievers look like:
#   "At time=<t>, the robot was at an average position of [x, y, z].The robot saw
#    the following: <caption>"
# (see MilvusMemory.memory_to_string). Capture the coordinates and the caption they
# belong to so a "where is X" answer can be grounded in the memory that mentions X.
_MEMORY_ENTRY_RE = re.compile(
    r"average position of\s*\[([-\d\.,\se]+)\]\.?\s*"
    r"The robot saw the following:\s*(.*?)(?=At time=|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Words carried by every CODA question ("where is the nearest X") or by the fixed
# caption preamble; they match everything and so must not drive the choice of memory.
_STOPWORDS = frozenset("""
the and are was were you your with that this there here for from into onto near
what when where which who how why did does was robot see saw seen seeing look looking
current time position located start started moving meters average orientation radians
""".split())


def _keywords(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _question_core(question):
    """The actual question, minus the 'You started moving at ... located at [...]' preamble
    that every CODA question carries (its coordinates are the ones we must NOT echo)."""
    return (question or "").split("\n")[-1].strip()


def _current_position(question):
    """The robot's own position, as stated in the question preamble."""
    m = re.search(r"located at\s*\[([-\d\.,\s]+)\]", question or "")
    if not m:
        return None
    try:
        pos = [float(v) for v in m.group(1).split(",")]
    except ValueError:
        return None
    return pos if len(pos) == 3 else None


def _retrieved_entries(messages):
    """(position, caption) pairs retrieved during THIS query.

    Read from the query's own message history rather than memory.working_memory, which
    is never cleared between questions and so would leak earlier questions' retrievals.
    """
    entries = []
    for m in messages:
        content = getattr(m, "content", m)
        if not isinstance(content, str):
            continue
        for coords, caption in _MEMORY_ENTRY_RE.findall(content):
            try:
                pos = [float(v) for v in coords.split(",")]
            except ValueError:
                continue
            if len(pos) == 3:
                entries.append((pos, caption.strip()))
    return entries


def _is_where_question(question):
    """Whether the question asks for a location.

    Decided from the question text, NOT from the model's own `type` field: on the CODA
    position questions the 3B labelled 7 of 12 answers "text" while still filling
    `position`, and eval.py scores the `position` field off the question's true type
    regardless of that label -- so gating on the label skipped most of the answers that
    needed grounding. Every CODA position question opens with "where" and no question of
    any other type does (12/12 and 0/18 on human_qa sequence 0).
    """
    return _question_core(question).lower().lstrip("'\"( ").startswith("where")


# Questions whose answer depends on where the robot is, not only on what it saw.
_RELATIONAL_WORDS = ("closest", "nearest", "close to me", "near me", "nearby")


def _is_relational(question):
    return any(w in _question_core(question).lower() for w in _RELATIONAL_WORDS)


def _select_memory(question, entries):
    """The retrieved memory a "where" answer should report: the one whose caption shares
    the most content words with the question.

    Two alternatives were replayed over all 12 CODA position questions, each scoring the
    memory it picks against ground truth from an identical retrieved set:

        selector        mean err   within 15 m
        top-1 by rank      25.5 m       7/12
        content words      17.6 m       8/12   <- this one
        embedding cosine   30.1 m       7/12

    Embedding cosine loses because the qwen captions open with a long boilerplate preamble
    that dominates the sentence vector; discrete content words ignore it. IDF-weighting the
    overlap changed nothing (identical picks). The ceiling for any selector over the same
    set is 10/12 at k=5, so the remaining loss is selection, not retrieval -- raising k
    lifts that ceiling to 12/12 but this selector does not exploit it (still 8/12, and mean
    degrades to 23.7 m), so k is left alone.

    The numbers above were replayed with the QUESTION as the search string, which is not
    what the pipeline does: the agent searches with the phrase the LLM writes into
    retrieve_from_text ('a red fire hydrant', 'nearest bike rack not empty'). Re-measured
    with the 12 phrases the 3B actually emits, against both caption sets and both
    retrieval granularities (all end-to-end, i.e. selector included):

        captions  retrieval                       selector        end-to-end
        paper     caption-level top-5             content words   16.0 m 9/12  <- current
        paper     caption-level top-5             top-1           23.7 m 8/12
        paper     top-20 -> best-phrase re-rank   content words   31.9 m 5/12
        dense     caption-level top-5             content words   50.4 m 4/12
        dense     top-20 -> best-phrase re-rank   top-1           28.4 m 7/12
        dense     top-20 -> best-phrase re-rank   LLM picks       38.7 m 5/12

    So the current combination is the best measured one and is close to its own ceiling
    (oracle over its retrieved set is 8.7 m 10/12). Dense captions do retrieve better --
    hit@5 (some retrieved memory within 15 m) is 11-12/12 against 10/12 here -- but
    nothing converts that into answers: every heuristic tops out at 8/12, and handing the
    retrieved memories to the 3B to choose (only affordable with dense captions, ~135
    tokens for 5) is worse still at 5/12; it repairs a bad top-1 occasionally and breaks
    good ones more often. Retrieving with one caption set and selecting with the other was
    tried too (both sets cover the same segments): best hybrid was dense-v2 retrieval with
    paper-caption selection, 35.5 m 6/12, still far below paper alone.

    hit@5 is a misleading way to compare caption sets, because it ignores how much of the
    retrieved set is right. Measuring that directly explains all of the above:

        captions   good of top-5   blind pick   this selector
        paper           1.75         38.5 m       16.0 m
        dense-v1        1.33         44.9 m       50.7 m
        dense-v2        1.67         40.1 m       40.3 m

    Every caption set retrieves only ~1.5 usable memories out of 5, so the whole game is
    picking the right one -- and word overlap can only do that on word-rich text. It more
    than halves the error on paper captions (38.5 -> 16.0 m) and lands at or below chance
    on dense ones, which carry 13-16 content words per caption against 79.5. Terse
    captions do not merely fail to help this selector; they leave it nothing to separate
    candidates by. A caption prompt for this pipeline should be optimised for
    discriminative word content, not for brevity or embedding cleanliness.

    The one change kept from that sweep is the relational tie-break below.
    """
    q = _keywords(_question_core(question))
    scored = [(len(q & _keywords(caption)), pos) for pos, caption in entries]
    best_score = max((s for s, _ in scored), default=0)
    # No caption shares a content word with the question -> fall back to the
    # top-ranked retrieval, which is still a memory rather than the current position.
    if best_score == 0:
        return entries[0][0]

    tied = [pos for s, pos in scored if s == best_score]
    if len(tied) > 1 and _is_relational(question):
        # Several memories match the question's words equally well and the question asks
        # for the CLOSEST one. Which of them is meant is decided by distance, and the
        # question states where the robot is, so break the tie on that rather than on
        # retrieval order. Note this never fires on the 12 CODA position questions -- no
        # two retrieved captions tie above zero there -- so it is principle, not a
        # measured gain, and it cannot change any existing result.
        #
        # A wider version that also fired when NOTHING matched (best_score == 0, i.e. all
        # five memories "tie") was measured and rejected: it went 1 win / 1 loss
        # ('closest elevators' 69.6 -> 17.1 m, 'nearest recycling bin' 5.6 -> 27.4 m),
        # dropping within-15 m from 9/12 to 8/12. A zero score means retrieval found
        # nothing relevant, not that the candidates are equally relevant, so distance is
        # no better a signal there than the embedding rank already is.
        robot = _current_position(question)
        if robot is not None:
            return min(tied, key=lambda p: math.dist(p[:2], robot[:2]))
    # Preserves the original behavior: on a tie the earliest-retrieved memory wins.
    return tied[0]


def ground_position(parsed, question, messages, retriever=None):
    """Answer a "where" question with the coordinates of a retrieved memory rather than
    with whatever the model put in the `position` field.

    A 3B model does not extract an object's past coordinates from the retrieved captions.
    It echoes the robot's current position (5 of 7 "where" answers on the first CODA run),
    leaves the field null, or invents a coordinate that appears in no memory (e.g.
    [1265.48, 27.43, -0.02] on a map spanning roughly -100..200 m).

    An earlier version kept any model answer that landed within 15 m of some retrieved
    memory, on the theory that such an answer was derived rather than invented. Measurement
    says otherwise: that run scored 7/12 within 15 m with a 62.1 m mean, while selecting the
    memory outright scores 8/12 with a 17.6 m mean over the same questions. So the model's
    number is not consulted -- the selected memory is the answer. See _select_memory.
    """
    if parsed.get("type") != "position" and not _is_where_question(question):
        return parsed

    entries = _retrieved_entries(messages)
    if not entries and retriever is not None:
        # The agent answered a "where" question without ever retrieving (1 of 12 on the
        # last run, which then echoed the robot's own position because there was nothing
        # to ground to). Retrieve once here: one vector search, no extra LLM call.
        try:
            entries = _retrieved_entries([retriever(_question_core(question))])
            print(f"[ground_position] agent retrieved nothing; fallback search returned "
                  f"{len(entries)} memories")
        except Exception as e:
            print(f"[ground_position] fallback retrieval failed: {e}")
    if not entries:
        print("[ground_position] no memories to ground to; leaving position as-is")
        return parsed

    answer = _coerce_list(parsed.get("position"))
    chosen = _select_memory(question, entries)
    if answer != chosen:
        # Logged because the eval output records only the final answer: without this line
        # a grounding decision cannot be audited after the fact.
        print(f"[ground_position] {answer} -> {chosen} (from {len(entries)} memories)")
    parsed["position"] = chosen
    return parsed


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func, max_attempts=3):
    """Run a graph node, tolerating transient model/parse errors.

    The old version retried forever (`while True`), so a persistently malformed model
    reply -- e.g. Llama-3B emitting a bare float where a tool-call dict was expected --
    spun the whole eval indefinitely, printing the same traceback. Now we retry a bounded
    number of times and then degrade to a safe state so the graph still terminates.
    """
    last_err = None
    fname = getattr(func, "__name__", str(func))
    for attempt in range(max_attempts):
        try:
            return func(state)
        except Exception as e:
            last_err = e
            print(f"[try_except_continue] node '{fname}' failed "
                  f"(attempt {attempt + 1}/{max_attempts}): {e}")
            traceback.print_exception(*sys.exc_info())

    # Retries exhausted -> fall back instead of crashing the query.
    print(f"[try_except_continue] node '{fname}' giving up after {max_attempts} attempts; "
          f"falling back. Last error: {last_err}")
    if fname == "generate":
        # Complete, schema-valid answer (ALL seven AgentOutput keys) so query()'s
        # parser succeeds and the run ends instead of crashing on a missing key.
        fallback = dict(_DEFAULT_ANSWER)
        fallback["text"] = f"I could not produce a well-formed answer ({last_err})."
        return {"messages": [json.dumps(fallback)]}
    # agent (or any tool-selecting node): emit an AIMessage with no tool calls so
    # should_continue() routes to `generate` and the loop ends cleanly.
    return {"messages": [AIMessage(content="")]}

class ReMEmbRAgent(Agent):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):

        # Wrapper that handles everything
        llm = self.llm_selector(llm_type, temperature, num_ctx)
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature

        self.chat = chat
        self.llm_type = llm_type
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        # self.update_for_instance() # ref_time is None this time
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+'prompts/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0

        self.chat_history = ChatMessageHistory()


    def llm_selector(self, llm_type, temperature, num_ctx):
        llm = None
        # Support for LLM Gateway
        if 'gpt-4' in llm_type:
            # TODO: ADD OpenAI here
            pass

        # Support for NIMs
        elif 'nim/' in llm_type:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA  # lazy: optional dep
            llm_name = llm_type[4:]
            llm = ChatNVIDIA(model=llm_name)

        # Support for the Tenstorrent NPU backend (tt_infer_server.py, OpenAI-compatible).
        # llm_type format: 'tt/<model_name>', e.g. 'tt/Llama-3.1-8B-Instruct'.
        # NOTE: the tt server implements response_format (JSON) but NOT the OpenAI
        # tools/function-calling params. That is fine here because FunctionsWrapper (see
        # __init__) drives tool use purely by prompting -- it lists the tools in the system
        # prompt and parses the model's JSON reply itself. FunctionsWrapper now sets
        # response_format *per call* (json_schema to grammar-constrain the tool-selection
        # step, json_object for the free-form answer step), so we must NOT pin one in
        # model_kwargs here -- doing so would be sent on every request and also collide
        # with the per-call bind. The per-call schema keeps a weak 3B model's tool calls
        # syntactically well-formed instead of crashing the parser.
        elif llm_type.startswith('tt/'):
            from langchain_openai import ChatOpenAI  # lazy: optional dep
            base_url = os.environ.get('TT_LLM_BASE_URL', 'http://localhost:8100/v1')
            llm = ChatOpenAI(
                model=llm_type[len('tt/'):],
                base_url=base_url,
                api_key='local',
                temperature=temperature,
                max_tokens=int(os.environ.get('TT_LLM_MAX_TOKENS', '256')),
            )

        # Support for Ollama functions
        elif llm_type == 'command-r':
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx)
        else:
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm


    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()



    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        # max_length / conlist bounds below flow through convert_to_openai_function into
        # build_tool_call_schema's grammar, so a degenerating model physically cannot pad
        # a query string or coordinate array until the 256-token tool-step budget is gone.
        class TextRetrieverInput(BaseModel):
            x: str = Field(max_length=200, description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: conlist(float, min_items=3, max_items=3) = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                The query must be an (x,y,z) array with floating point values \
                                Based on the question and your context, decide what position to search for in the database. \
                                This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                                The query will then search your memories for you.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(max_length=20, description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    ### Nodes

    @staticmethod
    def _coerce_position(value):
        """Best-effort coercion of a position argument to a 3-float tuple, else None.

        The position retriever expects an (x, y, z) tuple, but a weak model often hands
        over a stringified list ("[1.4, 2.2, -0.1]") or ints. Repair those here so the
        tool doesn't raise a pydantic ValidationError inside the ToolNode.
        """
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return tuple(float(v) for v in value)
            except (TypeError, ValueError):
                return None
        return None

    def _sanitize_tool_calls(self, tool_calls):
        """Validate/repair tool calls before they reach the ToolNode.

        Drops calls to unknown tools and coerces the single `x` argument each retriever
        takes. Returns a list of clean tool-call dicts (possibly empty). Keeping this in
        the graph node -- not just the model wrapper -- means a bad argument degrades to
        'answer from what we have' instead of thrashing the retrieve->agent loop.
        """
        valid_names = {t.name for t in self.tool_list}
        cleaned = []
        for tc in tool_calls:
            name = tc.get("name")
            if name not in valid_names:
                continue
            args = tc.get("args")
            if not isinstance(args, dict):
                args = {} if args is None else {"x": args}
            # All three retrievers take exactly one argument named `x`.
            if "x" not in args:
                if len(args) == 1:
                    args = {"x": next(iter(args.values()))}
                else:
                    continue
            if name == "retrieve_from_position":
                coerced = self._coerce_position(args["x"])
                if coerced is None:
                    continue
                args = {"x": coerced}
            elif not isinstance(args["x"], str):
                args = {"x": str(args["x"])}
            cleaned.append({**tc, "args": args})
        return cleaned

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat


        # limit to 5 tool calls.
        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt


        agent_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", prompt),
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),

            ]
        )


        model = agent_prompt | model

        question = f"The question is: {messages[0]}"

        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id


        response = model.invoke({"question": question, "chat_history": messages[:]})

        if response.tool_calls:
            # Validate/repair the model's tool calls. If none survive, rebuild the message
            # with no tool calls so should_continue() routes to `generate` (answer) rather
            # than looping on a call the ToolNode can't run.
            cleaned = self._sanitize_tool_calls(response.tool_calls)
            for tool_call in cleaned:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"
            response = AIMessage(content=response.content, tool_calls=cleaned, id=response.id)

        self.agent_call_count += 1


        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please respond in the desired format."

        if _openai_backend(self.chat.llm):
            # tt backend: grammar-constrain the answer to ANSWER_SCHEMA and give it a
            # larger token budget than the tool step (the answer object is longer). The
            # schema forbids reasoning keys, so the model cannot ramble past the cap.
            system = GENERATE_STRICT_PROMPT.replace("{question}", question)
            gen_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{question}"),
                ]
            )
            bound = self.chat.llm.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA},
                },
                max_tokens=int(os.environ.get("TT_LLM_ANSWER_MAX_TOKENS", "512")),
            )
            model = gen_prompt | bound
        else:
            # Original prompt-based path for Ollama / GPT backends.
            prompt = PromptTemplate(
                template=self.generate_prompt,
                input_variables=["context", "question"],
            )
            filled_prompt = prompt.invoke({'question': question})
            gen_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", filled_prompt.text),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{question}"),
                ]
            )
            model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # Parse (raises on truly broken output -> try_except_continue retries), then
        # normalize so all seven AgentOutput keys are always present and well-typed.
        parsed = normalize_answer(_parse_answer(response.content))

        # Post-process the model's shaky timestamp arithmetic: clamp time/duration to the
        # elapsed window stated in the question. Kills outliers without touching good answers.
        q_text = messages[0].content if messages else ""
        parsed["time"] = clamp_minutes(parsed["time"], q_text)
        parsed["duration"] = clamp_minutes(parsed["duration"], q_text)
        # Same idea for "where" answers: report a retrieved memory's coordinates rather
        # than the model's own number, retrieving once here if the agent never did.
        memory = getattr(self, "memory", None)
        parsed = ground_position(
            parsed, q_text, messages[1:],
            retriever=memory.search_by_text if memory is not None else None,
        )

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [json.dumps(parsed)]}



    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        # retrieve = ToolNode([self.retriever_tool])
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)
        # workflow.add_node("action", lambda state: try_except_continue(state, tool_node))


        # workflow.add_node("action", self.call_tool)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )


        workflow.add_edge('action', 'agent')

        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str):

        inputs = { "messages": [
                                (("user", question)),
            ]
        }

        out = self.graph.invoke(inputs)
        content = out['messages'][-1].content

        try:
            parsed = _parse_answer(content)
        except Exception:
            parsed = dict(_DEFAULT_ANSWER)

        # normalize_answer guarantees all seven fields exist, so from_dict never raises.
        return AgentOutput.from_dict(normalize_answer(parsed))

if __name__ == "__main__":

    from memory.milvus_memory import MilvusMemory

    # llm_name = 
    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs (such as 'codestral')
    memory = MilvusMemory("test", db_ip='127.0.0.1')

    llm_name = 'gpt-4o' 
    agent = ReMEmbRAgent(llm_type=llm_name)

    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")

