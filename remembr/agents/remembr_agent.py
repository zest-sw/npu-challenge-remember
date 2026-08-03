from typing import Annotated, Literal, Sequence, TypedDict
import json
import traceback
import uuid
import datetime
import sys, re

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
from langchain_core.messages import ToolMessage, AIMessage, ToolCall
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field


import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.memory import Memory

from remembr.agents.agent import Agent, AgentOutput


ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'text': {'type': 'string', 'maxLength': 300},
        'binary': {'type': ['string', 'null']},
        'position': {
            'anyOf': [
                {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'minItems': 3,
                    'maxItems': 3,
                },
                {'type': 'null'},
            ]
        },
        'orientation': {
            'anyOf': [
                {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'minItems': 1,
                    'maxItems': 1,
                },
                {'type': 'null'},
            ]
        },
        'time': {'type': ['number', 'null']},
        'duration': {'type': ['number', 'null']},
    },
    'required': ['type', 'text', 'binary', 'position', 'orientation', 'time', 'duration'],
    'additionalProperties': False,
}

GENERATE_SCHEMA_PROMPT = """You are a robot that can answer specific kinds of questions relating to your memory.
Use the retrieved conversation context to answer the user's question.
Respond with exactly one JSON object and nothing else.
The JSON object must contain all of these keys: type, text, binary, position, orientation, time, duration.
Use null for any field that does not apply.
- type: the answer category
- text: a short answer to the user
- binary: yes/no for binary questions, otherwise null
- position: an array like [x, y, z] for where questions, otherwise null
- orientation: an array like [yaw] when position is provided, otherwise null
- time: a number in minutes ago for "when" questions, otherwise null. Never output an absolute timestamp here.
- duration: a number in minutes for "how long" questions, otherwise null. Never output an absolute timestamp here.
For example, if the current time is 08:03:02 and the event happened at 07:55:32, time must be 7.5, not "07:55:32" and not "2023-01-16 07:55:32".
Expected answer type for this question: {expected_type}.
Do not answer with an image-description type unless the user asked what the image shows."""

TT_AGENT_SYSTEM_PROMPT = """You are a memory retrieval planner for a robot.
Choose exactly one action as JSON. Do not write XML, markdown, or prose.
Expected final answer type for this question: {expected_type}.

Available actions:
- retrieve_from_time: use only when the actual question asks for memories near an explicit HH:MM:SS timestamp.
- retrieve_from_position: use only when the schema permits it and the question asks for memories near an explicit [x, y, z] coordinate.
- retrieve_from_text: use for objects, places, activities, entrances, exits, doors, stairs, signs, durations, or other visual concepts.
- __conversational_response: use only after retrieved memory gives enough evidence to answer.

Rules:
1. On the first planner call for an episodic memory question, retrieve memory instead of answering directly.
2. Use only the listed action names.
3. For retrieve_from_time, tool_input.x must be exactly HH:MM:SS.
4. For retrieve_from_position, tool_input.x must be [x, y, z].
5. For retrieve_from_text, tool_input.x must be a short visual search phrase, not a full prompt.
6. For when questions about an event, object, or place, use retrieve_from_text with the event/object/place phrase; do not choose retrieve_from_time just because the preamble mentions start/current time.
7. For duration/how-long questions, use retrieve_from_text with the event or activity phrase; do not choose retrieve_from_time just because the question mentions time.
8. For where/object-location questions, use retrieve_from_text with the object or place phrase; do not choose retrieve_from_position just because the question includes the robot's current coordinates.
9. If retrieved memory is enough, summarize the relevant evidence with __conversational_response."""



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
    return json.loads(parsed)


def _extract_datetime(text: str):
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", text)
    if match is None:
        return None
    return datetime.datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S")


def _extract_current_datetime(question: str):
    match = re.search(r"current time is\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", question, re.IGNORECASE)
    if match is None:
        return _extract_datetime(question)
    return datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _canonicalize_minutes(value, question: str):
    if value is None or isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass

    current_dt = _extract_current_datetime(question)
    if current_dt is None:
        return None

    event_dt = _extract_datetime(stripped)
    if event_dt is None:
        hms = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", stripped)
        if hms is None:
            return None
        event_time = datetime.datetime.strptime(hms.group(1), "%H:%M:%S").time()
        event_dt = datetime.datetime.combine(current_dt.date(), event_time)
    delta_min = (current_dt - event_dt).total_seconds() / 60.0
    if delta_min < 0:
        # Retrieval context can cross date boundaries after offset conversion.
        # A future absolute time is not a valid minutes-ago answer for eval.
        return None
    return float(delta_min)


def _expected_answer_type(question: str) -> str:
    q = _asked_question_text(question).lower()
    if "how long" in q or "duration" in q:
        return "duration"
    if re.search(r"\bwhen\b", q):
        return "time"
    if re.search(r"\bwhere\b", q) or "located" in q or "location" in q:
        return "position"
    if re.match(r"\s*(can|could|did|do|does|is|are|was|were|has|have)\b", q):
        return "binary"
    return "text"


def _asked_question_text(question: str) -> str:
    text = re.sub(r"^\s*The question is:\s*", "", question.strip(), flags=re.IGNORECASE)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not re.match(r"please\s+respons", line.strip(), flags=re.IGNORECASE)
    ]
    return lines[-1] if lines else text


def _tt_text_query_from_question(question: str) -> str:
    asked = _asked_question_text(question)
    asked = re.sub(r"\?\s*$", "", asked).strip()
    replacements = [
        r"^when\s+(were|was)\s+(you|the robot)\s+",
        r"^when\s+did\s+(you|the robot)\s+",
        r"^where\s+(is|are)\s+",
        r"^how\s+long\s+(were|was)\s+(you|the robot)\s+",
        r"^(are|is|was|were)\s+(you|the robot)\s+",
    ]
    query = asked
    for pattern in replacements:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
    query = re.sub(r"\b(right now|currently|for)\b", "", query, flags=re.IGNORECASE).strip()
    query = re.sub(r"^(in|at|near|around|by)\s+", "", query, flags=re.IGNORECASE).strip()
    query = re.sub(r"^the\s+", "", query, flags=re.IGNORECASE).strip()
    return (query or asked)[:200]


def _tt_answer_schema_for(question: str) -> dict:
    expected_type = _expected_answer_type(question)
    schema = json.loads(json.dumps(ANSWER_SCHEMA))
    schema["properties"]["type"] = {"type": "string", "enum": [expected_type]}
    if expected_type == "time":
        schema["properties"]["time"] = {"type": "number"}
        schema["properties"]["duration"] = {"type": "null"}
        schema["properties"]["position"] = {"type": "null"}
        schema["properties"]["orientation"] = {"type": "null"}
        schema["properties"]["binary"] = {"type": "null"}
    elif expected_type == "duration":
        schema["properties"]["duration"] = {"type": "number"}
        schema["properties"]["time"] = {"type": "null"}
        schema["properties"]["position"] = {"type": "null"}
        schema["properties"]["orientation"] = {"type": "null"}
        schema["properties"]["binary"] = {"type": "null"}
    elif expected_type == "position":
        schema["properties"]["time"] = {"type": "null"}
        schema["properties"]["duration"] = {"type": "null"}
        schema["properties"]["binary"] = {"type": "null"}
    elif expected_type == "binary":
        schema["properties"]["binary"] = {"type": "string", "enum": ["yes", "no"]}
        schema["properties"]["time"] = {"type": "null"}
        schema["properties"]["duration"] = {"type": "null"}
        schema["properties"]["position"] = {"type": "null"}
        schema["properties"]["orientation"] = {"type": "null"}
    else:
        schema["properties"]["time"] = {"type": "null"}
        schema["properties"]["duration"] = {"type": "null"}
        schema["properties"]["position"] = {"type": "null"}
        schema["properties"]["orientation"] = {"type": "null"}
        schema["properties"]["binary"] = {"type": "null"}
    return schema

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
    

# Cap retries per node call. The original `while True` re-ran forever when the LLM
# couldn't produce parseable tool/answer JSON (common with small models: they truncate
# the verbose reasoning array or emit a wrong-shaped value), hanging eval on one
# question. After 3 tries we re-raise so the exception propagates to
# eval.answer_squad_question, whose MAX_ATTEMPTS=5 cap + None->unanswered handling then
# govern the question exactly as they do for the optimal+ (NonAgent) path.
def try_except_continue(state, func, max_retries=3):
    last_exc = None
    for _ in range(max_retries):
        try:
            return func(state)
        except Exception as e:
            last_exc = e
            print("I crashed trying to run:", func)
            traceback.print_exception(*sys.exc_info())
            continue
    raise last_exc

class ReMEmbRAgent(Agent):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):

        # Wrapper that handles everything
        llm = self.llm_selector(llm_type, temperature, num_ctx)
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature

        self.chat = chat
        self.llm = llm
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
        self.used_tt_actions = set()

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

        # Tenstorrent NPU backend: same local unified server as NonAgent, but here
        # the LLM is wrapped by FunctionsWrapper (prompt-based tool calling), so it
        # just needs to return valid JSON text. Use the server's generic json_object
        # mode (== Ollama format="json") rather than a fixed schema, because the
        # agent node emits {"tool", "tool_input"} while generate emits the answer
        # dict — the shape varies per call.
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

    def _tt_action_schema(self, question: str, allow_answer: bool):
        expected_type = _expected_answer_type(question)
        if expected_type == "time":
            asked = _asked_question_text(question)
            if re.search(r"\b\d{2}:\d{2}:\d{2}\b", asked):
                tools = ["retrieve_from_time", "retrieve_from_text"]
            else:
                # Benchmark "when" questions usually ask for the time of a
                # visual event/place, while the preamble contains unrelated
                # start/current timestamps. Search the event caption first.
                tools = ["retrieve_from_text"]
        elif expected_type == "position":
            # Object-location questions are answered best by retrieving the
            # caption that mentions the target, then using that memory's stored
            # position. The benchmark question preamble always includes the
            # robot's current coordinates, so allowing position search here makes
            # the planner overfit to the preamble instead of the asked object.
            tools = ["retrieve_from_text"]
        else:
            # Duration/binary/text questions should retrieve event/object evidence,
            # not blindly query a timestamp from the question.
            tools = ["retrieve_from_text"]
        if allow_answer:
            tools.append("__conversational_response")
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": tools},
                "tool_input": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "anyOf": [
                                {"type": "string", "maxLength": 200},
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                },
                            ]
                        },
                        "response": {"type": "string", "maxLength": 500},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["tool", "tool_input"],
            "additionalProperties": False,
        }

    def _parse_tt_action(self, content: str, question: str = "") -> dict:
        try:
            action = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"TT planner did not return valid JSON: {content}") from e

        tool = action.get("tool")
        tool_input = action.get("tool_input")
        if not isinstance(tool_input, dict):
            raise ValueError(f"TT planner tool_input must be an object: {action}")

        if tool == "retrieve_from_text":
            query = tool_input.get("x")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"retrieve_from_text requires non-empty string x: {action}")
            action["tool_input"] = {"x": query.strip()[:200]}
        elif tool == "retrieve_from_time":
            query = tool_input.get("x")
            if not isinstance(query, str):
                raise ValueError(f"retrieve_from_time requires HH:MM:SS string x: {action}")
            # The planner sometimes includes the date ("2023-01-16 07:55:32")
            # even though MilvusMemory.search_by_time only wants HMS. Treat that
            # as a canonicalization step, not a planner failure.
            match = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", query.strip())
            if match is None:
                action["tool"] = "retrieve_from_text"
                action["tool_input"] = {"x": _tt_text_query_from_question(question)}
                return action
            action["tool_input"] = {"x": match.group(1)}
        elif tool == "retrieve_from_position":
            query = tool_input.get("x")
            if not isinstance(query, list) or len(query) != 3:
                raise ValueError(f"retrieve_from_position requires [x, y, z]: {action}")
            action["tool_input"] = {"x": [float(v) for v in query]}
        elif tool == "__conversational_response":
            response = tool_input.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"__conversational_response requires response: {action}")
            action["tool_input"] = {"response": response.strip()[:500]}
        else:
            raise ValueError(f"TT planner selected unknown tool: {action}")
        return action

    def _tt_agent(self, state):
        messages = state["messages"]
        max_tool_calls = int(os.environ.get("TT_MAX_TOOL_CALLS", "1"))
        if self.agent_call_count >= max_tool_calls:
            return {"messages": [AIMessage(
                content=(
                    "Use the retrieved memory context above to produce the final answer. "
                    "Do not call another retrieval tool."
                )
            )]}
        allow_answer = self.agent_call_count > 0
        question = getattr(messages[0], "content", str(messages[0]))
        expected_type = _expected_answer_type(question)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TT_AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{previous_tool_requests}"),
                ("human", "{question}"),
            ]
        )
        model = prompt | self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_action",
                    "schema": self._tt_action_schema(question, allow_answer=allow_answer),
                },
            },
            max_tokens=int(os.environ.get("TT_TOOL_MAX_TOKENS", "96")),
            temperature=float(os.environ.get("TT_TOOL_TEMPERATURE", "0")),
        )

        # The local TT backend handles ordinary message content better than
        # LangChain ToolMessage objects. Preserve tool outputs as assistant text.
        chat_history = []
        for message in messages[1:]:
            if isinstance(message, ToolMessage):
                chat_history.append(AIMessage(id=message.id, content=message.content))
            else:
                chat_history.append(message)

        response = model.invoke({
            "question": f"The question is: {question}",
            "chat_history": chat_history,
            "previous_tool_requests": self.previous_tool_requests,
            "expected_type": expected_type,
        })
        action = self._parse_tt_action("".join(response.content.splitlines()), question=question)
        self.agent_call_count += 1

        if action["tool"] == "__conversational_response":
            return {"messages": [AIMessage(content=action["tool_input"]["response"])]}

        action_key = json.dumps(
            {"tool": action["tool"], "tool_input": action["tool_input"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        if action_key in self.used_tt_actions:
            return {"messages": [AIMessage(
                content=(
                    "The requested memory has already been retrieved. "
                    "Use the retrieved memory context above to produce the final answer."
                )
            )]}
        self.used_tt_actions.add(action_key)

        self.previous_tool_requests += (
            f"I previously used the {action['tool']} tool with the arguments: "
            f"{action['tool_input']}.\n"
        )
        tool_call = ToolCall(
            name=action["tool"],
            args=action["tool_input"],
            id=f"call_{str(uuid.uuid4()).replace('-', '')}",
        )
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}



    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
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
            x: tuple = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
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
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
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

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        if self.llm_type.startswith('tt/'):
            return self._tt_agent(state)

        messages = state["messages"]

        model = self.chat


        # limit to 5 tool calls.
        tool_kwargs = {}
        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions, **tool_kwargs)
            prompt = self.agent_prompt
        else:
            model = model.bind_tools(tools=[], **tool_kwargs)
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
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub(r"\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"

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
        raw_question = messages[0].content
        question = raw_question + "\n Please responsed in the desired format."
        if self.llm_type.startswith('tt/'):
            expected_type = _expected_answer_type(raw_question)
            gen_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", GENERATE_SCHEMA_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{question}"),
                ]
            )
            model = gen_prompt | self.llm.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "agent_output", "schema": _tt_answer_schema_for(raw_question)},
                },
                max_tokens=int(os.environ.get('TT_LLM_MAX_TOKENS', '256')),
            )
        else:
            last_message = messages[-1]
            docs = last_message.content

            prompt = PromptTemplate(
                template=self.generate_prompt,
                input_variables=["context", "question"],
            )
            filled_prompt = prompt.invoke({'question':question})


            gen_prompt = ChatPromptTemplate.from_messages(
                [
                    # ("human", "What do you do?"),
                    ("system", filled_prompt.text),
                    MessagesPlaceholder("chat_history"),
                    # ("ai", filled_prompt.text),
                    ("human", "{question}"),

                ]
            )

            model = gen_prompt | self.chat

        invoke_inputs = {
            "question": question,
            "chat_history": messages[1:],
        }
        if self.llm_type.startswith('tt/'):
            invoke_inputs["expected_type"] = _expected_answer_type(raw_question)
        response = model.invoke(invoke_inputs)

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = json.loads(response)
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")

            parsed['time'] = _canonicalize_minutes(parsed.get('time'), question)
            parsed['duration'] = _canonicalize_minutes(parsed.get('duration'), question)
            
            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.used_tt_actions = set()
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
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = json.loads(response)
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)


        return response

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
