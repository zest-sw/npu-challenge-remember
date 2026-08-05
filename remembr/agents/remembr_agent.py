from typing import Annotated, Literal, Sequence, TypedDict
import json
import traceback
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
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field


import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string, parse_json
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.memory import Memory

from remembr.agents.agent import (ANSWER_SCHEMA, Agent, AgentOutput, answer_summary,
                                  schema_for_type, settle_answer_type, validate_answer)
from remembr.utils import trace



GENERATE_SCHEMA_PROMPT = """You are a robot that can answer specific kinds of questions relating to your memory.
Use the retrieved conversation context to answer the user's question.
Respond with exactly one JSON object and nothing else.
The JSON object must contain all of these keys: type, text, binary, position, orientation, time, duration.
Use null for any field that does not apply.
- type: exactly one of position, time, duration, binary, text -- nothing else
- text: a short answer to the user
- binary: yes/no for binary questions, otherwise null
- position: an array like [x, y, z] for where questions, otherwise null
- orientation: the yaw in radians as a number when position is provided, otherwise null
- time: minutes ago as a number, not words. 11, not "11 minutes ago". Otherwise null
- duration: minutes as a number, not words. Otherwise null"""



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


# How many turns the planner may spend calling tools before it must answer. Each
# turn is a full LLM generation (~9-13s measured on 8B), so this is the main knob
# on how long one question takes.
MAX_TOOL_TURNS = 3

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
        self.answer_type_prompt = file_to_string(top_level_path+'prompts/answer_type_prompt.txt').strip()

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        # (tool name, args) already run for this question, so a repeat can be dropped.
        self.tool_calls_made = set()

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



    def create_tools(self, memory):


        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        # Give these descriptions NO value to copy, but DO tell the model to search.
        # A filled-in sample is an instruction to a small model, not an illustration:
        # measured on the board (Llama-3.1-8B), every irrelevant argument the planner
        # produced was a verbatim copy of the sample that used to sit here — asked
        # when the robot left a building it searched for "a green car driving down the
        # road" and "a crowd gathering", queried time 08:02:03 and position
        # (0.5, 0.2, 0.1). 4 of 4 bad arguments traced back to those strings.
        #
        # But cutting the samples took the imperative out with them, and that went
        # too far: with only "the thing to look for", the model stopped calling the
        # tool at all and answered "I do not have enough information" from an empty
        # memory (0 retrievals across 9 questions). Restoring "There must always be
        # text here / decide what to search for" — the instruction, not the examples —
        # brought retrieval back with a sane query ("place to sit"). Keep both
        # properties: an imperative to search, and nothing that can be copied.
        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The thing to look for, as a short visual phrase \
                                taken from what the question asks about. There must always be \
                                text here. Based on the question, decide what to search your \
                                memory for.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple = Field(description="An (x,y,z) array copied from a coordinate a \
                                RETRIEVED MEMORY gave you. Never the position stated in the \
                                question -- that is where you are now, not where the thing \
                                you are looking for is. Never invent one.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="An HH:MM:SS clock time, copied from the question \
                                or from a retrieved memory. Never invent one. Only useful \
                                when you already know when something happened.")

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
        # A retrieval-only variant for the turns where the planner still has tool
        # budget. `__conversational_response` is useless there: this node runs to pick
        # tools, and `generate` writes the answer afterwards. Worse, the model emitted it
        # alongside a real tool call, FunctionsWrapper kept the tool and threw the reply
        # away (see #35), and we still paid to generate it -- 500 characters, ~125 tokens,
        # about 31s at the measured decode rate, for text nothing reads. Dropping the
        # option makes it ungenerable. The turn after the budget runs out does not call
        # the LLM at all (see MAX_TOOL_TURNS above), so no turn needs the option.
        self.tool_call_schema = {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"const": "__conversational_response"},
                            "tool_input": {
                                "type": "object",
                                "properties": {
                                    "response": {"type": "string", "maxLength": 500}
                                },
                                "required": ["response"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "tool_input"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"const": "retrieve_from_text"},
                            "tool_input": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "string", "maxLength": 200}
                                },
                                "required": ["x"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "tool_input"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"const": "retrieve_from_position"},
                            "tool_input": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 3,
                                        "maxItems": 3,
                                    }
                                },
                                "required": ["x"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "tool_input"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"const": "retrieve_from_time"},
                            "tool_input": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "string",
                                        "pattern": "^\\d{2}:\\d{2}:\\d{2}$"
                                    }
                                },
                                "required": ["x"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "tool_input"],
                        "additionalProperties": False,
                    },
                ]
            },
            "minItems": 1,
            "maxItems": 4,
        }
        # Every tool stays available on every turn. The prompt says to start with
        # retrieve_from_text -- to learn when or where something happened you have to find
        # the thing first, and only a retrieved memory can give you a time or a position to
        # search by -- but that is left as instruction rather than grammar on purpose.
        #
        # Restricting turn 1 to text was tried and pulled back. It was derived from this
        # dataset having 0 of 30 questions that name a clock time, which overfits: "what did
        # you see at 08:02?" legitimately starts with a time search. And the evidence for
        # "the prompt gets ignored, only grammar holds" was collected while this prompt was
        # itself telling the model to call at least two tools and showing a two-call example.
        # That prompt is fixed now, so the claim is untested. Measure the instruction first;
        # constrain only if it does not hold.
        #
        # The conversational reply stays in every variant: it is the loop's only exit signal
        # (`should_continue` ends on a message with no tool_calls, and FunctionsWrapper only
        # produces one for `__conversational_response`). Removing it was measured at
        # 270s -> 389s per question because every question then burned all 3 turns. Its
        # length is the waste -- the reply is discarded whenever it arrives next to a real
        # tool call (#35) and we still pay ~125 tokens -- so it is capped at 40 characters.
        self.turn_schema = self._schema_allowing(None, reply_length=40)
        # Used while no memory has been retrieved yet. Searching by a time or a position
        # needs a value, and the only honest sources are a retrieved memory or the
        # question -- and the question's values are already ruled out (its position is
        # where you stand now, its time is the present, neither is what you are looking
        # for). With nothing retrieved there is no value to search by, so text is the
        # only move that can work.
        #
        # This is enforced rather than asked because it was measured both ways. The
        # prompt now says "Start with retrieve_from_text" in as many words, and the
        # planner still spent all three turns on retrieve_from_time -- 07:52:11,
        # 08:00:53, 08:00:00 -- and never called text search once. The instruction does
        # not hold.
        #
        # Cost: a question that legitimately opens with a known time ("what did you see
        # at 08:02?") loses its first turn. This dataset has none of those (0 of 30 name
        # a clock time) and the remaining two turns can still do it.
        self.no_memory_yet_schema = self._schema_allowing(
            {"__conversational_response", "retrieve_from_text"}, reply_length=40)

    def _schema_allowing(self, tool_names, reply_length):
        """A copy of tool_call_schema keeping only `tool_names` (None keeps all)."""
        schema = json.loads(json.dumps(self.tool_call_schema))
        options = schema["items"]["oneOf"]
        if tool_names is not None:
            options = [o for o in options
                       if o["properties"]["tool"].get("const") in tool_names]
            schema["items"]["oneOf"] = options
        for option in options:
            if option["properties"]["tool"].get("const") == "__conversational_response":
                option["properties"]["tool_input"]["properties"]["response"]["maxLength"] = reply_length
        return schema

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
        messages = state["messages"]

        if self.agent_call_count >= MAX_TOOL_TURNS:
            # Tool budget spent -> hand straight to generate() without an LLM call.
            # This branch used to bind tools=[] and still run a full generation, so
            # the model could not emit a tool call and instead wrote prose that
            # should_continue routed to "end" and generate() then discarded by
            # re-answering from the same context. Measured on the board: that turn
            # cost 53s of a 118s question -- 45% -- for output nothing reads.
            # Returning a tool-call-free message reaches generate() the same way,
            # for free. (self.agent_gen_only_prompt is no longer used by this path.)
            self.agent_call_count += 1
            return {"messages": [AIMessage(
                content="Enough memory has been retrieved. Answer the question from the context above."
            )]}

        model = self.chat


        # limit to 5 tool calls.
        tool_kwargs = {}
        if self.llm_type.startswith('tt/'):
            tool_kwargs = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tool_calls",
                        "schema": (self.no_memory_yet_schema
                                   if not self.memory.get_working_memory()
                                   else self.turn_schema)},
                }
            }
        model = model.bind_tools(tools=self.tool_definitions, **tool_kwargs)
        prompt = self.agent_prompt


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

        turn = self.agent_call_count + 1

        # A turn that only repeats a search already run is the loop telling us it has
        # nothing left to look for. Measured on the board: with a 3-turn budget the
        # planner asked `retrieve_from_time(48)` on turns 1 and 3, and 8 on turns 2 and 3
        # of another question -- identical arguments, identical results, a third of the
        # budget spent twice. `previous_tool_requests` does tell it what it already used
        # (that was #48's fix) and it repeats anyway, so drop the repeats here instead of
        # asking again. With every call dropped the message has no tool_calls left, which
        # `should_continue` already routes to `generate` -- so a repeat becomes an early
        # exit rather than a wasted turn.
        if response.tool_calls:
            fresh = []
            for tool_call in response.tool_calls:
                signature = (tool_call['name'], repr(tool_call['args']))
                if signature in self.tool_calls_made:
                    trace.step(f"턴{turn}", f"{tool_call['name']}({tool_call['args']}) — 중복, 건너뜀")
                    continue
                self.tool_calls_made.add(signature)
                fresh.append(tool_call)
            if len(fresh) != len(response.tool_calls):
                response.tool_calls = fresh

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    # Strip the braces only, not what is inside them. They have to go
                    # because `previous_tool_requests` is fed to a ChatPromptTemplate,
                    # which reads `{...}` as a variable. But the old
                    # `re.sub("\{.*?\}", "", ...)` deleted the whole dict -- its own repr
                    # is brace-delimited -- so the model was told "I previously used the
                    # retrieve_from_text tool with the arguments: ." and could not tell it
                    # had already run that search. Measured on the board: it repeated the
                    # identical search on the next turn, burning a third of the 3-turn
                    # budget.
                    args = str(tool_call['args']).replace('{', '').replace('}', '')
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"
                    trace.step(f"턴{turn}", f"{tool_call['name']}({tool_call['args']})")
        else:
            # No tool call means should_continue routes straight to generate. Worth
            # seeing: this is what a planner that gave up looks like.
            trace.step(f"턴{turn}", f"tool 호출 없음 — {str(response.content)[:80]!r}")

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
                + "\n Please responsed in the desired format."
        if self.llm_type.startswith('tt/'):
            # Settle the answer type first, then answer under a grammar that cannot leave
            # that type's field null. Without this the planner produced `type="position"`
            # for a time question, scored unanswered, and eval retried the whole 3-turn
            # tool loop five times -- up to 20 LLM calls for one question. NonAgent has
            # had this since #40; this path was the gap.
            answer_type = settle_answer_type(self.llm, messages[0].content,
                                             self.answer_type_prompt)
            trace.step("유형", answer_type if answer_type else "확정 실패 — permissive 스키마")
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
                    "json_schema": {"name": "agent_output",
                                    "schema": schema_for_type(answer_type)},
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

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = json.loads(response)
            else:
                parsed = parse_json(response)

            validate_answer(parsed)

        except Exception as error:
            trace.warn(f"답변 파싱 실패: {error}")
            trace.note(f"원본: {response[:160]}")
            raise ValueError("Generate call failed. Retrying...")

        trace.step("답변", answer_summary(parsed))
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.tool_calls_made = set()
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
