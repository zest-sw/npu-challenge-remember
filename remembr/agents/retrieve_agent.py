"""Fixed query -> retrieve -> answer, with no tool-calling loop.

Why this exists
---------------
`ReMEmbRAgent` makes the planner invent its search query as a tool argument, and
on this board it does not: asked when the robot left a building it searched for
"a green car driving down the road", elsewhere "rainbow" and even "your available
tools". Retrieval then returns unrelated memories and every later stage reasons
from the wrong evidence.

Two probes (`remembr/scripts/probe_query_generation.py`,
`probe_query_format.py`) established that neither retrieval nor the model is at
fault -- handed a human's query the same captions/embeddings/Milvus land 3 of 4
position questions within 5 m, and asked the same thing as a plain question the
model matches that 3/4 (and beat the human on one). The tool-calling format is
what breaks it. So drop the format instead of fixing it: ask for the query
plainly, retrieve with it, and let `NonAgent` answer.

`NonAgent` already answers from `memory.get_working_memory()`, and
`Memory.search_by_text()` already appends what it finds to working memory. This
class is the two of them wired together, which is also why it is comparable to
`optimal+`: identical answer stage, real retrieval instead of oracle context.

    python remembr/scripts/eval.py --model retrieve+tt/Llama-3.1-8B-Instruct ...
"""
import os

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from remembr.agents.agent import AgentOutput
from remembr.agents.non_agent import NonAgent
from remembr.utils import trace
from remembr.utils.util import file_to_string

# The phrase is at most 6 words, so this is generous. Capped because the server
# charges per generated token and a rambling query is a worse query anyway.
QUERY_MAX_TOKENS = 24


class RetrieveAgent(NonAgent):
    def __init__(self, llm_type='llama3', num_ctx=8192, temperature=0):
        super().__init__(llm_type=llm_type, num_ctx=num_ctx, temperature=temperature)
        self.query_prompt = file_to_string(
            os.path.join(os.path.dirname(__file__), '..', 'prompts', 'query_system_prompt.txt')
        ).strip()

        # A separate chain on purpose: NonAgent's is locked to the answer JSON
        # schema, and the whole point here is to ask for the query with no schema
        # and no tool spec in front of it. That plain form is the one the model
        # answers well -- constraining it is what was being avoided.
        if llm_type.startswith('tt/'):
            from langchain_openai import ChatOpenAI

            self.query_chain = ChatOpenAI(
                model=llm_type[len('tt/'):],
                base_url=os.environ.get('TT_LLM_BASE_URL', 'http://localhost:8100/v1'),
                api_key='local',
                temperature=temperature,
                max_tokens=QUERY_MAX_TOKENS,
            )
        else:
            self.query_chain = ChatOllama(model=llm_type, num_ctx=num_ctx, temperature=temperature)

    def search_query_for(self, question: str) -> str:
        """The phrase to search memory with, from the question alone.

        Only the asked line is sent. The question's preamble states the current
        time and position, and including it pulled the model toward searching for
        those instead of what was asked.
        """
        asked = self.asked_line(question)
        response = self.query_chain.invoke(
            [SystemMessage(content=self.query_prompt), HumanMessage(content=asked)]
        )
        # A model that adds quotes or a trailing period still gave a usable
        # phrase; strip the packaging rather than reject the answer.
        return response.content.strip().strip('"\'').rstrip('.').strip()

    @staticmethod
    def asked_line(question: str) -> str:
        lines = [line.strip() for line in question.strip().splitlines() if line.strip()]
        return lines[-1] if lines else question.strip()

    def query(self, question: str) -> AgentOutput:
        search_query = self.search_query_for(question)
        trace.step("검색어", repr(search_query))
        if search_query:
            # Fills memory.working_memory, which NonAgent.query then answers from.
            # ponytail: one text search, no time/position search and no second
            # round. Enough for the position questions the probe measured; if
            # "how long were you in X" needs both the entering and the leaving
            # memory, add a second query here rather than restoring the agent loop.
            # The memory traces what came back, so nothing to add here.
            self.memory.search_by_text(search_query)
        else:
            trace.warn("빈 검색어 — 컨텍스트 없이 답변한다")
        return super().query(question)
