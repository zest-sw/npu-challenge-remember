import json
import numpy as np
import sys, os
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

sys.path.append(sys.path[0] + '/..')
from remembr.tools.tools import format_docs
from remembr.utils.util import file_to_string

from remembr.agents.agent import Agent, AgentOutput
from remembr.utils import trace
from remembr.memory.memory import Memory


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    parsed = json.loads(parsed)
    return parsed


POSITION_ONLY_SCHEMA = {
    'type': 'object',
    'properties': {
        'position': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 3,
            'maxItems': 3,
        },
    },
    'required': ['position'],
    'additionalProperties': False,
}


def _answer_summary(output: AgentOutput) -> str:
    """The filled fields only, so the trace line shows what the model committed to."""
    filled = [(name, getattr(output, name, None))
              for name in ('type', 'position', 'time', 'duration', 'binary')]
    return "  ".join(f"{name}={value!r}" for name, value in filled if value is not None)


class NonAgent(Agent):
    def __init__(self, llm_type='llama3', num_ctx=8192, temperature=0):
        
        self.llm_type = llm_type

        if llm_type == 'gpt-4o':
            # TODO: ADD OpenAI key here!
            pass
        elif llm_type.startswith('tt/'):
            # Tenstorrent NPU backend: talk to the local unified inference server
            # (tt_infer_server.py) via its OpenAI-compatible /v1/chat/completions endpoint.
            # llm_type format: 'tt/<model_name>', e.g. 'tt/Llama-3.2-3B-Instruct'.
            from langchain_openai import ChatOpenAI
            base_url = os.environ.get('TT_LLM_BASE_URL', 'http://localhost:8100/v1')
            # max_tokens here = max GENERATED tokens (not context); keep modest so
            # prefill+decode fits the server's max_seq_len.
            # A JSON *schema* triggers the server's grammar-constrained decoding
            # (the equivalent of Ollama's format="json", but stronger): it forces
            # the required keys, caps free-text length, and forbids extra keys so a
            # small model can't ramble in a reasoning field and blow the budget.
            answer_schema = POSITION_ONLY_SCHEMA
            self.chain = ChatOpenAI(
                model=llm_type[len('tt/'):],
                base_url=base_url,
                api_key='local',
                temperature=temperature,
                max_tokens=int(os.environ.get('TT_LLM_MAX_TOKENS', '256')),
                model_kwargs={'response_format': {
                    'type': 'json_schema',
                    'json_schema': {'name': 'agent_output', 'schema': answer_schema},
                }},
            )
            self.answer_schema = answer_schema
        else:
            self.chain = ChatOllama(model=llm_type, num_ctx=num_ctx, temperature=temperature)
            self.answer_schema = None
        self.prompt = file_to_string(str(os.path.dirname(__file__)) + '/../' + 'prompts/non_agent_system_prompt.txt')


    def set_memory(self, memory: Memory):
        self.memory = memory
        pass


    def query(self, question: str) -> AgentOutput:


        # GROUND TRUTH RETRIEVAL
        # Prompt
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'context': self.memory.memory_to_string(self.memory.get_working_memory()), 'question': question})
        inputs = filled_prompt.text

        chain = self.chain
        trace.step("유형", "position 고정")

        # Cap the retries. This loop was `while True`, which hangs outright when
        # the model cannot produce parseable output: at temperature=0 every retry
        # is byte-identical, so a single bad generation repeats forever. Measured
        # on the board — one question spent 975s in 26 identical retries of the
        # same truncated JSON and would never have returned, stalling the whole
        # eval on question 10 of 30. eval.answer_squad_question already has its own
        # MAX_ATTEMPTS=5 around this call and treats an exception as an unanswered
        # attempt, so raising here lets the run continue instead of blocking.
        # (temperature>0 makes retries differ and usually recovers, but the loop
        # must not depend on the caller choosing a temperature to terminate.)
        MAX_PARSE_RETRIES = 5
        last_error = None

        for parse_try in range(1, MAX_PARSE_RETRIES + 1):

            response = chain.invoke(inputs)
            response = ''.join(response.content.splitlines())
            try:
                if '```json' not in response:
                    # try parsing on its own since we cannot always trust llms
                    parsed = json.loads(response) 
                else:
                    parsed = parse_json(response)

                position = parsed.get("position")
                if (not isinstance(position, list) or len(position) != 3
                        or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                   for value in position)):
                    raise ValueError(
                        f"Position must contain exactly three numbers: {position!r}"
                    )
                # Keep the public/eval return contract while making the model generate
                # only the field that position evaluation consumes.
                response = AgentOutput(
                    type="position",
                    text=None,
                    binary=None,
                    position=position,
                    orientation=None,
                    duration=None,
                    time=None,
                )

            except Exception as e:
                last_error = e
                trace.warn(f"파싱 실패 {parse_try}/{MAX_PARSE_RETRIES}: {e}")
                trace.note(f"원본: {response}")
                continue

            trace.step("답변", _answer_summary(response))
            break

        else:
            # Every retry failed. Raise rather than return something unparsed:
            # eval's attempt loop records this as an unanswered attempt and moves
            # on, which is what we want. Silently returning a broken answer would
            # be scored as if the model had answered.
            raise ValueError(
                f"NonAgent could not produce a parseable answer in {MAX_PARSE_RETRIES} "
                f"tries; last error: {last_error}"
            )

        return response
