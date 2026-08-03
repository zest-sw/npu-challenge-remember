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
from remembr.memory.memory import Memory


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    parsed = json.loads(parsed)
    return parsed

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
            answer_schema = {
                'type': 'object',
                'properties': {
                    'type': {'type': 'string'},
                    'text': {'type': 'string', 'maxLength': 300},
                    'binary': {'type': ['string', 'null']},
                    # maxLength matters here, it is not cosmetic. Without it the
                    # grammar permits an unbounded string and the model runs away
                    # inside it: observed on the board, position filled with
                    # "[-55.74, 86.31, -2.11], [0, 0, 0], [0, 0, 1], [0, 0, 0], ..."
                    # until max_tokens cut the JSON mid-string, so every parse
                    # raised "Unterminated string" and the answer was never usable.
                    # A 3-D coordinate needs ~60 characters at full float precision.
                    'position': {'type': ['string', 'null'], 'maxLength': 80},
                    'orientation': {'type': ['string', 'null'], 'maxLength': 40},
                    'time': {'type': ['number', 'string', 'null']},
                    'duration': {'type': ['number', 'string', 'null']},
                },
                # AgentOutput requires all of these (no defaults), so force the model
                # to emit every key — otherwise AgentOutput.from_dict() raises and
                # NonAgent.query retries forever.
                'required': ['type', 'text', 'binary', 'position', 'orientation', 'time', 'duration'],
                'additionalProperties': False,
            }
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
        else:
            self.chain = ChatOllama(model=llm_type, num_ctx=num_ctx, temperature=temperature)
        self.prompt = file_to_string(str(os.path.dirname(__file__)) + '/../' + 'prompts/non_agent_system_prompt.txt')


    def set_memory(self, memory: Memory):
        self.memory = memory
        pass


    def query(self, question: str) -> AgentOutput:



        response_example = """{"reasoning", "-input your reasoning in here for the type of question, then the answer-", 
                                "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
                                "text: "--a text answer here--",
                                "binary: "yes/no",
                                "position: "[x,y,z]",
                                "time: "5.3",
                                }"""

        output_format = f"""{response_example}"""


        # GROUND TRUTH RETRIEVAL
        # Prompt
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'context': self.memory.memory_to_string(self.memory.get_working_memory()), 'question':question, 'output_format':output_format})
        inputs = filled_prompt.text

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

        for _ in range(MAX_PARSE_RETRIES):

            response = self.chain.invoke(inputs)
            response = ''.join(response.content.splitlines())
            try:
                if '```json' not in response:
                    # try parsing on its own since we cannot always trust llms
                    parsed = json.loads(response) 
                else:
                    parsed = parse_json(response)

                # then check it has all the required keys
                keys_to_check_for = ["time", "text", "binary", "position"]
                for key in keys_to_check_for:
                    if key not in parsed:
                        raise ValueError("Missing all the required keys during generate. Retrying...")
                response = AgentOutput.from_dict(parsed)

            except Exception as e:
                last_error = e
                print(response)
                print(e)
                print("Generate call failed. Retrying...")
                continue

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