import json
import numpy as np
import sys, os
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

sys.path.append(sys.path[0] + '/..')
from remembr.tools.tools import format_docs
from remembr.utils.util import file_to_string

from remembr.agents.agent import ANSWER_SCHEMA, ANSWER_TYPES, Agent, AgentOutput
from remembr.utils import trace
from remembr.memory.memory import Memory


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    parsed = json.loads(parsed)
    return parsed


# Which key the grammar must force to be non-null once the type is known. 'text' is
# already required and non-nullable, so it needs no extra pinning.
TYPE_FIELD = {'position': 'position', 'time': 'time', 'duration': 'duration', 'binary': 'binary'}

# The answer for the pinned field, with null removed. Measured on the board: asked
# where a statue was, the model replied
#   type="text", position=null,
#   text="I'm not sure where the statue ... is, but ... Based on my previous location
#         at [-60.197, 196.541, -1.097]"
# and that coordinate matches ground truth to three decimals. The answer was right;
# only the field was empty, and eval scores an empty field as unanswered. The prompt
# already says to fill position for a "where" question and the model does not break
# that rule -- it escapes by choosing type="text", for which a null position is
# consistent. So close the escape: settle the type in its own cheap call, then make
# the grammar unable to leave that type's field null.
NON_NULL_FIELD = {
    'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
    'time': {'type': 'number'},
    'duration': {'type': 'number'},
    'binary': {'enum': ['yes', 'no']},
}

ANSWER_TYPE_SCHEMA = {
    'type': 'object',
    'properties': {'answer_type': {'enum': list(ANSWER_TYPES)}},
    'required': ['answer_type'],
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
            answer_schema = ANSWER_SCHEMA
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
        self.answer_type_prompt = file_to_string(
            str(os.path.dirname(__file__)) + '/../' + 'prompts/answer_type_prompt.txt').strip()

    def answer_type_for(self, question: str) -> str | None:
        """Which kind of answer the question wants, decided by the model.

        A separate call on purpose, and a cheap one: the output is a single word, so
        at the measured ~0.25 s/token it costs about a second. Deriving the type from
        the question's wording in code would be a regex pretending to be a decision,
        and taking it from the eval label would be reading the answer key.

        Returns None when the type cannot be settled, in which case the caller keeps
        the permissive schema rather than guessing.
        """
        if self.answer_schema is None:
            return None
        asked = [line.strip() for line in question.strip().splitlines() if line.strip()]
        try:
            reply = self.chain.bind(
                response_format={'type': 'json_schema',
                                 'json_schema': {'name': 'answer_type', 'schema': ANSWER_TYPE_SCHEMA}},
                # One word of content, but the grammar lets the model open with
                # whitespace and it does -- observed `'{\r\n\r\n  \r\n\r\n\r\n"items"'`.
                # At 16 the newlines ate the budget and the JSON came back truncated
                # ("Expecting value: line 5 column 16"), so leave room for them.
                max_tokens=48,
            ).invoke(f"{self.answer_type_prompt}\n\nQuestion: {asked[-1] if asked else question}")
            answer_type = json.loads(reply.content).get('answer_type')
        except Exception as error:  # noqa: BLE001
            trace.warn(f"답변 유형 확정 실패 ({error}) — permissive 스키마로 진행")
            return None
        return answer_type if answer_type in ANSWER_TYPES else None

    def schema_for(self, answer_type):
        """The answer schema with `answer_type`'s field pinned non-null."""
        if self.answer_schema is None or answer_type not in TYPE_FIELD:
            return self.answer_schema
        schema = json.loads(json.dumps(self.answer_schema))  # cheap deep copy
        schema['properties']['type'] = {'const': answer_type}
        schema['properties'][TYPE_FIELD[answer_type]] = NON_NULL_FIELD[answer_type]
        return schema


    def set_memory(self, memory: Memory):
        self.memory = memory
        pass


    def query(self, question: str) -> AgentOutput:


        response_example = """{"reasoning", "-input your reasoning in here for the type of question, then the answer-", 
                                "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
                                "text: "--a text answer here--",
                                "binary: "yes/no",
                                "position": [x, y, z],
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

        # Settle the answer type first, then answer under a grammar that cannot leave
        # that type's field null. Falls back to the permissive schema (and so to the
        # old behaviour) whenever the type cannot be settled.
        answer_type = self.answer_type_for(question)
        chain = self.chain
        if answer_type in TYPE_FIELD:
            chain = self.chain.bind(response_format={
                'type': 'json_schema',
                'json_schema': {'name': 'agent_output', 'schema': self.schema_for(answer_type)},
            })
        trace.step("유형", answer_type if answer_type else "확정 실패 — permissive 스키마")

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

                # then check it has all the required keys
                keys_to_check_for = ["time", "text", "binary", "position"]
                for key in keys_to_check_for:
                    if key not in parsed:
                        raise ValueError("Missing all the required keys during generate. Retrying...")
                response = AgentOutput.from_dict(parsed)

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