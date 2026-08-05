from dataclasses import dataclass, fields
import inspect

@dataclass
class AgentOutput:
    type: str
    text: str
    binary: str
    position: list
    orientation: float
    duration: float
    time: float

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })



# The five answer categories, defined by what eval.evaluate_output branches on. The
# grammar lists them so anything else is ungenerable: the tt/ prompt used to say only
# "type: the answer category" without naming them, and the model duly invented
# "image" and "yes/no" (the latter lifted from the `binary` line right below it).
ANSWER_TYPES = ('position', 'time', 'duration', 'binary', 'text')

# The grammar the server enforces for an answer, in one place because it is the same
# contract as AgentOutput above -- two spellings of one thing, so they live together.
# They drifted twice while spelled separately in each agent: `position` was a loose
# string in NonAgent long after ReMEmbRAgent constrained it to three numbers, and
# `orientation` was a string here, a [yaw] array there, and "[euler_rotation]" in the
# prompts, for a field AgentOutput declares as a float.
#
# Two rules to keep when editing:
#  1. Never write a nullable field as `{'type': ['x', 'null'], ...}`. lm-format-enforcer
#     silently drops the sibling keywords on a type union -- measured with the real
#     tokenizer, a maxLength of 40 let 60+ characters through in that form and stopped at
#     exactly 40 as `anyOf`. Use `anyOf`.
#  2. Every string branch needs a maxLength, or the model runs to max_tokens inside it and
#     truncates the JSON, which throws away the whole answer.
# `test_schema_constraints.py` checks both, and that this covers exactly AgentOutput.
ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'enum': list(ANSWER_TYPES)},
        'text': {'type': 'string', 'maxLength': 300},
        # eval compares this case-insensitively against "yes"/"no", so an enum makes
        # anything else ungenerable rather than merely short.
        'binary': {'anyOf': [{'enum': ['yes', 'no']}, {'type': 'null'}]},
        # Three numbers, because a coordinate is three numbers. As a loose string this
        # field produced shapes that were scored as if they were answers: a run-on list
        # of coordinates that truncated mid-string, and `[[-22.67, 37.41, -1.5]]`, which
        # eval()s fine and then scores against a 3-vector by numpy broadcasting.
        'position': {
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
                {'type': 'null'},
            ]
        },
        # A number: AgentOutput declares `orientation: float`, MemoryItem carries
        # `theta: float`, and the memory string the model reads says "with an average
        # orientation of {theta} radians". Saying "number" removes the runaway rather
        # than capping it -- number grammar is finite, so no maxLength is needed.
        'orientation': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        # Numbers only. The string branch was there because upstream tolerates "5.3"
        # and eval()s it, but the model wrote prose into it -- observed
        # `"time": "11 minutes ago"`, which eval() cannot parse, so the answer scored
        # as unanswered even though the model knew it was 11 minutes. Minutes as a
        # float is what AgentOutput declares and what eval compares against.
        'time': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'duration': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
    },
    # AgentOutput takes no defaults, so every key must be present or from_dict() raises
    # and the caller retries.
    'required': [field.name for field in fields(AgentOutput)],
    'additionalProperties': False,
}


# Which key the grammar must force non-null once the answer type is settled. 'text' is
# already required and non-nullable, so it needs no extra pinning.
TYPE_FIELD = {'position': 'position', 'time': 'time', 'duration': 'duration',
              'binary': 'binary'}

# That field with null removed. Measured on the board: asked where a statue was, the
# model replied type="text", position=null, and wrote the correct coordinate into the
# text field to three decimals. The answer was right; the field was empty, and eval
# scores an empty field as unanswered. The prompt already says to fill position for a
# "where" question and the model does not break that rule -- it escapes by choosing
# type="text", for which a null position is consistent. So settle the type in its own
# cheap call, then make the grammar unable to leave that type's field null.
NON_NULL_FIELD = {
    'position': {'type': 'array', 'items': {'type': 'number'},
                 'minItems': 3, 'maxItems': 3},
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

# One word of content, but the grammar lets the model open with whitespace and it does
# -- observed `'{\r\n\r\n  \r\n\r\n\r\n"items"'`. At 16 the newlines ate the budget and
# the JSON came back truncated ("Expecting value: line 5 column 16").
ANSWER_TYPE_MAX_TOKENS = 48


def asked_line(question: str) -> str:
    """The question without its preamble.

    The preamble states the current time and position, and sending it made the model
    search for those instead of what was asked.
    """
    lines = [line.strip() for line in question.strip().splitlines() if line.strip()]
    return lines[-1] if lines else question.strip()


def settle_answer_type(chain, question: str, instructions: str):
    """Which kind of answer the question wants, decided by the model.

    A separate call on purpose, and a cheap one: the output is a single word, so at the
    measured ~0.25 s/token it costs about a second. Deriving the type from the question's
    wording in code would be a regex pretending to be a decision, and taking it from the
    eval label would be reading the answer key.

    Takes the chain rather than living on one agent, because both answer paths need it:
    it was on NonAgent only, and ReMEmbRAgent without it produced a `position` answer for
    a time question, scored unanswered, and burned all five eval attempts repeating the
    whole 3-turn tool loop each time.

    Returns None when the type cannot be settled, so the caller keeps the permissive
    schema rather than guessing.
    """
    import json as _json

    try:
        reply = chain.bind(
            response_format={'type': 'json_schema',
                             'json_schema': {'name': 'answer_type',
                                             'schema': ANSWER_TYPE_SCHEMA}},
            max_tokens=ANSWER_TYPE_MAX_TOKENS,
        ).invoke(f"{instructions}\n\nQuestion: {asked_line(question)}")
        settled = _json.loads(reply.content).get('answer_type')
    except Exception as error:  # noqa: BLE001
        from remembr.utils import trace

        trace.warn(f"답변 유형 확정 실패 ({error}) — permissive 스키마로 진행")
        return None
    return settled if settled in ANSWER_TYPES else None


def schema_for_type(answer_type):
    """ANSWER_SCHEMA with `answer_type`'s field pinned non-null and `type` fixed."""
    import json as _json

    if answer_type not in TYPE_FIELD:
        return ANSWER_SCHEMA
    schema = _json.loads(_json.dumps(ANSWER_SCHEMA))  # cheap deep copy
    schema['properties']['type'] = {'const': answer_type}
    schema['properties'][TYPE_FIELD[answer_type]] = NON_NULL_FIELD[answer_type]
    return schema


# The fields worth reading in a trace line, in the order a human scans them.
_SUMMARY_FIELDS = ('type', 'position', 'time', 'duration', 'binary')


def answer_summary(answer) -> str:
    """The filled fields of an answer, for one trace line.

    Takes an AgentOutput or the parsed dict, because the two answer paths hold it
    differently at the point they log it -- and each had written its own version of this.
    """
    def value(name):
        return answer.get(name) if isinstance(answer, dict) else getattr(answer, name, None)

    filled = [(name, value(name)) for name in _SUMMARY_FIELDS]
    return "  ".join(f"{name}={found!r}" for name, found in filled if found is not None)


# Both answer paths validate the same way. They did not: NonAgent and VLMNonAgent checked
# four keys, ReMEmbRAgent checked five (it included `duration`), and only ReMEmbRAgent
# checked that a position has three numbers. So the same malformed answer passed on one
# path and failed on another.
def validate_answer(parsed):
    """Raise ValueError if `parsed` is not a scoreable answer."""
    missing = [name for name in ANSWER_SCHEMA['required'] if name not in parsed]
    if missing:
        raise ValueError(f"answer is missing required keys: {missing}")
    position = parsed.get('position')
    if position is not None and len(position) != 3:
        raise ValueError(f"position must be three numbers, got {position!r}")


class Agent:
    def query(self, query: str) -> AgentOutput:
        raise NotImplementedError

    def query_position(self, query: str) -> list:
        return self.query(query).position
    
    def query_duration(self, query: str) -> float:
        return self.query(query).duration

    def query_time(self, query: str) -> float:
        return self.query(query).time
    
    def query_yes_no(self, query: str) -> bool:
        str_bool = self.query(query).binary
        if str_bool.lower == 'yes':
            return True
        return False
