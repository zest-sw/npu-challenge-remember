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



# Naming them in the grammar makes anything else ungenerable. Unnamed, the model
# invented "image" and "yes/no".
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
        # Numbers only. The string branch upstream tolerates invited prose --
        # `"time": "11 minutes ago"` scored as unanswered because eval() cannot parse it.
        'time': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'duration': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
    },
    # AgentOutput takes no defaults, so every key must be present or from_dict() raises
    # and the caller retries.
    'required': [field.name for field in fields(AgentOutput)],
    'additionalProperties': False,
}


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
