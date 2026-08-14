"""Self-check for the direct+ pipeline: question -> search -> a bare [x, y, z].

No NPU, no Milvus, no server:

    python remembr/agents/test_direct_position_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import AgentOutput  # noqa: E402
from remembr.agents.direct_position_agent import (  # noqa: E402
    POSITION_SCHEMA,
    position_output,
)


def test_schema_is_exactly_three_numbers():
    """The grammar is the whole contract: three numbers, nothing else generable."""
    assert POSITION_SCHEMA == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }, POSITION_SCHEMA


def test_a_bare_array_becomes_a_position_answer():
    out = position_output([1, 2.5, -3])
    assert isinstance(out, AgentOutput)
    assert out.type == "position"
    assert out.position == [1.0, 2.5, -3.0]
    assert out.time is None and out.duration is None and out.binary is None


def test_wrong_shapes_are_rejected():
    # Shapes the old loose-string field let through and eval then mis-scored.
    for bad in ([1, 2], [[1, 2, 3]], "at [1,2,3]", [1, 2, "a"], [True, 1, 2], None):
        try:
            position_output(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
