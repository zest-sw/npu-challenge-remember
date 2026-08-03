"""Self-check for the space/time calculators in tools.py.

These replace arithmetic the LLM used to do, so a silent regression here shows up
as eval error rather than a crash. Run directly; no NPU, no Milvus, no server:

    python remembr/tools/test_spacetime_calc.py
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.tools.tools import (  # noqa: E402
    distance,
    duration_minutes,
    minutes_ago,
    nearest,
    parse_timestamp,
)

# The real preamble format, from form_question_jsons.py:214.
QUESTION = (
    "You started moving at 2023-01-16 07:50:00. The current time is "
    "2023-01-16 08:03:02 and you are located at [10.0, 0.0, 0.0]. \n "
    "Where is the closest place to sit?"
)

# The real memory rendering, from milvus_memory.memory_to_string().
CONTEXT = (
    "At time=2023-01-16 07:55:32, the robot was at an average position of "
    "[100.0, 0.0, 0.0].The robot saw the following: a bench near the fountain\n\n"
    "At time=2023-01-16 07:58:10, the robot was at an average position of "
    "[12.0, 0.0, 0.0].The robot saw the following: a wooden bench by the door\n\n"
)


def test_parse_timestamp():
    assert parse_timestamp("2023-01-16 08:03:02") == datetime.datetime(2023, 1, 16, 8, 3, 2)
    # A bare clock time borrows the reference date (captions sometimes echo only HMS).
    ref = datetime.datetime(2023, 1, 16, 8, 3, 2)
    assert parse_timestamp("07:55:32", reference=ref) == datetime.datetime(2023, 1, 16, 7, 55, 32)
    # Bare time with no reference is not guessable -> None, not a wrong date.
    assert parse_timestamp("07:55:32") is None
    assert parse_timestamp("sometime this morning") is None
    assert parse_timestamp(None) is None


def test_minutes_ago():
    assert minutes_ago("2023-01-16 07:55:32", "2023-01-16 08:03:02") == 7.5
    # Future events are not a valid minutes-ago answer; must not return negative.
    assert minutes_ago("2023-01-16 09:00:00", "2023-01-16 08:03:02") is None
    assert minutes_ago("garbage", "2023-01-16 08:03:02") is None


def test_duration_minutes():
    assert duration_minutes("2023-01-16 07:55:00", "2023-01-16 07:57:30") == 2.5
    # Order-independent: the model reports the pair backwards routinely.
    assert duration_minutes("2023-01-16 07:57:30", "2023-01-16 07:55:00") == 2.5
    assert duration_minutes("2023-01-16 07:55:00", None) is None


def test_distance_and_nearest():
    assert distance([0, 0, 0], [3, 4, 0]) == 5.0
    # z matters: the benchmark asks about elevators/stairs across floors.
    assert distance([0, 0, 0], [0, 0, 7]) == 7.0
    assert distance("[0, 0, 0]", "(3, 4, 0)") == 5.0
    assert distance("nope", [0, 0, 0]) is None

    index, coordinate, d = nearest([[100, 0, 0], [12, 0, 0]], [10, 0, 0])
    assert (index, coordinate, d) == (1, [12.0, 0.0, 0.0], 2.0)
    # A JSON string of candidates behaves the same.
    assert nearest("[[100,0,0],[12,0,0]]", [10, 0, 0])[0] == 1
    # A single flat coordinate is treated as one candidate, not three.
    assert nearest([12, 0, 0], [10, 0, 0])[1] == [12.0, 0.0, 0.0]
    assert nearest([], [10, 0, 0]) == (None, None, None)


def test_calculator_takes_every_operand_explicitly():
    """The tools are pure: "now" / "origin" are arguments, never hidden state.

    An earlier version regex-parsed them out of the question prose and stashed
    them on the agent, which hid an operand from the model. These asserts pin the
    tool bodies' behaviour so that cannot come back.
    """
    # compute_minutes_ago(event_time, now)
    assert minutes_ago("2023-01-16 07:55:32", "2023-01-16 08:03:02") == 7.5
    # compute_duration_minutes(start_time, end_time)
    assert duration_minutes("2023-01-16 07:55:00", "2023-01-16 07:57:30") == 2.5
    # find_nearest_position(candidates, origin) -- both as strings, as the tool gets them
    assert nearest("[[100,0,0],[12,0,0]]", "[10,0,0]")[1] == [12.0, 0.0, 0.0]


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
