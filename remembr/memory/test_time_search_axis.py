"""Self-check: a clock time from the question maps inside the recording.

The bug it guards (#54): `search_by_time` built the date from `localtime(time_offset)`
and took the time-of-day from its argument. The model copies that argument from the
question text, which is written in the dataset's timezone -- 17 hours from this machine's.
So the composed instant landed 7.0 hours outside a 450-second recording, and Milvus
returned the window boundary for every query. Three different clock times in one question
came back with the identical four memories.

The fix keeps the clock interface -- the questions state times that way, so the model
copies rather than computes -- and anchors it: the question's stated current time paired
with the epoch it names. Both the date and the time-of-day then come from the same frame
and the offset cancels.

(Minutes-ago was tried instead and was worse: the model had to do the arithmetic and
asked for 27, 48 and 40 minutes back on a 7.5-minute recording.)

No Milvus, no board, no embedding model:

    python remembr/memory/test_time_search_axis.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.memory.milvus_memory import MilvusMemory  # noqa: E402

# The first SUBSET question. The text calls these 07:55:32 and 08:03:02; this machine
# renders the same epochs as 00:55:32 and 01:03:02 -- the 17-hour gap.
START = 1673884533.0
END = 1673884983.0
WINDOW = END - START
STATED_NOW = "2023-01-16 08:03:02"
STATED_START = "07:55:32"


def memory(anchored=True):
    """Only the time-axis fields, so no DB or embedding model is needed."""
    made = MilvusMemory.__new__(MilvusMemory)
    made.time_offset = START
    made.now_clock = STATED_NOW if anchored else None
    made.now_epoch = END if anchored else None
    return made


def test_the_anchor_measures_the_timezone_gap():
    assert abs(memory()._timezone_shift() / 3600 + 17.0) < 0.01


def test_the_questions_own_times_map_to_the_windows_ends():
    made = memory()
    assert abs(made._clock_to_relative_seconds(STATED_START) - 0.0) < 1.0
    assert abs(made._clock_to_relative_seconds("08:03:02") - WINDOW) < 1.0


def test_any_clock_time_the_question_states_lands_inside():
    """The whole point. Every time between start and now maps into 0..450 seconds."""
    made = memory()
    for asked in ("07:55:32", "07:57:00", "07:59:00", "08:01:30", "08:03:02"):
        point = made._clock_to_relative_seconds(asked)
        assert 0 <= point <= WINDOW + 1, (asked, point)


def test_a_full_date_is_accepted_too():
    """The prompt tells the model HH:MM:SS, but it sometimes sends the whole thing."""
    made = memory()
    both = (made._clock_to_relative_seconds("07:59:00"),
            made._clock_to_relative_seconds("01/16/2023 07:59:00"))
    assert abs(both[0] - both[1]) < 1.0, both


def test_the_old_localtime_composition_landed_outside_the_window():
    """Reproduce the bug, so this file says why it exists. Without the anchor the date
    comes from localtime and the result leaves the recording."""
    import datetime
    import time
    from time import localtime, strftime

    composed = strftime("%m/%d/%Y", localtime(START)) + " " + STATED_START
    machine = time.mktime(datetime.datetime.strptime(composed, "%m/%d/%Y %H:%M:%S").timetuple())
    relative = machine - START
    assert not (0 <= relative <= WINDOW), (
        f"the bug no longer reproduces (relative={relative:.0f}s, window={WINDOW:.0f}s) -- "
        f"if the machine timezone changed, this test needs a fixed tz to stay meaningful"
    )
    assert relative > 6 * 3600, relative  # ~7 hours out, as measured


def test_a_live_robot_needs_no_anchor():
    """There the machine clock and the clock the robot talks about are the same."""
    made = memory(anchored=False)
    assert made._timezone_shift() == 0.0
    # It still converts, using localtime for the date as the original code did.
    assert isinstance(made._clock_to_relative_seconds("07:59:00"), float)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
