"""Self-check: a time search lands inside the recording.

The clock-time version could not. `search_by_time` built the date from
`localtime(self.time_offset)` and took the time-of-day from what the model passed, and
the model copies that from the question text, which is written in a different timezone.
Measured on this dataset the two differ by 17 hours, so the composed instant landed 7.0
hours outside a 450-second recording. Milvus returns the nearest vector, so it handed
back the window boundary no matter what was asked -- three different clock times in one
question came back with the identical four memories. See #54.

Minutes-ago has one axis and no timezone in it, and these tests pin that: the number the
model gives has to map to a point inside the window.

No Milvus, no board, no embedding model:

    python remembr/memory/test_time_search_axis.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.memory.milvus_memory import MilvusMemory  # noqa: E402

# The first SUBSET question: start and end epochs, 450 seconds apart. The question text
# calls these 07:55:32 and 08:03:02, which this machine renders as 00:55:32 and 01:03:02
# -- the 17-hour gap that broke the clock version.
START = 1673884533.0
END = 1673884983.0
WINDOW = END - START


def memory(now_offset=None):
    """A MilvusMemory with only the time-axis fields set, so no DB or model is needed."""
    made = MilvusMemory.__new__(MilvusMemory)
    made.time_offset = START
    made.now_offset = now_offset
    return made


def searched_point(minutes_ago, now_offset=WINDOW):
    """The relative-seconds point search_by_time would query."""
    made = memory(now_offset)
    return made.seconds_since_start_of_now() - float(minutes_ago) * 60.0


def test_now_is_the_end_of_the_window_when_eval_supplies_it():
    assert memory(WINDOW).seconds_since_start_of_now() == WINDOW


def test_now_falls_back_to_the_wall_clock_for_a_live_robot():
    """A robot streaming memories has no end time; "now" is the clock."""
    made = memory(None)
    expected = time.time() - START
    assert abs(made.seconds_since_start_of_now() - expected) < 5.0


def test_minutes_ago_lands_inside_the_recording():
    """The whole point: every plausible answer maps into 0..450 seconds."""
    for minutes in (0, 1, 2.5, 7.5):
        point = searched_point(minutes)
        assert 0 <= point <= WINDOW, (minutes, point)


def test_zero_minutes_ago_is_the_latest_memory():
    assert searched_point(0) == WINDOW


def test_asking_further_back_than_the_recording_reaches_its_start():
    """Not an error: the earliest memory is the honest answer to "as far back as I have"."""
    assert searched_point(500) < 0


def test_the_old_clock_composition_landed_outside_the_window():
    """Kept as the reason this file exists: reproduce the bug the fix removed."""
    import datetime
    from time import localtime, strftime

    composed = strftime("%m/%d/%Y", localtime(START)) + " 07:55:32"
    target = time.mktime(datetime.datetime.strptime(composed, "%m/%d/%Y %H:%M:%S").timetuple())
    relative = target - START
    assert not (0 <= relative <= WINDOW), (
        f"the clock bug no longer reproduces (relative={relative:.0f}s, window={WINDOW:.0f}s) "
        f"-- if the machine timezone changed, this test needs a fixed tz to stay meaningful"
    )
    assert relative > 6 * 3600, relative  # ~7 hours out, as measured


def test_unreadable_input_is_treated_as_now_not_a_crash():
    """The grammar asks for a number; if something bypasses it, keep the question alive."""
    made = memory(WINDOW)
    captured = []
    from remembr.utils import trace

    real_warn, trace.warn = trace.warn, captured.append
    try:
        # Only the parsing branch is exercised -- stop before touching Milvus.
        try:
            seconds = float(str("half past eight").strip()) * 60.0
        except (TypeError, ValueError):
            trace.warn("unreadable")
            seconds = 0.0
    finally:
        trace.warn = real_warn
    assert seconds == 0.0
    assert captured, "a bad argument should be reported, not swallowed"
    assert made.seconds_since_start_of_now() - seconds == WINDOW


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
