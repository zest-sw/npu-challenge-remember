"""Self-check for the pipeline trace.

Two things matter. It must print nothing when off, so normal runs are unaffected and
nobody has to remember to turn it back off. And `warn` must print even when off,
because the lines it carries -- a parse failure, an empty search query -- are worth
seeing in every run and used to be bare prints scattered through the agents.

    python remembr/tests/test_trace.py
"""
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.utils import trace  # noqa: E402


def captured(enabled, body):
    """Run body with tracing forced on/off; return (stdout, stderr)."""
    was = trace.ENABLED
    trace.ENABLED = enabled
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            body()
    finally:
        trace.ENABLED = was
    return out.getvalue(), err.getvalue()


def test_silent_when_off():
    out, err = captured(False, lambda: (
        trace.question(1, 25, "position", "Where is the bench?"),
        trace.step("검색어", "'bench'"),
        trace.note("something"),
    ))
    assert out == "", out
    assert err == "", err


def test_prints_when_on():
    out, _ = captured(True, lambda: (
        trace.question(3, 25, "position", "Where is the nearest recycling bin?"),
        trace.step("검색어", "'recycling bin'"),
        trace.step("검색", "메모리 4개", arrow="←"),
        trace.step("유형", "position"),
    ))
    assert "[ 3/25 position] Where is the nearest recycling bin?" in out, out
    assert "검색어" in out and "'recycling bin'" in out, out
    # Direction is readable without parsing the words.
    assert "→" in out and "←" in out, out


def test_warn_prints_even_when_off():
    """A parse failure must not be silent just because tracing is off."""
    _, err = captured(False, lambda: trace.warn("파싱 실패"))
    assert "파싱 실패" in err, err


def test_warn_goes_to_stderr_not_stdout():
    """So that piping the trace to a file keeps the failures visible on the terminal."""
    out, err = captured(True, lambda: trace.warn("boom"))
    assert out == "", out
    assert "boom" in err, err


def test_enabled_follows_the_env_var():
    for value, expected in (("1", True), ("0", False), (None, False)):
        before = os.environ.get("REMEMBR_TRACE")
        if value is None:
            os.environ.pop("REMEMBR_TRACE", None)
        else:
            os.environ["REMEMBR_TRACE"] = value
        try:
            import importlib

            reloaded = importlib.reload(trace)
            assert reloaded.ENABLED is expected, (value, reloaded.ENABLED)
        finally:
            if before is None:
                os.environ.pop("REMEMBR_TRACE", None)
            else:
                os.environ["REMEMBR_TRACE"] = before
            importlib.reload(trace)


def test_clip_puts_everything_on_one_line():
    from remembr.utils import trace

    assert trace.clip("여러   공백과\n줄바꿈") == "여러 공백과 줄바꿈"


def test_clip_marks_where_it_cut():
    """A silently cut line reads as a short value rather than a long one."""
    from remembr.utils import trace

    got = trace.clip("x" * 200, 20)
    assert len(got) == 20 and got.endswith("…"), got


def test_clip_leaves_a_short_line_alone():
    from remembr.utils import trace

    assert trace.clip("짧다", 20) == "짧다"


def test_step_and_note_clip_so_callers_do_not_have_to():
    """The point of moving it here: 160, 80 and 64 were all in use at call sites."""
    import inspect

    from remembr.utils import trace

    for function in (trace.step, trace.note):
        assert "clip(" in inspect.getsource(function), function.__name__


def test_nobody_slices_for_display_any_more():
    import os

    root = os.path.join(os.path.dirname(__file__), '..', '..')
    for relative in ('remembr/agents/remembr_agent.py', 'remembr/agents/non_agent.py',
                     'remembr/memory/milvus_memory.py'):
        source = open(os.path.join(root, relative)).read()
        for banned in ('[:160]', '[:80]', '_one_line'):
            assert banned not in source, (relative, banned)


def test_plain_prints_only_when_the_trace_is_off(capsys=None):
    """The two outputs say the same things, so one of them has to yield."""
    import io
    from contextlib import redirect_stdout

    from remembr.utils import trace

    was = trace.ENABLED
    try:
        trace.ENABLED = False
        out = io.StringIO()
        with redirect_stdout(out):
            trace.plain("Question:", "where is the bin")
        assert out.getvalue().strip() == "Question: where is the bin"

        trace.ENABLED = True
        out = io.StringIO()
        with redirect_stdout(out):
            trace.plain("Question:", "where is the bin")
        assert out.getvalue() == ""
    finally:
        trace.ENABLED = was


def test_nobody_writes_the_inverted_flag_check_by_hand():
    """Eight call sites carried `if not trace.ENABLED: print(...)`, two of them nested."""
    import os

    root = os.path.join(os.path.dirname(__file__), '..', '..')
    for relative in ('remembr/scripts/eval.py',):
        source = open(os.path.join(root, relative)).read()
        assert 'if not trace.ENABLED' not in source, relative


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
