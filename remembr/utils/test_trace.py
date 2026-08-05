"""Self-check for the pipeline trace.

Two things matter. It must print nothing when off, so normal runs are unaffected and
nobody has to remember to turn it back off. And `warn` must print even when off,
because the lines it carries -- a parse failure, an empty search query -- are worth
seeing in every run and used to be bare prints scattered through the agents.

    python remembr/utils/test_trace.py
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


def test_a_block_prints_as_one_call():
    """ToolNode runs a turn's tool calls concurrently, so a header plus its detail
    lines must not be separate prints -- two threads interleaved mid-line before."""
    calls = []
    was = trace.ENABLED
    trace.ENABLED = True
    real = print
    try:
        import builtins

        builtins.print = lambda *a, **k: calls.append(a[0] if a else "")
        trace.step("검색", "time('x')  2개", arrow="←", extra=["1 ...", "2 ..."])
    finally:
        builtins.print = real
        trace.ENABLED = was
    assert len(calls) == 1, f"expected one print, got {len(calls)}"
    assert calls[0].count("\n") == 2, calls[0]


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


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
