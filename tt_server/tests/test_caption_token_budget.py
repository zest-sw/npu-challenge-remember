"""Self-check: a caption request that would not fit the KV cache is refused, not run.

The board wedged partway through a caption pass. What it looked like:

    Running Vision Blocks: 100%|...| 27/27
    INFO: "POST /caption HTTP/1.1" 200 OK
    Running Vision Blocks:   0%|   | 0/27          <- and there it stayed

    thread A   Rl+  CPU climbing 1s per wall second, inside ttnn
    main       Sl+  futex_wait_queue

Python runs signal handlers on the main thread only, and the main thread was blocked in a
futex, so Ctrl-C was never processed and /health never answered. Killing the process was
the only way out.

The cause was not the disconnect that preceded it. `describe()` sized its token buffer as
`prefill + max_gen_len` and never compared that to `max_seq_len`, and the frames were sent
at the camera's full 1224x1024 -- about 1200 image tokens by themselves:

    TT_VLM_MAX_SEQ_LEN   1024
    paper prompt         prefill 1279 + 256 = 1535
    terse prompt         prefill 1429 + 128 = 1557

So every caption request in the 272-caption pass and both probes decoded past the cache.
That is undefined rather than fatal, which is why it mostly worked: eight captions
succeeded and the ninth spun.

Two changes, and one test each:
  - the engine refuses an over-budget request with InvalidRequest (-> HTTP 400)
  - the captioner downscales frames so requests fit in the first place

No board, no server, no ttnn:

    python tt_server/tests/test_caption_token_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tt_server.domain.errors import InvalidRequest  # noqa: E402
from tt_server.domain.models import check_generation_fits  # noqa: E402


class engine:
    """The rule lives in the domain layer, so this needs no torch, ttnn or device."""

    def __init__(self, max_seq_len=1024):
        self.max_seq_len = max_seq_len

    def _check_fits(self, prefill_len, max_gen_len):
        check_generation_fits(prefill_len, max_gen_len, self.max_seq_len)


def test_a_request_that_fits_is_allowed():
    engine()._check_fits(693, 128)          # 768px frame + terse prompt = 821
    engine()._check_fits(1024 - 128, 128)   # exactly the limit


def test_the_measured_full_resolution_request_is_refused():
    """The one that wedged the board: 1224x1024 frame, terse prompt, max_gen_len 128."""
    try:
        engine()._check_fits(1429, 128)
    except InvalidRequest as error:
        assert '1557' in str(error), error
        assert 'max_seq_len 1024' in str(error), error
    else:
        raise AssertionError("1429 + 128 over a 1024 cache should be refused")


def test_the_paper_prompt_at_full_resolution_is_refused_too():
    """The 272 captions already on disk were produced this way."""
    try:
        engine()._check_fits(1279, 256)
    except InvalidRequest:
        pass
    else:
        raise AssertionError("1279 + 256 over a 1024 cache should be refused")


def test_raising_max_seq_len_lets_full_resolution_through():
    """The other way to fix a rejected request, which the message names."""
    engine(max_seq_len=2048)._check_fits(1429, 128)


def test_the_error_says_how_to_fix_it():
    """A 400 the caller cannot act on is only a politer hang."""
    try:
        engine()._check_fits(1429, 128)
    except InvalidRequest as error:
        text = str(error)
        for hint in ('smaller image', 'max_gen_len', 'TT_VLM_MAX_SEQ_LEN'):
            assert hint in text, (hint, text)


def test_it_is_an_invalid_request_so_the_http_layer_returns_400():
    from tt_server.domain.errors import InferenceError

    assert issubclass(InvalidRequest, InferenceError)


def test_describe_checks_before_it_touches_the_device():
    """The check has to run before prefill_forward, or it prevents nothing."""
    source = open(os.path.join(os.path.dirname(__file__), '..', 'infrastructure',
                               'tenstorrent', 'qwen_vl_engine.py')).read()
    body = source.split('def describe')[1]
    assert body.index('check_generation_fits') < body.index('prefill_forward'), (
        "check_generation_fits must run before the device call it protects"
    )


def test_the_captioner_downscales_frames():
    """Enforcement alone would turn every caption into a 400. The frames have to shrink."""
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'remembr'))
    from captioners import tenstorrent_captioner as captioner

    assert captioner.MAX_IMAGE_LONG_SIDE <= 896, captioner.MAX_IMAGE_LONG_SIDE
    encoded = captioner._encode_image(Image.new('RGB', (1224, 1024)))
    import base64
    from io import BytesIO
    sent = Image.open(BytesIO(base64.b64decode(encoded)))
    assert max(sent.size) == captioner.MAX_IMAGE_LONG_SIDE, sent.size
    # thumbnail() rounds to whole pixels, so compare within a pixel's worth of ratio.
    assert abs(sent.size[0] / sent.size[1] - 1224 / 1024) < 0.01, ('aspect ratio', sent.size)


def test_a_frame_already_small_enough_is_left_alone():
    from PIL import Image

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'remembr'))
    from captioners import tenstorrent_captioner as captioner

    import base64
    from io import BytesIO
    encoded = captioner._encode_image(Image.new('RGB', (320, 240)))
    sent = Image.open(BytesIO(base64.b64decode(encoded)))
    assert sent.size == (320, 240), sent.size


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
            print(f"ok  {name}")
    print("all passed")
