"""Self-check: how big a caption frame may be, and what happens when it is too big.

Two rules in the domain layer so they test without fastapi, torch, ttnn or a board:
`cap_long_side` (the HTTP layer used to squash every frame to 224x224, #59) and
`check_generation_fits` (decoding past the KV cache writes into whatever DRAM follows).

The second was written believing prefill was 1279-1429 tokens; it was 79-277, because
the server squashed the image first. Nothing had exceeded the cache, and what wedged
the board is still unknown. It is kept because lifting the squash makes prefill 1429,
where the limit becomes real -- the numbers below are budget arithmetic, not a
diagnosed failure.

    python tt_server/tests/test_caption_token_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tt_server.domain.errors import InvalidRequest  # noqa: E402
from tt_server.domain.models import (  # noqa: E402
    cap_long_side,
    check_generation_fits,
)


def read(relative):
    return open(os.path.join(os.path.dirname(__file__), '..', '..', relative)).read()


def read_app():
    return read('tt_server/presentation/http/app.py')




class engine:
    """The rule lives in the domain layer, so this needs no torch, ttnn or device."""

    def __init__(self, max_seq_len=1024):
        self.max_seq_len = max_seq_len

    def _check_fits(self, prefill_len, max_gen_len):
        check_generation_fits(prefill_len, max_gen_len, self.max_seq_len)


def test_a_request_that_fits_is_allowed():
    engine()._check_fits(693, 128)          # 768px frame + terse prompt = 821
    engine()._check_fits(1024 - 128, 128)   # exactly the limit


def test_a_full_resolution_request_is_refused_when_the_cache_is_small():
    """1224x1024 frame + terse prompt = 1429 prefill, which needs max_seq_len 2048."""
    try:
        engine()._check_fits(1429, 128)
    except InvalidRequest as error:
        assert '1557' in str(error), error
        assert 'max_seq_len 1024' in str(error), error
    else:
        raise AssertionError("1429 + 128 over a 1024 cache should be refused")


def test_the_paper_prompt_at_full_resolution_is_refused_too():
    """Same arithmetic with the original prompt and the old max_gen_len of 256."""
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


def test_the_server_no_longer_squashes_every_frame_to_224():
    """The line that made resolution untestable, and captions unable to read a sign."""
    import ast

    source = read_app()
    tree = ast.parse(source)
    decode = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_decode_image')
    body = ast.unparse(decode.body[1:])          # [0] is the docstring, which quotes it
    assert 'resize' not in body, body
    assert 'cap_long_side' in body, body


def test_the_server_caps_the_long_side_and_keeps_the_aspect_ratio():
    from PIL import Image

    sent = cap_long_side(Image.new('RGB', (1224, 1024)), 768)
    assert max(sent.size) == 768, sent.size
    assert abs(sent.size[0] / sent.size[1] - 1224 / 1024) < 0.01, sent.size


def test_a_frame_already_within_the_cap_is_untouched():
    from PIL import Image

    assert cap_long_side(Image.new('RGB', (320, 240)), 768).size == (320, 240)


def test_zero_disables_downscaling():
    from PIL import Image

    assert cap_long_side(Image.new('RGB', (1224, 1024)), 0).size == (1224, 1024)


def test_the_captioner_does_not_resize_behind_the_server():
    """Two places deciding resolution is how the 224 squash stayed invisible."""
    captioner = read('remembr/captioners/tenstorrent_captioner.py')
    assert 'thumbnail' not in captioner, captioner


def test_the_engine_can_warm_the_real_frame_shape():
    """tt-metal warms 224x224 only. Any other shape compiles on its first request, which
    looks identical to a hang -- minutes of silence, no /health, no Ctrl-C (#62). The
    engine must be able to pay that at load instead."""
    source = read('tt_server/infrastructure/tenstorrent/qwen_vl_engine.py')
    assert 'TT_VLM_WARMUP_SIZE' in source, source[:400]
    load = source.split('def load')[1].split('\n    def ')[0]
    assert '_warm_real_shape' in load, load


def test_the_warmup_runs_after_the_backend_warmup_not_before():
    """It calls describe(), which needs generator, processor and caches in place."""
    source = read('tt_server/infrastructure/tenstorrent/qwen_vl_engine.py')
    load = source.split('def load')[1].split('\n    def ')[0]
    assert load.index('warmup_model') < load.index('_warm_real_shape'), load


def test_a_failed_warmup_does_not_stop_the_server():
    """The shape may simply not fit; the request-time guard says so clearly. Refusing to
    serve over a failed warmup would turn a bad setting into an outage."""
    source = read('tt_server/infrastructure/tenstorrent/qwen_vl_engine.py')
    warm = source.split('def _warm_real_shape')[1]
    assert 'except Exception' in warm and 'serving anyway' in warm, warm


def test_the_vlm_decode_does_not_use_device_tracing():
    """tt-metal caches the decode trace on `hasattr(self, "trace_ids_text")` alone, so it
    outlives the request that captured it, and later prefills allocate device buffers
    beside it -- tt-metal warns those buffers may be corrupted. Same class as #21."""
    source = read('tt_server/infrastructure/tenstorrent/qwen_vl_engine.py')
    describe = source.split('def describe')[1]
    assert 'enable_trace=False' in describe, describe[-900:]
    assert 'enable_trace=True' not in describe, describe[-900:]


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
            print(f"ok  {name}")
    print("all passed")
