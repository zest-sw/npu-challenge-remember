import os
import base64
from io import BytesIO

import requests
from PIL import Image

# Support both launch styles: the smoke test runs with the outer repo root on
# PYTHONPATH (so `remembr.*` resolves), while the eval/preprocess scripts add the
# inner `remembr/` dir to sys.path and import as `captioners.*`.
try:
    from remembr.captioners.captioner import Captioner
except ModuleNotFoundError:
    from captioners.captioner import Captioner


# NOTE: the original ReMEmbR prompt began with a literal "<video>" token — that
# is a VILA-ism. Qwen2.5-VL does not use it; passing it verbatim alongside more
# than one image confuses the model (it refuses with "I can't assist with that").
# So we drop the token and phrase the prompt for one-or-more still frames.
DEFAULT_PROMPT = (
    "These are sequential frames from a robot's camera. "
    "Describe in detail what is visible: the people, objects, environmental "
    "features, and any events or activities. Be specific and concise."
)


# Frames went to the server at the camera's full 1224x1024, which Qwen's processor turns
# into ~1200 image tokens -- over TT_VLM_MAX_SEQ_LEN (1024) before the text prompt is even
# counted. Every caption request had been decoding past the KV cache, which is undefined
# and eventually spins a thread inside ttnn and wedges the board. Measured prefill for the
# terse prompt against the long side of the frame:
#
#     1224 -> 1429    896 -> 857    768 -> 693    640 -> 553    512 -> 421
#
# 768 leaves ~200 tokens of headroom at max_gen_len 128. Raise it (and TT_VLM_MAX_SEQ_LEN
# with it) if the captions turn out to be missing text on signs -- the server now rejects
# an over-budget request with a clear message instead of hanging, so this is safe to tune.
MAX_IMAGE_LONG_SIDE = int(os.environ.get("TT_VLM_IMAGE_LONG_SIDE", "768"))


def _encode_image(image: Image.Image) -> str:
    """PIL image -> base64-encoded JPEG string (RGB), downscaled to fit the token budget."""
    image = image.convert("RGB")
    if MAX_IMAGE_LONG_SIDE and max(image.size) > MAX_IMAGE_LONG_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_LONG_SIDE, MAX_IMAGE_LONG_SIDE), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TenstorrentCaptioner(Captioner):
    """Captioner backed by a Qwen2.5-VL model running on a Tenstorrent NPU.

    Instead of loading the VLM in-process (which would require the tt-metal /
    ttnn stack and conflicts with remembr's dependency pins), this talks to the
    local unified inference server (``tt_infer_server.py``) over HTTP.
    """

    def __init__(self, url: str = None, prompt: str = DEFAULT_PROMPT, timeout: float = None,
                 max_gen_len: int = None):
        self.url = url or os.environ.get("TT_VLM_URL", "http://localhost:8100/caption")
        self.prompt = prompt
        # First caption after a fresh server launch compiles ttnn traces
        # (prefill+decode) — a one-time cost that can take several minutes on a
        # cold co-resident VLM. Default the timeout high enough to survive it;
        # override with TT_VLM_TIMEOUT (seconds).
        if timeout is None:
            timeout = float(os.environ.get("TT_VLM_TIMEOUT", "900"))
        self.timeout = timeout
        # Max tokens the VLM may generate per caption. The server defaults to 128,
        # which truncates the (verbose, step-by-step) campus captions mid-sentence.
        # Bump it so captions finish; the model still stops early on EOT, so this is
        # only an upper bound. Must satisfy prefill+max_gen_len <= TT_VLM_MAX_SEQ_LEN.
        if max_gen_len is None:
            max_gen_len = int(os.environ.get("TT_VLM_MAX_GEN_LEN", "256"))
        self.max_gen_len = max_gen_len

    def caption(self, images: list[Image.Image]) -> str:
        payload = {
            "images": [_encode_image(im) for im in images],
            "prompt": self.prompt,
            "max_gen_len": self.max_gen_len,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["caption"]
