import os
import base64
from io import BytesIO

import requests
from PIL import Image

from remembr.captioners.captioner import Captioner


# NOTE: the original ReMEmbR prompt began with a literal "<video>" token — that
# is a VILA-ism. Qwen2.5-VL does not use it; passing it verbatim alongside more
# than one image confuses the model (it refuses with "I can't assist with that").
# So we drop the token and phrase the prompt for one-or-more still frames.
DEFAULT_PROMPT = (
    "These are sequential frames from a robot's camera. "
    "Describe in detail what is visible: the people, objects, environmental "
    "features, and any events or activities. Be specific and concise."
)


def _encode_image(image: Image.Image) -> str:
    """PIL image -> base64-encoded JPEG string (RGB)."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TenstorrentCaptioner(Captioner):
    """Captioner backed by a Qwen2.5-VL model running on a Tenstorrent NPU.

    Instead of loading the VLM in-process (which would require the tt-metal /
    ttnn stack and conflicts with remembr's dependency pins), this talks to the
    local unified inference server (``tt_infer_server.py``) over HTTP.
    """

    def __init__(self, url: str = None, prompt: str = DEFAULT_PROMPT, timeout: float = None):
        self.url = url or os.environ.get("TT_VLM_URL", "http://localhost:8100/caption")
        self.prompt = prompt
        # First caption after a fresh server launch compiles ttnn traces
        # (prefill+decode) — a one-time cost that can take several minutes on a
        # cold co-resident VLM. Default the timeout high enough to survive it;
        # override with TT_VLM_TIMEOUT (seconds).
        if timeout is None:
            timeout = float(os.environ.get("TT_VLM_TIMEOUT", "900"))
        self.timeout = timeout

    def caption(self, images: list[Image.Image]) -> str:
        payload = {
            "images": [_encode_image(im) for im in images],
            "prompt": self.prompt,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["caption"]
