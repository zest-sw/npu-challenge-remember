import os
import base64
from io import BytesIO

import requests
from PIL import Image

from remembr.captioners.captioner import Captioner


DEFAULT_PROMPT = (
    "<video> Please describe in detail what you see in these frames. "
    "Specifically focus on the people, objects, environmental features, "
    "events/activities, and other interesting details. Be very specific."
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

    def __init__(self, url: str = None, prompt: str = DEFAULT_PROMPT, timeout: float = 300.0):
        self.url = url or os.environ.get("TT_VLM_URL", "http://localhost:8100/caption")
        self.prompt = prompt
        self.timeout = timeout

    def caption(self, images: list[Image.Image]) -> str:
        payload = {
            "images": [_encode_image(im) for im in images],
            "prompt": self.prompt,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["caption"]
