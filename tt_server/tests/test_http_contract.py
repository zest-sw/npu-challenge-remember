import base64
from io import BytesIO
import unittest

from PIL import Image

from tt_server.application.service import InferenceService
from tt_server.domain.errors import CapabilityUnavailable, InvalidRequest
from tt_server.domain.models import LifecycleState, RuntimeSnapshot
from tt_server.presentation.http import create_app
from tt_server.presentation.http.schemas import (
    CaptionRequest,
    ChatCompletionRequest,
    ChatMessage,
    ModeRequest,
)


class StubRuntime:
    def __init__(self):
        self.current = RuntimeSnapshot(
            LifecycleState.READY,
            "3b+3b",
            "meta-llama/test",
            "Qwen/test",
            "performance",
        )

    def snapshot(self):
        return self.current

    def chat(self, command):
        return f"answer:{command.prompt}"

    def caption(self, command):
        return f"caption:{len(command.images)}"

    def switch_mode(self, mode):
        if mode == "caption-3b":
            self.current = RuntimeSnapshot(
                LifecycleState.READY,
                mode,
                None,
                "Qwen/Qwen2.5-VL-3B-Instruct",
            )
            return self.current
        raise CapabilityUnavailable("not available in test")


def encoded_image():
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class HttpContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime = StubRuntime()
        self.app = create_app(InferenceService(self.runtime))

    def endpoint(self, path, method):
        for route in self.app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        self.fail(f"missing route {method} {path}")

    def test_health_preserves_existing_fields(self):
        body = self.endpoint("/health", "GET")()
        self.assertTrue(body["ready"])
        self.assertTrue(body["llm"])
        self.assertTrue(body["vlm"])
        self.assertEqual(body["mode"], "3b+3b")
        self.assertEqual(body["debug"], {"decode": False, "sample": False})
        self.assertEqual(
            body["worker"],
            {
                "busy": False,
                "queue_size": 0,
                "active_job": None,
                "active_elapsed_s": None,
            },
        )

    def test_health_reports_debug_flags(self):
        self.runtime.current = RuntimeSnapshot(
            LifecycleState.READY,
            "3b+3b",
            "meta-llama/test",
            "Qwen/test",
            "performance",
            debug_decode=True,
            debug_sample=True,
        )
        body = self.endpoint("/health", "GET")()
        self.assertEqual(body["debug"], {"decode": True, "sample": True})

    def test_chat_preserves_openai_response_shape(self):
        body = self.endpoint("/v1/chat/completions", "POST")(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
        )
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "answer:hello")

    def test_chat_reports_the_loaded_model_not_the_requested_one(self):
        """The reply must name whatever actually generated it.

        The board holds one model at a time and `model` in the request is not
        honoured, so echoing it back claimed a model had run when it had not —
        a client asking for a 3B against a server in 8B mode was told "3B", and
        eval.py names its results file from that same argument.
        """
        body = self.endpoint("/v1/chat/completions", "POST")(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
                model="some/other-model-that-is-not-loaded",
            )
        )
        self.assertEqual(body["model"], self.runtime.current.llm_model)

    def test_caption_decodes_images(self):
        body = self.endpoint("/caption", "POST")(
            CaptionRequest(images=[encoded_image()])
        )
        self.assertEqual(body, {"caption": "caption:1"})

    def test_invalid_image_is_an_application_error(self):
        with self.assertRaisesRegex(InvalidRequest, "invalid base64 image"):
            self.endpoint("/caption", "POST")(
                CaptionRequest(images=["not-an-image"])
            )

    def test_mode_response_preserves_contract(self):
        body = self.endpoint("/mode", "POST")(
            ModeRequest(mode="caption-3b")
        )
        self.assertTrue(body["switched"])
        self.assertEqual(body["mode"], "caption-3b")


if __name__ == "__main__":
    unittest.main()
