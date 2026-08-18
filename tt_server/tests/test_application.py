import unittest

from tt_server.application.commands import CaptionCommand, ChatCommand
from tt_server.application.service import InferenceService
from tt_server.domain.errors import InvalidRequest
from tt_server.domain.models import ChatResult, LifecycleState, RuntimeSnapshot


class StubRuntime:
    def snapshot(self):
        return RuntimeSnapshot(LifecycleState.READY, "3b+3b", "llm", "vlm")

    def chat(self, command):
        return ChatResult(command.prompt)

    def caption(self, command):
        return command.prompt

    def switch_mode(self, mode):
        return self.snapshot()


class InferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = InferenceService(StubRuntime(), max_caption_frames=5)

    def test_rejects_six_or_more_caption_frames(self):
        with self.assertRaisesRegex(InvalidRequest, "at most 5"):
            self.service.caption(CaptionCommand(tuple(range(6)), "caption"))

    def test_rejects_negative_temperature(self):
        with self.assertRaisesRegex(InvalidRequest, "temperature"):
            self.service.chat(ChatCommand("hello", temperature=-0.1))

    def test_forwards_valid_commands(self):
        self.assertEqual(self.service.chat(ChatCommand("hello")).text, "hello")
        self.assertEqual(
            self.service.caption(CaptionCommand((object(),), "describe")),
            "describe",
        )


if __name__ == "__main__":
    unittest.main()
