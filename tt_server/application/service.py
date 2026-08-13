from tt_server.config.modes import MODES
from tt_server.domain.errors import InvalidRequest

from .commands import CaptionCommand, ChatCommand
from .ports import RuntimePort


class InferenceService:
    def __init__(self, runtime: RuntimePort, max_caption_frames: int = 5):
        self.runtime = runtime
        self.max_caption_frames = max_caption_frames

    def health(self):
        return self.runtime.snapshot()

    def device_memory(self):
        return self.runtime.device_memory()

    def available_modes(self):
        return tuple(MODES)

    def chat(self, command: ChatCommand) -> str:
        if command.max_tokens < 1:
            raise InvalidRequest("max_tokens must be at least 1")
        if command.temperature < 0:
            raise InvalidRequest("temperature must not be negative")
        return self.runtime.chat(command)

    def caption(self, command: CaptionCommand) -> str:
        self.validate_caption_frame_count(len(command.images))
        if command.max_gen_len < 1:
            raise InvalidRequest("max_gen_len must be at least 1")
        return self.runtime.caption(command)

    def validate_caption_frame_count(self, count: int) -> None:
        if count == 0:
            raise InvalidRequest("no images provided")
        if count > self.max_caption_frames:
            raise InvalidRequest(
                f"too many images: {count}; at most {self.max_caption_frames} frames are allowed"
            )

    def switch_mode(self, mode: str):
        return self.runtime.switch_mode(mode)
