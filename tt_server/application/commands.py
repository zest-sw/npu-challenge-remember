from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatCommand:
    prompt: str
    max_tokens: int = 256
    force_json: bool = False
    json_schema: dict | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class CaptionCommand:
    images: tuple[Any, ...]
    prompt: str
    max_gen_len: int = 128
