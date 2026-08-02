from typing import Any

from pydantic import BaseModel, Field


class ModeRequest(BaseModel):
    mode: str


class CaptionRequest(BaseModel):
    images: list[str] = Field(default_factory=list)
    prompt: str | None = None
    max_gen_len: int | None = None


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    model: str | None = None
    response_format: dict[str, Any] | None = None
