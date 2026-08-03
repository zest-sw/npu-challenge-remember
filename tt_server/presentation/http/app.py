import base64
import io
import time
import uuid

from fastapi import FastAPI
from PIL import Image

from tt_server.application.commands import CaptionCommand, ChatCommand
from tt_server.application.service import InferenceService
from tt_server.domain.errors import (
    CapabilityUnavailable,
    InferenceError,
    InvalidRequest,
    RuntimeNotReady,
    RuntimeSwitching,
    UnknownMode,
)

from .schemas import CaptionRequest, ChatCompletionRequest, ModeRequest


DEFAULT_VLM_PROMPT = (
    "<video> Describe in detail what you see: people, objects, environment, and "
    "activities. Be specific."
)


def create_app(service: InferenceService) -> FastAPI:
    app = FastAPI(title="tt-infer-server")

    @app.exception_handler(InferenceError)
    async def inference_error_handler(_request, error: InferenceError):
        if isinstance(error, (InvalidRequest, UnknownMode)):
            status = 400
        elif isinstance(error, CapabilityUnavailable):
            status = 409
        elif isinstance(error, (RuntimeSwitching, RuntimeNotReady)):
            status = 503
        else:
            status = 500
        return _http_error(status, str(error))

    @app.get("/health")
    def health():
        return _health_payload(service.health())

    @app.get("/mode")
    def get_mode():
        snapshot = service.health()
        return {
            "mode": snapshot.mode,
            "switching": snapshot.switching,
            "llm": snapshot.llm_model,
            "vlm": snapshot.vlm_model,
            "available": list(service.available_modes()),
            "state": snapshot.lifecycle.value,
            "error": snapshot.error,
        }

    @app.post("/mode")
    def set_mode(request: ModeRequest):
        previous = service.health()
        snapshot = service.switch_mode(request.mode)
        return {
            "mode": snapshot.mode,
            "switched": previous.mode != snapshot.mode,
            "switching": snapshot.switching,
            "llm": snapshot.llm_model,
            "vlm": snapshot.vlm_model,
            "available": list(service.available_modes()),
            "state": snapshot.lifecycle.value,
            "error": snapshot.error,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        response_format = request.response_format or {}
        force_json = response_format.get("type") in ("json_object", "json_schema")
        json_schema = None
        if response_format.get("type") == "json_schema":
            wrapper = response_format.get("json_schema") or {}
            json_schema = wrapper.get("schema", wrapper)

        command = ChatCommand(
            prompt=_messages_to_prompt(request.messages),
            max_tokens=request.max_tokens or 256,
            force_json=force_json,
            json_schema=json_schema,
            temperature=request.temperature or 0.0,
        )
        text = service.chat(command)
        snapshot = service.health()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            # Report what actually generated this, never the client's requested
            # name. The board holds one model at a time and the request's `model`
            # is not honoured (see /mode), so echoing it back told callers a model
            # ran that had not: a client asking for Llama-3.2-3B against a server
            # in llm-8b mode got "Llama-3.2-3B" in the reply while 8B did the work.
            # That is how eval.py ends up naming a results file after a model that
            # never ran, since it labels output from its own --model argument.
            "model": snapshot.llm_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    @app.post("/caption")
    def caption(request: CaptionRequest):
        service.validate_caption_frame_count(len(request.images))
        images = tuple(_decode_image(encoded) for encoded in request.images)
        text = service.caption(
            CaptionCommand(
                images=images,
                prompt=request.prompt if request.prompt is not None else DEFAULT_VLM_PROMPT,
                max_gen_len=request.max_gen_len or 128,
            )
        )
        return {"caption": text}

    return app


def _messages_to_prompt(messages) -> str:
    parts = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = str(content)
        parts.append(content if message.role == "user" else f"[{message.role}] {content}")
    return "\n\n".join(parts)


def _decode_image(encoded: str) -> Image.Image:
    try:
        raw = base64.b64decode(encoded)
        return Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    except Exception as error:  # noqa: BLE001
        raise InvalidRequest(f"invalid base64 image: {error}") from error


def _health_payload(snapshot):
    return {
        "ready": snapshot.ready,
        "mode": snapshot.mode,
        "switching": snapshot.switching,
        "llm": snapshot.llm_model is not None,
        "vlm": snapshot.vlm_model is not None,
        "llm_model": snapshot.llm_model,
        "llm_precision": snapshot.llm_precision,
        "state": snapshot.lifecycle.value,
        "error": snapshot.error,
    }


def _http_error(status: int, detail: str):
    # Reuse FastAPI's standard {"detail": ...} error shape from custom handlers.
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"detail": detail})
