import base64
import io
import os
import time
import uuid

from fastapi import FastAPI
from PIL import Image

from tt_server.application.commands import CaptionCommand, ChatCommand
from tt_server.application.service import InferenceService
from tt_server.domain.models import cap_long_side
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

# Longest side a caption frame may reach the VLM at. See _decode_image for why this
# exists and what it costs. 0 disables downscaling entirely.
MAX_IMAGE_LONG_SIDE = int(os.environ.get("TT_VLM_IMAGE_LONG_SIDE", "768"))


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
    """Decode a frame, downscaling only if it is over the budget.

    This used to be `.resize((224, 224))` -- unconditional, whatever the client sent,
    with no setting to change it. The camera is 1224x1024, so every frame reached the
    VLM at 1/25 of its area. That is why captions never read a sign or named a building
    (#59): at 224x224 the text is not there to read. Of 272 captions, 33 named a
    building and exactly 1 mentioned the police call pole the questions ask about; a
    later prompt that said "copy any words visible on signs, exactly" produced quoted
    text in 0 of 8 samples. No prompt can fix a missing pixel.

    It also made resolution untestable: a sweep from 1224 down to 256 returned ten
    identical 21.6s responses, because the server was answering the same request each
    time.

    Cap the long side instead, and keep the aspect ratio. Measured on Qwen3-VL-8B:

        long side   image tokens   prefill (terse)   fits max_seq_len
             1224           1216              1429   needs 2048
              768            465               693   1024
              224             49               277   1024

    The default keeps a stock TT_VLM_MAX_SEQ_LEN=1024 server working. For full
    resolution raise both together, since prefill has to fit the KV cache:

        TT_VLM_IMAGE_LONG_SIDE=1224 TT_VLM_MAX_SEQ_LEN=2048
    """
    try:
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as error:  # noqa: BLE001
        raise InvalidRequest(f"invalid base64 image: {error}") from error
    return cap_long_side(image, MAX_IMAGE_LONG_SIDE)


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
