from dataclasses import dataclass
from enum import Enum

from .errors import InvalidRequest


def cap_long_side(image, long_side: int):
    """Downscale a frame only if it is over `long_side`, keeping the aspect ratio.
    0 leaves it alone.

    The HTTP layer used to force every frame to 224x224, so captions could not read
    signs that were no longer in the pixels (#59). Lives here, not in the HTTP layer,
    because how much image you may send is the same decision as the token budget it
    has to fit -- and so it is testable without fastapi."""
    if long_side and max(image.size) > long_side:
        image = image.copy()
        image.thumbnail((long_side, long_side))
    return image


def check_generation_fits(prefill_len: int, max_gen_len: int, max_seq_len: int) -> None:
    """Refuse a request that would decode past the KV cache instead of hanging on it.

    Past the cache `decode_forward` writes through a NoC that has no bounds concept, so
    it lands in whatever DRAM follows: usually harmless, occasionally a spinning core and
    a board only a reboot clears. A client cannot check this itself -- it has neither the
    tokenizer nor `max_seq_len`, and image tokens depend on resolution."""
    if prefill_len + max_gen_len <= max_seq_len:
        return
    raise InvalidRequest(
        f"needs {prefill_len} prompt tokens + {max_gen_len} generated = "
        f"{prefill_len + max_gen_len}, over max_seq_len {max_seq_len}. "
        f"Send a smaller image (this prompt leaves room for "
        f"{max(0, max_seq_len - max_gen_len)} prompt tokens; a 1224x1024 frame is ~1200 "
        f"on its own), lower max_gen_len, or raise TT_VLM_MAX_SEQ_LEN."
    )


class LifecycleState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    SWITCHING = "switching"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RuntimeSnapshot:
    lifecycle: LifecycleState
    mode: str | None
    llm_model: str | None
    vlm_model: str | None
    llm_precision: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.lifecycle == LifecycleState.READY

    @property
    def switching(self) -> bool:
        return self.lifecycle == LifecycleState.SWITCHING
