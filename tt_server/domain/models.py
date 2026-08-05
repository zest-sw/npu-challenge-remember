from dataclasses import dataclass
from enum import Enum

from .errors import InvalidRequest


def check_generation_fits(prefill_len: int, max_gen_len: int, max_seq_len: int) -> None:
    """Refuse a request that would decode past the KV cache instead of hanging on it.

    The VLM engine sized its token buffer as `prefill + max_gen_len` and never compared
    that to `max_seq_len`. Past the cache, `decode_forward` is undefined: it raises
    nothing and returns nothing, one thread spins at 100% inside ttnn, and the board stays
    held. The main thread sits in a futex at that point, and Python runs signal handlers
    only on the main thread -- so Ctrl-C is never processed and /health never answers.
    Killing the process is the only way out.

    Measured on the CODa frames, every caption request had been over the limit the whole
    time. The camera is 1224x1024, which is ~1200 image tokens on its own:

        TT_VLM_MAX_SEQ_LEN   1024
        paper prompt         prefill 1279 + 256 = 1535
        terse prompt         prefill 1429 + 128 = 1557

    So the 272-caption pass and both probes ran outside the cache and mostly got away with
    it. Eight captions succeeded and the ninth wedged the board.

    A client cannot check this for itself: it has neither the tokenizer nor `max_seq_len`,
    and the image token count depends on resolution. A comment in the captioner asked it
    to ("Must satisfy prefill+max_gen_len <= TT_VLM_MAX_SEQ_LEN") and nothing enforced it.

    Lives in the domain layer so the rule is testable without torch, ttnn or a board.
    """
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
