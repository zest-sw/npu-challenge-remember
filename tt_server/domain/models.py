from dataclasses import dataclass
from enum import Enum


def cap_long_side(image, long_side: int):
    """Downscale a frame only if it is over `long_side`, keeping the aspect ratio.
    0 leaves it alone.

    Lives here rather than in the HTTP layer so it is testable without fastapi.
    """
    if long_side and max(image.size) > long_side:
        image = image.copy()
        image.thumbnail((long_side, long_side))
    return image


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
