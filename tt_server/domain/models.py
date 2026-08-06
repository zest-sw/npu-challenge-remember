from dataclasses import dataclass
from enum import Enum


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
    # debug modes
    debug_decode: bool = False
    debug_sample: bool = False

    @property
    def ready(self) -> bool:
        return self.lifecycle == LifecycleState.READY

    @property
    def switching(self) -> bool:
        return self.lifecycle == LifecycleState.SWITCHING
