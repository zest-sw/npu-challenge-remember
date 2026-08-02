from dataclasses import dataclass
import os

from .modes import MODES, ModeSpec


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    initial_mode: str
    initial_spec: ModeSpec
    llm_max_seq_len: int
    vlm_max_seq_len: int
    trace_region_size: int
    mesh_device: str | None
    shutdown_timeout: int
    worker_shutdown_timeout: int
    max_caption_frames: int
    debug_decode: bool
    debug_sample: bool

    @classmethod
    def from_env(cls) -> "ServerSettings":
        named_mode = os.environ.get("TT_MODE") or None
        if named_mode is not None and named_mode not in MODES:
            known = ", ".join(MODES)
            raise ValueError(f"TT_MODE={named_mode!r} is not a known mode. Available: {known}")

        if named_mode is not None:
            initial_mode = named_mode
            initial_spec = MODES[named_mode]
        else:
            llm = os.environ.get("TT_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
            vlm = os.environ.get("TT_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
            initial_mode = "default"
            initial_spec = ModeSpec(
                llm=llm if os.environ.get("TT_ENABLE_LLM", "1") == "1" else None,
                vlm=vlm if os.environ.get("TT_ENABLE_VLM", "1") == "1" else None,
            )

        return cls(
            host=os.environ.get("TT_HOST", "0.0.0.0"),
            port=int(os.environ.get("TT_PORT", "8100")),
            initial_mode=initial_mode,
            initial_spec=initial_spec,
            llm_max_seq_len=int(os.environ.get("TT_LLM_MAX_SEQ_LEN", "4096")),
            vlm_max_seq_len=int(os.environ.get("TT_VLM_MAX_SEQ_LEN", "2048")),
            trace_region_size=int(os.environ.get("TT_TRACE_REGION_SIZE", "30000000")),
            mesh_device=os.environ.get("MESH_DEVICE"),
            shutdown_timeout=int(os.environ.get("TT_SHUTDOWN_TIMEOUT", "5")),
            worker_shutdown_timeout=int(os.environ.get("TT_WORKER_SHUTDOWN_TIMEOUT", "180")),
            max_caption_frames=int(os.environ.get("TT_MAX_CAPTION_FRAMES", "5")),
            debug_decode=os.environ.get("TT_DEBUG_DECODE", "0") == "1",
            debug_sample=os.environ.get("TT_DEBUG_SAMPLE", "0") == "1",
        )
