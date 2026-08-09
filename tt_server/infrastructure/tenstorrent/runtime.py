import concurrent.futures
import gc
import logging
import os
import queue
import threading

from tt_server.application.commands import CaptionCommand, ChatCommand
from tt_server.config.modes import MODES, ModeSpec
from tt_server.config.settings import ServerSettings
from tt_server.domain.errors import (
    CapabilityUnavailable,
    RuntimeNotReady,
    RuntimeSwitching,
    UnknownMode,
)
from tt_server.domain.models import LifecycleState, RuntimeSnapshot

logger = logging.getLogger(__name__)


class TenstorrentRuntime:
    """Single-threaded actor that exclusively owns the NPU and all model engines."""

    def __init__(
        self,
        settings: ServerSettings,
        device_factory=None,
        llm_factory=None,
        vlm_factory=None,
    ):
        self.settings = settings
        self._device_factory = device_factory
        self._llm_factory = llm_factory
        self._vlm_factory = vlm_factory
        self._device = None
        self._llm = None
        self._vlm = None
        self._jobs = queue.Queue()
        self._started = threading.Event()
        self._startup_error = None
        self._state_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._snapshot = RuntimeSnapshot(LifecycleState.STARTING, None, None, None)
        self._worker = threading.Thread(
            target=self._worker_main,
            name="tt-device-worker",
            # A daemon worker can be discarded by interpreter shutdown before
            # close_mesh_device() runs. Keep it non-daemon as a final safety net.
            daemon=False,
        )

    def start(self) -> RuntimeSnapshot:
        if not self._worker.is_alive():
            self._worker.start()
        self._started.wait()
        if self._startup_error is not None:
            raise RuntimeError("failed to initialize the Tenstorrent runtime") from self._startup_error
        return self.snapshot()

    def snapshot(self) -> RuntimeSnapshot:
        with self._state_lock:
            return self._snapshot

    def chat(self, command: ChatCommand) -> str:
        future = self._admit("llm", self._run_chat, command)
        return future.result()

    def device_memory(self) -> dict:
        """Bytes the device allocator is holding, per buffer type.

        Read-only, and cheap enough to call between requests. Answers the question the
        logs cannot: whether a request gives back everything it took. Growth across
        identical requests is a leak; a falling `largest_free` while `allocated` holds
        steady is fragmentation, which is what eventually refuses an allocation even
        with bytes to spare.
        """
        import ttnn

        if self._device is None:
            return {}
        out = {}
        for name, kind in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
            try:
                view = ttnn.get_memory_view(self._device, kind)
                out[name] = {
                    "allocated": view.total_bytes_allocated_per_bank,
                    "free": view.total_bytes_free_per_bank,
                    "largest_free": view.largest_contiguous_bytes_free_per_bank,
                    "banks": view.num_banks,
                }
            except Exception as error:  # noqa: BLE001
                out[name] = {"error": str(error)[:80]}
        return out

    def caption(self, command: CaptionCommand) -> str:
        future = self._admit("vlm", self._run_caption, command)
        return future.result()

    def switch_mode(self, mode: str) -> RuntimeSnapshot:
        if mode not in MODES:
            raise UnknownMode(f"unknown mode {mode!r}; available: {list(MODES)}")

        with self._admission_lock:
            current = self.snapshot()
            if current.lifecycle == LifecycleState.SWITCHING:
                raise RuntimeSwitching("a mode switch is already in progress")
            if current.lifecycle not in (LifecycleState.READY, LifecycleState.FAILED):
                raise RuntimeNotReady(f"runtime is {current.lifecycle.value}")
            if current.lifecycle == LifecycleState.READY and current.mode == mode:
                return current
            self._set_snapshot(
                LifecycleState.SWITCHING,
                current.mode,
                current.llm_model,
                current.vlm_model,
                current.llm_precision,
            )
            future = self._enqueue(self._apply_mode, mode)
        return future.result()

    def close(self, timeout: float | None = None) -> bool:
        timeout = self.settings.worker_shutdown_timeout if timeout is None else timeout
        with self._admission_lock:
            if not self._worker.is_alive():
                return True
            current = self.snapshot()
            self._set_snapshot(
                LifecycleState.STOPPING,
                current.mode,
                current.llm_model,
                current.vlm_model,
                current.llm_precision,
            )
            self._jobs.put(None)
        self._worker.join(timeout=timeout)
        return not self._worker.is_alive()

    def _admit(self, capability: str, function, *args):
        with self._admission_lock:
            current = self.snapshot()
            if current.lifecycle == LifecycleState.SWITCHING:
                raise RuntimeSwitching("mode switch in progress, retry")
            if current.lifecycle != LifecycleState.READY:
                raise RuntimeNotReady(f"runtime is {current.lifecycle.value}")
            if capability == "llm" and current.llm_model is None:
                raise CapabilityUnavailable(
                    f"chat unavailable in mode {current.mode!r} (no LLM loaded); switch modes first"
                )
            if capability == "vlm" and current.vlm_model is None:
                raise CapabilityUnavailable(
                    f"caption unavailable in mode {current.mode!r} (no VLM loaded); switch modes first"
                )
            return self._enqueue(function, *args)

    def _enqueue(self, function, *args):
        future = concurrent.futures.Future()
        self._jobs.put((function, args, future))
        return future

    def _worker_main(self) -> None:
        try:
            self._ensure_factories()
            self._device = self._device_factory.open()
            self._load_spec(self.settings.initial_spec, self.settings.initial_mode)
        except BaseException as error:  # startup failure must always unblock start()
            self._startup_error = error
            self._set_snapshot(LifecycleState.FAILED, None, None, None, error=str(error))
            logger.exception("Tenstorrent runtime startup failed")
            self._release_device()
            self._started.set()
            return

        self._started.set()
        logger.info("Server device worker ready; mode=%s", self.snapshot().mode)
        try:
            while True:
                item = self._jobs.get()
                if item is None:
                    break
                function, args, future = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(function(*args))
                except BaseException as error:
                    future.set_exception(error)
        finally:
            self._release_device()
            current = self.snapshot()
            self._set_snapshot(
                LifecycleState.STOPPED,
                current.mode,
                None,
                None,
                error=current.error,
            )

    def _ensure_factories(self) -> None:
        if self._device_factory is None:
            from .device import DeviceFactory

            self._device_factory = DeviceFactory(
                self.settings.mesh_device,
                self.settings.trace_region_size,
            )
        if self._llm_factory is None:
            from .llama_engine import LlamaEngine

            self._llm_factory = LlamaEngine
        if self._vlm_factory is None:
            from .qwen_vl_engine import QwenVLEngine

            self._vlm_factory = QwenVLEngine

    def _load_spec(self, spec: ModeSpec, mode_name: str) -> RuntimeSnapshot:
        self._llm = None
        self._vlm = None
        if spec.llm:
            self._llm = self._llm_factory(
                self._device,
                spec.llm,
                self.settings.llm_max_seq_len,
                self.settings.debug_decode,
                self.settings.debug_sample,
            )
            self._llm.load()
        if spec.vlm:
            self._vlm = self._vlm_factory(
                self._device,
                spec.vlm,
                self.settings.vlm_max_seq_len,
            )
            self._vlm.load()
            os.environ["HF_MODEL"] = spec.vlm

        return self._set_snapshot(
            LifecycleState.READY,
            mode_name,
            spec.llm,
            spec.vlm,
            getattr(self._llm, "precision", None),
        )

    def _apply_mode(self, mode: str) -> RuntimeSnapshot:
        try:
            if self._device is not None:
                self._device_factory.synchronize(self._device)
                self._drop_engines()
                device = self._device
                self._device = None
                self._device_factory.close(device)
            self._device = self._device_factory.open()
            return self._load_spec(MODES[mode], mode)
        except BaseException as error:
            self._release_device()
            self._set_snapshot(LifecycleState.FAILED, None, None, None, error=str(error))
            raise

    def _run_chat(self, command: ChatCommand) -> str:
        if self._llm is None:
            mode = self.snapshot().mode
            raise CapabilityUnavailable(
                f"chat unavailable in mode {mode!r} (no LLM loaded); switch modes first"
            )
        return self._llm.generate(
            command.prompt,
            command.max_tokens,
            command.force_json,
            command.json_schema,
            command.temperature,
        )

    def _run_caption(self, command: CaptionCommand) -> str:
        if self._vlm is None:
            mode = self.snapshot().mode
            raise CapabilityUnavailable(
                f"caption unavailable in mode {mode!r} (no VLM loaded); switch modes first"
            )
        return self._vlm.describe(command.images, command.prompt, command.max_gen_len)

    def _release_device(self) -> None:
        if self._device is None:
            return
        # Engine tensor destructors must run while their owning device is open.
        self._drop_engines()
        device = self._device
        self._device = None
        try:
            self._device_factory.close(device)
        except BaseException:  # preserve the original failure during cleanup
            logger.exception("Failed to close the Tenstorrent device")

    def _drop_engines(self) -> None:
        self._llm = None
        self._vlm = None
        gc.collect()

    def _set_snapshot(
        self,
        lifecycle: LifecycleState,
        mode: str | None,
        llm_model: str | None,
        vlm_model: str | None,
        llm_precision: str | None = None,
        error: str | None = None,
    ) -> RuntimeSnapshot:
        snapshot = RuntimeSnapshot(
            lifecycle=lifecycle,
            mode=mode,
            llm_model=llm_model,
            vlm_model=vlm_model,
            llm_precision=llm_precision,
            error=error,
        )
        with self._state_lock:
            self._snapshot = snapshot
        return snapshot
