import logging
import os
import signal
import threading

from tt_server.application.service import InferenceService
from tt_server.config.settings import ServerSettings
from tt_server.infrastructure.tenstorrent.runtime import TenstorrentRuntime
from tt_server.presentation.http import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    try:
        settings = ServerSettings.from_env()
    except ValueError as error:
        raise SystemExit(str(error)) from error

    runtime = TenstorrentRuntime(settings)
    service = InferenceService(runtime, max_caption_frames=settings.max_caption_frames)
    app = create_app(service)
    logger.info("Loading models on the NPU (this can take several minutes on first run)...")
    # Serve during the load, so /health can say "starting" instead of not answering at
    # all. Blocking here made a slow warmup indistinguishable from a wedged board.
    def load() -> None:
        try:
            runtime.start()
        except Exception:
            logger.exception("NPU load failed; shutting down")
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=load, name="tt-load", daemon=True).start()
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            timeout_graceful_shutdown=settings.shutdown_timeout,
        )
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        logger.info("Shutting down: releasing the NPU cleanly — do NOT force-kill (-9)...")
        if runtime.close(settings.worker_shutdown_timeout):
            logger.info("NPU released cleanly. Safe to exit.")
        else:
            logger.warning(
                f"Device worker still busy after {settings.worker_shutdown_timeout}s; "
                "device may not have closed cleanly."
            )
