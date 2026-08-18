import threading
import time
import unittest

from tt_server.application.commands import CaptionCommand, ChatCommand
from tt_server.config.modes import MODES
from tt_server.config.settings import ServerSettings
from tt_server.domain.errors import CapabilityUnavailable
from tt_server.domain.models import ChatResult, LifecycleState
from tt_server.infrastructure.tenstorrent.runtime import TenstorrentRuntime


def settings(initial_mode="3b+3b"):
    return ServerSettings(
        host="127.0.0.1",
        port=8100,
        initial_mode=initial_mode,
        initial_spec=MODES[initial_mode],
        llm_max_seq_len=1024,
        vlm_max_seq_len=1024,
        trace_region_size=1,
        mesh_device="N150",
        shutdown_timeout=1,
        worker_shutdown_timeout=1,
        max_caption_frames=5,
        debug_decode=False,
        debug_sample=False,
    )


class FakeDeviceFactory:
    def __init__(self):
        self.events = []
        self.counter = 0

    def open(self):
        self.counter += 1
        device = f"device-{self.counter}"
        self.events.append(("open", device, threading.get_ident()))
        return device

    def synchronize(self, device):
        self.events.append(("sync", device, threading.get_ident()))

    def close(self, device):
        self.events.append(("close", device, threading.get_ident()))


class FakeLLM:
    def __init__(self, device, model, max_seq_len, debug_decode, debug_sample):
        self.device = device
        self.model_name = model
        self.precision = "fake"

    def load(self):
        pass

    def generate(self, prompt, max_tokens, force_json, schema, temperature):
        return ChatResult(f"llm:{prompt}")


class FakeVLM:
    def __init__(self, device, model, max_seq_len):
        self.device = device
        self.model_name = model

    def load(self):
        pass

    def describe(self, images, prompt, max_gen_len):
        return f"vlm:{prompt}:{len(images)}"


class BlockingVLM(FakeVLM):
    started = threading.Event()
    release = threading.Event()

    def describe(self, images, prompt, max_gen_len):
        type(self).started.set()
        type(self).release.wait(timeout=5)
        return super().describe(images, prompt, max_gen_len)


class FailingQwen3VLM(FakeVLM):
    def load(self):
        if "Qwen3-VL" in self.model_name:
            raise RuntimeError("qwen3 load failed")


class FailingLLM(FakeLLM):
    def load(self):
        raise RuntimeError("load failed")


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, llm_factory=FakeLLM, vlm_factory=FakeVLM):
        devices = FakeDeviceFactory()
        runtime = TenstorrentRuntime(
            settings(),
            device_factory=devices,
            llm_factory=llm_factory,
            vlm_factory=vlm_factory,
        )
        return runtime, devices

    def test_inference_and_device_lifecycle_run_on_worker(self):
        runtime, devices = self.make_runtime()
        snapshot = runtime.start()
        self.assertTrue(snapshot.ready)
        self.assertEqual(runtime.chat(ChatCommand("hello")).text, "llm:hello")
        self.assertEqual(
            runtime.caption(CaptionCommand((object(),), "scene")),
            "vlm:scene:1",
        )
        self.assertTrue(runtime.close())
        thread_ids = {event[2] for event in devices.events}
        self.assertEqual(len(thread_ids), 1)
        self.assertNotIn(threading.get_ident(), thread_ids)

    def test_mode_switch_closes_and_reopens_device(self):
        runtime, devices = self.make_runtime()
        runtime.start()
        snapshot = runtime.switch_mode("caption-3b")
        self.assertEqual(snapshot.mode, "caption-3b")
        self.assertIsNone(snapshot.llm_model)
        with self.assertRaises(CapabilityUnavailable):
            runtime.chat(ChatCommand("unavailable"))
        runtime.close()
        names = [event[0] for event in devices.events]
        self.assertEqual(names[:4], ["open", "sync", "close", "open"])

    def test_startup_failure_unblocks_and_releases_device(self):
        runtime, devices = self.make_runtime(llm_factory=FailingLLM)
        with self.assertLogs(
            "tt_server.infrastructure.tenstorrent.runtime", level="ERROR"
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
                runtime.start()
        self.assertEqual(runtime.snapshot().lifecycle, LifecycleState.FAILED)
        self.assertEqual([event[0] for event in devices.events], ["open", "close"])

    def test_failed_mode_switch_can_recover_with_a_new_mode(self):
        runtime, devices = self.make_runtime(vlm_factory=FailingQwen3VLM)
        runtime.start()
        with self.assertRaisesRegex(RuntimeError, "qwen3 load failed"):
            runtime.switch_mode("caption-8b")
        failed = runtime.snapshot()
        self.assertEqual(failed.lifecycle, LifecycleState.FAILED)
        self.assertIsNone(failed.mode)
        recovered = runtime.switch_mode("caption-3b")
        self.assertTrue(recovered.ready)
        self.assertEqual(recovered.mode, "caption-3b")
        runtime.close()
        self.assertEqual(
            [event[0] for event in devices.events],
            ["open", "sync", "close", "open", "close", "open", "close"],
        )

    def test_snapshot_reports_active_worker_job(self):
        BlockingVLM.started.clear()
        BlockingVLM.release.clear()

        runtime, _ = self.make_runtime(vlm_factory=BlockingVLM)
        runtime.start()

        result = {}
        thread = threading.Thread(
            target=lambda: result.setdefault(
                "caption", runtime.caption(CaptionCommand((object(),), "scene"))
            )
        )
        thread.start()
        self.assertTrue(BlockingVLM.started.wait(timeout=1))

        second_result = {}
        second = threading.Thread(
            target=lambda: second_result.setdefault(
                "caption", runtime.caption(CaptionCommand((object(),), "queued"))
            )
        )
        second.start()
        queued = None
        for _ in range(100):
            queued = runtime.snapshot()
            if queued.worker_queue_size == 1:
                break
            time.sleep(0.01)
        self.assertEqual(queued.worker_queue_size, 1)

        snapshot = runtime.snapshot()
        self.assertTrue(snapshot.worker_busy)
        self.assertEqual(snapshot.worker_active_job, "caption")
        self.assertEqual(snapshot.worker_queue_size, 1)
        self.assertIsNotNone(snapshot.worker_active_elapsed_s)
        self.assertGreaterEqual(snapshot.worker_active_elapsed_s, 0)

        time.sleep(0.01)
        later = runtime.snapshot()
        self.assertGreaterEqual(
            later.worker_active_elapsed_s, snapshot.worker_active_elapsed_s
        )

        BlockingVLM.release.set()
        thread.join(timeout=1)
        second.join(timeout=1)
        self.assertEqual(result["caption"], "vlm:scene:1")
        self.assertEqual(second_result["caption"], "vlm:queued:1")
        idle = runtime.snapshot()
        self.assertFalse(idle.worker_busy)
        self.assertIsNone(idle.worker_active_job)
        self.assertIsNone(idle.worker_active_elapsed_s)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
