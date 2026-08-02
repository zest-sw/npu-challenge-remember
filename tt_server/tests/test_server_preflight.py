import fcntl
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


PREFLIGHT = Path(__file__).resolve().parents[1] / "server_preflight.sh"


class ServerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.lock_file = self.root / "server.lock"
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "TT_SERVER_LOCK_FILE": str(self.lock_file),
            "BOS_DEV_NODE": str(self.root / "missing-device"),
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def command(self, name: str, body: str) -> None:
        path = self.bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_preflight(self):
        return subprocess.run(
            ["bash", "-c", f"source {PREFLIGHT!s}; tt_server_preflight"],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_ready_server_is_a_successful_noop_signal(self):
        self.command("curl", "echo '{\"ready\":true,\"mode\":\"3b+3b\"}'")
        self.command("pgrep", "exit 1")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 10)
        self.assertIn("server already ready", result.stderr)
        self.assertIn('"ready":true', result.stderr)

    def test_existing_unhealthy_server_blocks_startup(self):
        self.command("curl", "exit 22")
        self.command("pgrep", "echo '123 python tt_infer_server.py'")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 1)
        self.assertIn("process exists", result.stderr)
        self.assertIn("123 python", result.stderr)

    def test_device_holder_blocks_startup(self):
        device = self.root / "bos0"
        device.touch()
        self.environment["BOS_DEV_NODE"] = str(device)
        self.command("curl", "exit 22")
        self.command("pgrep", "exit 1")
        self.command("fuser", "echo '456'")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 1)
        self.assertIn("already held", result.stderr)
        self.assertIn("456", result.stderr)

    def test_free_device_allows_startup_to_continue(self):
        self.command("curl", "exit 22")
        self.command("pgrep", "exit 1")
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0)

    def test_concurrent_launcher_lock_skips_second_start(self):
        with self.lock_file.open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_preflight()
        self.assertEqual(result.returncode, 10)
        self.assertIn("running or starting", result.stderr)

    def test_lock_fd_is_inherited_by_server_process(self):
        self.command("curl", "exit 22")
        self.command("pgrep", "exit 1")
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {PREFLIGHT!s}; tt_server_preflight && "
                "exec python3 -c 'import os; os.fstat(9); print(\"lock-inherited\")'",
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "lock-inherited")


if __name__ == "__main__":
    unittest.main()
