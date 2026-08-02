import os
import unittest
from unittest.mock import patch

from tt_server.config.settings import ServerSettings


class SettingsTests(unittest.TestCase):
    def test_unknown_named_mode_fails_before_runtime_creation(self):
        with patch.dict(os.environ, {"TT_MODE": "typo"}, clear=True):
            with self.assertRaisesRegex(ValueError, "not a known mode"):
                ServerSettings.from_env()

    def test_legacy_model_environment_is_preserved(self):
        environment = {
            "TT_ENABLE_LLM": "0",
            "TT_ENABLE_VLM": "1",
            "TT_VLM_MODEL": "custom/vlm",
        }
        with patch.dict(os.environ, environment, clear=True):
            loaded = ServerSettings.from_env()
        self.assertEqual(loaded.initial_mode, "default")
        self.assertIsNone(loaded.initial_spec.llm)
        self.assertEqual(loaded.initial_spec.vlm, "custom/vlm")


if __name__ == "__main__":
    unittest.main()
