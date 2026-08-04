"""The LLM engine must never ask the device to sample.

tt-metal caches the decode trace on `hasattr(self, "trace_ids_text")` alone
(generator.py:370), so the trace is not keyed on whether it returns a token id
or full logits. If the engine has two decode paths, whichever kind of request
arrives first captures the trace and every request of the other kind then reads
the wrong buffer for the rest of the process's life -- JSON answers come back as
'{"!!!!!!!!...' and plain answers as 'Hello9001/+,,+,**+++'. See issue #21.

Catching that for real needs a board, and the tempting regression is someone
re-adding a device-sampling fast path for speed, so this guards the source: the
engine passes `sampling_params=None` on every decode, always.
"""
import unittest
from pathlib import Path

ENGINE = Path(__file__).parents[1] / "infrastructure" / "tenstorrent" / "llama_engine.py"


class TestSingleDecodePath(unittest.TestCase):
    def test_every_decode_call_reads_logits(self):
        source = ENGINE.read_text()
        calls = source.count("decode_forward_text(")
        nones = source.count("sampling_params=None")
        self.assertEqual(
            calls,
            nones,
            f"{calls} decode call(s) but {nones} with sampling_params=None; "
            "a decode that lets the device sample returns token ids, not logits",
        )


if __name__ == "__main__":
    unittest.main()
