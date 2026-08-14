"""Host-only checks for interactive_retrieve's data boundary."""

import json
import tempfile
from pathlib import Path

from remembr.scripts.interactive_retrieve import load_caption_entries


def entry() -> dict:
    return {
        "position": [1.0, 2.0, 3.0],
        "theta": 0.0,
        "time": 1.5,
        "caption": "A white box is beside the wall.",
        "file_start": "0.000000.pkl",
        "file_end": "0.166667.pkl",
        "text_embedding": [0.0] * 1024,
    }


def write_fixture(root: Path, value: dict) -> tuple[Path, Path]:
    data_dir = root / "0"
    data_dir.mkdir()
    (data_dir / "0.000000.pkl").touch()
    (data_dir / "0.166667.pkl").touch()
    captions = root / "captions.json"
    captions.write_text(json.dumps([value]), encoding="utf-8")
    return captions, data_dir


def test_accepts_caption_file_linking_to_source_pkls():
    with tempfile.TemporaryDirectory() as temp_dir:
        captions, data_dir = write_fixture(Path(temp_dir), entry())
        assert load_caption_entries(captions, data_dir)[0]["caption"].startswith("A white")


def test_rejects_missing_source_pkl():
    with tempfile.TemporaryDirectory() as temp_dir:
        value = entry()
        value["file_end"] = "missing.pkl"
        captions, data_dir = write_fixture(Path(temp_dir), value)
        try:
            load_caption_entries(captions, data_dir)
        except ValueError as error:
            assert "missing PKL" in str(error), error
        else:
            raise AssertionError("missing PKL was accepted")


def test_rejects_wrong_embedding_dimension():
    with tempfile.TemporaryDirectory() as temp_dir:
        value = entry()
        value["text_embedding"] = [0.0] * 3
        captions, data_dir = write_fixture(Path(temp_dir), value)
        try:
            load_caption_entries(captions, data_dir)
        except ValueError as error:
            assert "1024D" in str(error), error
        else:
            raise AssertionError("wrong embedding dimension was accepted")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
    print("all passed")
