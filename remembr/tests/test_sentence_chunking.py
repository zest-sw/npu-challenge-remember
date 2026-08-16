from types import SimpleNamespace
import json
import sys
from pathlib import Path

import pytest

from remembr.memory.sentence_chunking import (
    build_sentence_chunk_index,
    dedupe_parent_documents,
    parent_id_from_record_id,
    sentence_record_id,
    split_caption_sentences,
)


INNER_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INNER_REPO))


class StubEmbedder:
    model_name = "stub"

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


def doc(record_id, text):
    return SimpleNamespace(metadata={"id": record_id}, page_content=text)


def test_split_caption_sentences_keeps_useful_sentences_and_decimal():
    caption = "A red hydrant stands here. Okay. It is 3.5 meters from the curb."
    assert split_caption_sentences(caption) == [
        "A red hydrant stands here.",
        "It is 3.5 meters from the curb.",
    ]


def test_build_sentence_chunks_batches_embeddings_and_preserves_parent():
    embedder = StubEmbedder()
    chunks = build_sentence_chunk_index([
        {"id": "frame-a.pkl", "caption": "A bench is empty. A bicycle is beside it."},
        {"id": "frame-b.pkl", "caption": "A red hydrant stands at the curb."},
    ], embedder)

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [
        "A bench is empty.",
        "A bicycle is beside it.",
        "A red hydrant stands at the curb.",
    ]
    assert [chunk["sentence_index"] for chunk in chunks["frame-a.pkl"]] == [0, 1]
    assert chunks["frame-b.pkl"][0]["text_embedding"] == [2.0, 33.0]


def test_parent_dedupe_keeps_best_sentence_and_fills_distinct_parents():
    ranked = [
        doc(sentence_record_id("parent-a", 1), "best A"),
        doc(sentence_record_id("parent-a", 0), "second A"),
        doc(sentence_record_id("parent-b", 2), "best B"),
        doc(sentence_record_id("parent-c", 0), "best C"),
    ]

    selected = dedupe_parent_documents(ranked, limit=3)

    assert [item.page_content for item in selected] == ["best A", "best B", "best C"]
    assert parent_id_from_record_id(selected[0].metadata["id"]) == "parent-a"


def test_parent_dedupe_fails_loudly_for_unchunked_rows():
    with pytest.raises(ValueError, match="no parent id"):
        dedupe_parent_documents([doc("plain-row", "caption")], limit=1)


def test_eval_loads_sentence_rows_with_parent_ids_and_default_k7(tmp_path, monkeypatch):
    from remembr.scripts import eval as eval_module

    captions_dir = tmp_path / "captions" / "0" / "captions"
    captions_dir.mkdir(parents=True)
    captions = [
        {
            "id": "frame-a.pkl", "file_start": "1.0.pkl", "file_end": "2.0.pkl",
            "position": [1.0, 2.0, 3.0], "theta": 0.0, "time": 1.5,
            "caption": "A bench is empty. A bicycle is beside it.",
            "text_embedding": [99.0, 99.0],
        },
        {
            "id": "frame-b.pkl", "file_start": "2.0.pkl", "file_end": "3.0.pkl",
            "position": [4.0, 5.0, 6.0], "theta": 0.0, "time": 2.5,
            "caption": "A red hydrant stands here.",
            "text_embedding": [88.0, 88.0],
        },
    ]
    (captions_dir / "fixture.json").write_text(json.dumps(captions))

    created = []

    class FakeMemory:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.embedder = StubEmbedder()
            self.inserted = []
            created.append(self)

        def reset(self):
            pass

        def insert(self, item, text_embedding=None, record_id=None):
            self.inserted.append((item, text_embedding, record_id))

    monkeypatch.setattr(eval_module, "MilvusMemory", FakeMemory)
    args = SimpleNamespace(
        sequence_id=0, data_dir=str(tmp_path), coda_dir=str(tmp_path / "coda"),
        caption_file="fixture", model="direct+tt/test", memory_chunking="sentence",
        retrieval_k=None,
    )
    qa = {"start_time": 1.0, "end_time": 3.0}

    memory, parents = eval_module.load_memory(args, qa, use_milvus=True)

    assert len(parents) == 2
    assert memory.args[0] == "eval_memory_0"
    assert memory.kwargs["text_search_k"] == 7
    assert memory.kwargs["dedupe_text_parents"] is True
    assert [row[0].caption for row in memory.inserted] == [
        "A bench is empty.",
        "A bicycle is beside it.",
        "A red hydrant stands here.",
    ]
    assert [row[2] for row in memory.inserted] == [
        sentence_record_id("frame-a.pkl", 0),
        sentence_record_id("frame-a.pkl", 1),
        sentence_record_id("frame-b.pkl", 0),
    ]
    assert memory.inserted[0][0].position == [1.0, 2.0, 3.0]
    assert memory.inserted[0][1] != captions[0]["text_embedding"]


def test_eval_uses_explicit_collection_name(tmp_path, monkeypatch):
    from remembr.scripts import eval as eval_module

    captions_dir = tmp_path / "captions" / "0" / "captions"
    captions_dir.mkdir(parents=True)
    captions = [{
        "id": "frame.pkl", "file_start": "1.0.pkl", "file_end": "2.0.pkl",
        "position": [1.0, 2.0, 3.0], "theta": 0.0, "time": 1.5,
        "caption": "A bench is empty.", "text_embedding": [1.0, 2.0],
    }]
    (captions_dir / "fixture.json").write_text(json.dumps(captions))
    created = []

    class FakeMemory:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.embedder = StubEmbedder()
            created.append(self)

        def reset(self):
            pass

        def insert(self, *args, **kwargs):
            pass

    monkeypatch.setattr(eval_module, "MilvusMemory", FakeMemory)
    args = SimpleNamespace(
        sequence_id=0, data_dir=str(tmp_path), coda_dir=str(tmp_path / "coda"),
        caption_file="fixture", model="direct+tt/test", memory_chunking="caption",
        retrieval_k=None, db_name="eval_sentence_isolated",
    )

    eval_module.load_memory(args, {"start_time": 1.0, "end_time": 2.0}, use_milvus=True)

    assert created[0].args[0] == "eval_sentence_isolated"
