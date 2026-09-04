"""Self-check for sentence-level indexing in MilvusMemory.

No Milvus, no NPU, no embedding model:

    python remembr/tests/test_sentence_index.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.memory.milvus_memory import (  # noqa: E402
    POSITION_TIME_K,
    SENTENCE_OVERFETCH,
    TEXT_SEARCH_K,
    MilvusMemory,
    first_per_memory,
    sentences_of,
)


class FakeDoc:
    def __init__(self, caption, time=0.0):
        self.page_content = caption
        self.metadata = {"time": [time, 0.0], "position": [0.0, 0.0, 0.0]}


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.asked = None

    def invoke(self, query):
        self.asked = query
        return self.docs


class FakeEmbedder:
    """Returns a distinct vector per text so rows can be told apart."""

    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


def make_memory(index_unit, text_k=TEXT_SEARCH_K):
    """Build the memory without __init__, which would connect to Milvus."""
    memory = MilvusMemory.__new__(MilvusMemory)
    memory.index_unit = index_unit
    memory.text_k = text_k
    memory.embedder = FakeEmbedder()
    memory.time_offset = 0.0
    return memory


CAPTION = ("Concrete sidewalk stretches forward. A metal bench sits empty near the wall. "
           "A gray utility box stands between the sidewalk and the enclosure.")


def test_a_caption_splits_into_its_sentences():
    assert sentences_of(CAPTION) == [
        "Concrete sidewalk stretches forward.",
        "A metal bench sits empty near the wall.",
        "A gray utility box stands between the sidewalk and the enclosure.",
    ]


def test_question_and_exclamation_marks_also_end_a_sentence():
    assert sentences_of("A door is open! Is anyone standing there? Nobody is visible.") == [
        "A door is open!", "Is anyone standing there?", "Nobody is visible."]


def test_fragments_too_short_to_carry_meaning_are_dropped():
    """A two-word fragment embeds to noise and would compete for one of the k slots."""
    assert sentences_of("Ok. A red fire hydrant stands on the corner.") == [
        "A red fire hydrant stands on the corner."]


def test_a_caption_with_nothing_long_enough_is_kept_whole():
    """Dropping every sentence would lose the memory outright, which is worse than
    indexing a short caption as one unit."""
    assert sentences_of("Ok. Yes.") == ["Ok. Yes."]
    assert sentences_of("   ") == []


def test_each_memory_appears_once_at_its_best_sentence():
    """Sentence rows repeat their parent caption. Without this, one caption with two
    strong sentences would take two of the five slots the model chooses between."""
    docs = [FakeDoc("first"), FakeDoc("second"), FakeDoc("first"), FakeDoc("third")]
    assert [doc.page_content for doc in first_per_memory(docs, 5)] == [
        "first", "second", "third"]


def test_two_memories_that_share_a_caption_are_both_kept():
    """A stationary robot repeats itself, and those are still separate memories at
    separate times. Collapsing them would answer with the wrong moment."""
    docs = [FakeDoc("a hallway", time=10.0), FakeDoc("a hallway", time=99.0)]
    assert len(first_per_memory(docs, 5)) == 2


def test_deduplication_stops_once_the_limit_is_reached():
    docs = [FakeDoc(str(n)) for n in range(10)]
    assert len(first_per_memory(docs, 3)) == 3


def test_over_fetching_only_happens_when_rows_are_sentences():
    """Caption mode has one row per memory, so asking for more would change what the
    original returns."""
    assert make_memory("caption").fetch_count(TEXT_SEARCH_K) == TEXT_SEARCH_K
    assert make_memory("sentence").fetch_count(TEXT_SEARCH_K) > TEXT_SEARCH_K


def test_a_text_pool_smaller_than_one_memory_is_refused():
    """The pool size is a constructor argument so a k sweep does not have to reach into
    `text_retriever.search_kwargs`, which is what the reverted rerank experiment did."""
    try:
        MilvusMemory(db_collection_name="never_built", text_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for an empty text pool")


def test_text_search_returns_as_many_memories_as_the_pool_allows():
    memory = make_memory("sentence", text_k=3)
    memory.text_retriever = FakeRetriever([FakeDoc(str(n)) for n in range(10)])
    memory._unflushed_inserts = False
    memory.working_memory = []
    memory.search_by_text("where is the bench?")
    assert memory.text_retriever.asked == "where is the bench?"
    assert len(memory.get_working_memory()) == 3, memory.get_working_memory()


def test_caption_mode_returns_search_hits_untouched():
    """Two memories may legitimately share a caption, so caption mode must not dedupe."""
    docs = [FakeDoc("a hallway"), FakeDoc("a hallway")]
    assert make_memory("caption")._distinct(docs, POSITION_TIME_K) == docs


def test_sentence_mode_deduplicates_search_hits():
    docs = [FakeDoc("a hallway"), FakeDoc("a hallway")]
    assert len(make_memory("sentence")._distinct(docs, POSITION_TIME_K)) == 1


ITEM = {"caption": CAPTION, "position": [1.0, 2.0, 3.0], "theta": 0.5, "time": [10.0, 0.0]}


def test_sentence_mode_stores_one_row_per_sentence():
    rows = make_memory("sentence").rows_for(dict(ITEM))
    assert len(rows) == 3
    # Each row is searched by its own sentence but answers with the whole caption, so
    # the model still reads the memory in context.
    assert [row["caption"] for row in rows] == [CAPTION] * 3
    assert [row["position"] for row in rows] == [[1.0, 2.0, 3.0]] * 3
    assert [row["text_embedding"] for row in rows] == [
        [float(len(sentence))] for sentence in sentences_of(CAPTION)]


def test_sentence_rows_get_distinct_ids():
    """id is the primary key; repeating it within one caption would drop rows."""
    ids = [row["id"] for row in make_memory("sentence").rows_for(dict(ITEM))]
    assert len(set(ids)) == len(ids), ids


def test_caption_mode_stores_one_row_and_reuses_the_given_embedding():
    """The caption files ship precomputed embeddings; recomputing them would be slower
    and would not reproduce earlier measurements bit for bit."""
    rows = make_memory("caption").rows_for(dict(ITEM), text_embedding=[9.0])
    assert len(rows) == 1
    assert rows[0]["text_embedding"] == [9.0]
    assert rows[0]["caption"] == CAPTION


def test_caption_mode_embeds_the_caption_when_none_was_given():
    rows = make_memory("caption").rows_for(dict(ITEM))
    assert rows[0]["text_embedding"] == [float(len(CAPTION))]


def test_sentence_mode_ignores_a_precomputed_caption_embedding():
    """It describes the whole caption, which is the thing sentence mode exists to avoid."""
    rows = make_memory("sentence").rows_for(dict(ITEM), text_embedding=[9.0])
    assert all(row["text_embedding"] != [9.0] for row in rows), rows


def test_a_memory_with_no_caption_is_still_stored():
    """MemoryItem allows an empty caption, and position and time search still need the
    row. Indexing zero sentences would silently drop it from the collection."""
    rows = make_memory("sentence").rows_for(dict(ITEM, caption=""))
    assert len(rows) == 1, rows


def test_an_unknown_index_unit_is_refused():
    try:
        MilvusMemory(db_collection_name="never_built", index_unit="paragraph")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for an unknown index unit")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
