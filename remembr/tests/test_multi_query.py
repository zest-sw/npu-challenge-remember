"""Offline checks for multi-query reciprocal-rank fusion."""

from types import SimpleNamespace

import pytest

from remembr.memory.multi_query import reciprocal_rank_fusion
from remembr.memory.sentence_chunking import sentence_record_id


def doc(record_id, text=None):
    return SimpleNamespace(metadata={"id": record_id}, page_content=text or record_id)


def test_rrf_rewards_memories_supported_by_multiple_queries():
    a, b, c, d = (doc(name) for name in "abcd")
    fused = reciprocal_rank_fusion([[a, b, c], [d, b, c]], limit=3)
    assert [item.metadata["id"] for item in fused] == ["b", "c", "a"]


def test_rrf_keeps_final_candidate_count_and_stable_ties():
    a, b, c = (doc(name) for name in "abc")
    fused = reciprocal_rank_fusion([[a, b, c]], limit=2)
    assert fused == [a, b]


def test_sentence_rows_vote_once_per_parent_and_return_distinct_parents():
    a0 = doc(sentence_record_id("a", 0), "A weak sentence")
    a1 = doc(sentence_record_id("a", 1), "A strongest sentence")
    b0 = doc(sentence_record_id("b", 0), "B sentence")
    c0 = doc(sentence_record_id("c", 0), "C sentence")

    fused = reciprocal_rank_fusion(
        [[b0, a0], [a1, c0]],
        limit=3,
        dedupe_text_parents=True,
    )

    assert [item.metadata["id"].split("::sentence::")[0] for item in fused] == [
        "a", "b", "c"
    ]
    # Query two ranks A first, so its matching sentence is the representative shown
    # to the selector rather than the weaker sentence first seen in query one.
    assert fused[0].page_content == "A strongest sentence"


def test_rrf_rejects_missing_ids_and_invalid_limits():
    with pytest.raises(ValueError, match="record id"):
        reciprocal_rank_fusion([[SimpleNamespace(metadata={}, page_content="x")]], limit=1)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([], limit=0)
