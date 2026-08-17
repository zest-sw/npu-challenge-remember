"""Rank fusion for several text queries over the same visual memory.

Each query produces its own similarity ranking.  Reciprocal-rank fusion (RRF)
combines those rankings without pretending that embedding distances from separate
queries are directly comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from remembr.memory.sentence_chunking import parent_id_from_record_id


RRF_CONSTANT = 60


def _document_key(document, dedupe_text_parents: bool):
    record_id = (getattr(document, "metadata", None) or {}).get("id")
    if record_id is None:
        raise ValueError("text retrieval result has no record id")
    if not dedupe_text_parents:
        return record_id

    parent_id = parent_id_from_record_id(record_id)
    if parent_id is None:
        raise ValueError(f"sentence retrieval result has no parent id: {record_id!r}")
    return parent_id


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence],
    limit: int,
    *,
    dedupe_text_parents: bool = False,
    constant: int = RRF_CONSTANT,
    weights: Sequence[float] | None = None,
):
    """Fuse query rankings and return at most ``limit`` distinct parent memories.

    A sentence index may contain several rows for one 3-second caption.  Such rows
    count as one parent per query, otherwise a verbose caption would receive several
    votes from the same query and occupy candidate slots more than once.  The best
    ranked sentence becomes that parent's representative text.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if constant < 0:
        raise ValueError(f"RRF constant must be non-negative, got {constant}")

    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights length must match rankings length: {len(weights)} != {len(rankings)}"
        )
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"RRF weights must be positive, got {list(weights)!r}")

    scores = {}
    representatives = {}
    representative_ranks = {}
    first_seen = {}
    seen_order = 0

    for ranking, weight in zip(rankings, weights):
        seen_in_query = set()
        parent_rank = 0
        for document in ranking:
            key = _document_key(document, dedupe_text_parents)
            if key in seen_in_query:
                continue
            seen_in_query.add(key)
            parent_rank += 1
            contribution = weight / (constant + parent_rank)
            scores[key] = scores.get(key, 0.0) + contribution
            if key not in representatives:
                representatives[key] = document
                representative_ranks[key] = parent_rank
                first_seen[key] = seen_order
                seen_order += 1
            elif parent_rank < representative_ranks[key]:
                representatives[key] = document
                representative_ranks[key] = parent_rank

    ordered_keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [representatives[key] for key in ordered_keys[:limit]]
