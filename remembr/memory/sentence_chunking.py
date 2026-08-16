"""Sentence-level retrieval chunks derived from caption memory items.

The caption JSON remains the source of truth for a temporal segment.  This module
only gives that parent item several text vectors: one per useful sentence, all
carrying the parent's time and pose when they are inserted into Milvus.
"""

from __future__ import annotations

import os
import re
from typing import Iterable


SENTENCE_ID_SEPARATOR = "::sentence::"
MIN_SENTENCE_WORDS = 3
_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“])')
_CHUNK_CACHE = {}


def split_caption_sentences(caption: str, min_words: int = MIN_SENTENCE_WORDS) -> list[str]:
    """Split the generated English captions conservatively at sentence endings.

    The caption models emit plain prose, so a deliberately small rule is enough and
    avoids adding a tokenizer dependency.  Very short fragments are retrieval noise;
    if filtering would remove the entire non-empty caption, keep it as one chunk so a
    memory segment never disappears.
    """
    normalized = " ".join((caption or "").split())
    if not normalized:
        return []
    sentences = [part.strip() for part in _BOUNDARY.split(normalized)]
    kept = [sentence for sentence in sentences if len(sentence.split()) >= min_words]
    return kept or [normalized]


def sentence_record_id(parent_id: str, sentence_index: int) -> str:
    """Return a unique Milvus row id that still exposes its parent caption id."""
    if SENTENCE_ID_SEPARATOR in parent_id:
        raise ValueError(f"parent id contains reserved separator: {parent_id!r}")
    return f"{parent_id}{SENTENCE_ID_SEPARATOR}{sentence_index}"


def parent_id_from_record_id(record_id: str) -> str | None:
    """Read the parent id from a sentence row, or None for an unchunked row."""
    if not isinstance(record_id, str) or SENTENCE_ID_SEPARATOR not in record_id:
        return None
    return record_id.rsplit(SENTENCE_ID_SEPARATOR, 1)[0]


def dedupe_parent_documents(documents: Iterable, limit: int, require_parent_ids: bool = True):
    """Keep the highest-ranked sentence for each parent until ``limit`` parents.

    ``documents`` must already be ordered by vector similarity.  Keeping the first
    occurrence therefore keeps each parent's best matching sentence.
    """
    selected = []
    seen = set()
    for document in documents:
        record_id = (getattr(document, "metadata", None) or {}).get("id")
        parent_id = parent_id_from_record_id(record_id)
        if parent_id is None:
            if require_parent_ids:
                raise ValueError(f"sentence retrieval result has no parent id: {record_id!r}")
            parent_id = record_id if record_id is not None else id(document)
        if parent_id in seen:
            continue
        seen.add(parent_id)
        selected.append(document)
        if len(selected) >= limit:
            break
    return selected


def build_sentence_chunk_index(captions: list[dict], embedder) -> dict[str, list[dict]]:
    """Embed all caption sentences in one batch and group them by parent id."""
    texts = []
    locations = []
    for caption in captions:
        parent_id = str(caption["id"])
        for sentence_index, text in enumerate(split_caption_sentences(caption["caption"])):
            texts.append(text)
            locations.append((parent_id, sentence_index))

    embeddings = embedder.embed_documents(texts) if texts else []
    if len(embeddings) != len(texts):
        raise ValueError(
            f"embedder returned {len(embeddings)} vectors for {len(texts)} sentences"
        )

    chunks = {str(caption["id"]): [] for caption in captions}
    for text, embedding, (parent_id, sentence_index) in zip(texts, embeddings, locations):
        chunks[parent_id].append({
            "caption": text,
            "sentence_index": sentence_index,
            "text_embedding": embedding,
        })
    return chunks


def cached_sentence_chunk_index(captions_path: str, captions: list[dict], embedder):
    """Build sentence vectors once per caption file during an eval process."""
    stat = os.stat(captions_path)
    model_name = getattr(embedder, "model_name", type(embedder).__name__)
    key = (os.path.realpath(captions_path), stat.st_mtime_ns, stat.st_size, model_name)
    if key not in _CHUNK_CACHE:
        _CHUNK_CACHE[key] = build_sentence_chunk_index(captions, embedder)
    return _CHUNK_CACHE[key]
