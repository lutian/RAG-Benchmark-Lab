"""Retrieval model: the search methods being compared and how rankings are fused.

Every method returns a ranked list of chunks. Hybrid search does not mix raw scores
(cosine similarity and full-text rank live on unrelated scales); it fuses *ranks*
with Reciprocal Rank Fusion::

    score(chunk) = sum over rankings of 1 / (k + rank_in_that_ranking)
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rag_lab.domain.chunks import Chunk


class SearchMethod(StrEnum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the score one method gave it; higher is better."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class FusedChunk:
    """A fused hit: RRF score plus the 1-based rank each input ranking gave it."""

    chunk: Chunk
    score: float
    ranks: Mapping[str, int] = field(default_factory=dict)


def _identity(chunk: Chunk) -> tuple[object, int]:
    return (chunk.document_id, chunk.index)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Chunk]], k: int, limit: int
) -> list[FusedChunk]:
    """Fuse named rankings with RRF; ties keep the order of first appearance.

    ``k`` damps the advantage of the very first positions (60 is the value from the
    original paper and the common default).
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    scores: dict[tuple[object, int], float] = {}
    ranks: dict[tuple[object, int], dict[str, int]] = {}
    chunks: dict[tuple[object, int], Chunk] = {}
    for name, ranking in rankings.items():
        for position, chunk in enumerate(ranking, start=1):
            key = _identity(chunk)
            chunks.setdefault(key, chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (k + position)
            ranks.setdefault(key, {})[name] = position

    # dicts keep insertion order and sorted() is stable, so ties keep first appearance.
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:limit]
    return [FusedChunk(chunks[key], scores[key], ranks[key]) for key in ordered]
