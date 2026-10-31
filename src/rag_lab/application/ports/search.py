"""Search port: ranked lookups over one chunk set, always scoped by collection."""

from collections.abc import Sequence, Set
from typing import Protocol
from uuid import UUID

from rag_lab.domain.retrieval import ScoredChunk


class SearchRepository(Protocol):
    """Each method returns at most ``limit`` chunks, best first.

    ``document_ids`` restricts the search to those documents; ``None`` means all.
    """

    def vector(
        self,
        collection_id: UUID,
        config_key: str,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        """Exact nearest neighbours by cosine similarity (score in [-1, 1])."""
        ...

    def fulltext(
        self,
        collection_id: UUID,
        config_key: str,
        query: str,
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        """Chunks sharing at least one term with the query, ranked by term coverage.

        Terms are normalised with the collection's full-text language.
        """
        ...
