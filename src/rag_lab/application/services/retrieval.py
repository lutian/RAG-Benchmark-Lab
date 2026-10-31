"""Retrieval use case: search one chunk set with the selected method.

Flow of :meth:`RetrievalService.search`::

    validate request -> check chunk set (and, for vector/hybrid, the embedding binding
    and index coverage) -> embed the query (vector/hybrid only) -> search -> fuse (hybrid)

Hybrid search takes the top ``candidates`` of each method and fuses them with
Reciprocal Rank Fusion. Every hit keeps the rank each method gave it, so a result
can be explained ("full-text found it at #1, vectors at #7").
"""

import time
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from uuid import UUID

from rag_lab.application.errors import ConflictError, InvalidRequestError, NotFoundError
from rag_lab.application.ports.embeddings import Embedder
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.retrieval import ScoredChunk, SearchMethod, reciprocal_rank_fusion

MAX_TOP_K = 100


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int
    # 1-based rank per underlying method ("vector", "fulltext").
    ranks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchTimings:
    embed_ms: float
    search_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    collection_id: UUID
    config_key: str
    query: str
    method: SearchMethod
    hits: tuple[SearchHit, ...]
    timings: SearchTimings


class RetrievalService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        embedder: Embedder,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> None:
        if candidates <= 0 or rrf_k <= 0:
            raise ValueError("candidates and rrf_k must be positive")
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k

    def search(
        self,
        collection_id: UUID,
        config_key: str,
        query: str,
        method: SearchMethod,
        top_k: int = 10,
        document_ids: Set[UUID] | None = None,
    ) -> SearchResult:
        started = time.perf_counter()
        query = query.strip()
        if not query:
            raise InvalidRequestError("query must not be blank")
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidRequestError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        uses_vectors = method in (SearchMethod.VECTOR, SearchMethod.HYBRID)

        with self._uow_factory() as uow:
            collection = self._require_chunk_set(uow, collection_id, config_key)
            if uses_vectors:
                self._require_vectors(uow, collection, config_key)

        embed_ms = 0.0
        query_vector: list[float] = []
        if uses_vectors:
            embed_started = time.perf_counter()
            query_vector = self._embedder.embed_query(query)
            embed_ms = _elapsed_ms(embed_started)

        search_started = time.perf_counter()
        # Hybrid needs a deeper candidate pool than the final top_k to fuse well.
        depth = max(top_k, self._candidates) if method is SearchMethod.HYBRID else top_k
        rankings: dict[str, list[ScoredChunk]] = {}
        with self._uow_factory() as uow:
            if uses_vectors:
                rankings["vector"] = uow.search.vector(
                    collection_id, config_key, query_vector, depth, document_ids
                )
            if method in (SearchMethod.FULLTEXT, SearchMethod.HYBRID):
                rankings["fulltext"] = uow.search.fulltext(
                    collection_id, config_key, query, depth, document_ids
                )
        hits = self._rank(method, rankings, top_k)
        search_ms = _elapsed_ms(search_started)

        return SearchResult(
            collection_id=collection_id,
            config_key=config_key,
            query=query,
            method=method,
            hits=hits,
            timings=SearchTimings(
                embed_ms=embed_ms, search_ms=search_ms, total_ms=_elapsed_ms(started)
            ),
        )

    def _rank(
        self, method: SearchMethod, rankings: dict[str, list[ScoredChunk]], top_k: int
    ) -> tuple[SearchHit, ...]:
        if method is SearchMethod.HYBRID:
            fused = reciprocal_rank_fusion(
                {name: [s.chunk for s in ranking] for name, ranking in rankings.items()},
                k=self._rrf_k,
                limit=top_k,
            )
            return tuple(
                SearchHit(hit.chunk, hit.score, rank, dict(hit.ranks))
                for rank, hit in enumerate(fused, start=1)
            )
        name, ranking = next(iter(rankings.items()))
        return tuple(
            SearchHit(scored.chunk, scored.score, rank, {name: rank})
            for rank, scored in enumerate(ranking[:top_k], start=1)
        )

    def _require_chunk_set(
        self, uow: UnitOfWork, collection_id: UUID, config_key: str
    ) -> Collection:
        collection = uow.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"collection {collection_id} not found")
        if config_key not in {c.key for c in collection.chunking_configs}:
            raise NotFoundError(f"chunk set {config_key!r} not found in this collection")
        return collection

    def _require_vectors(self, uow: UnitOfWork, collection: Collection, config_key: str) -> None:
        """Vector scores are only meaningful against a complete index of the same model."""
        if collection.embedding is None:
            raise ConflictError(
                f"collection {collection.name!r} is not indexed; index chunk set "
                f"{config_key!r} first or use full-text search"
            )
        spec = self._embedder.spec
        if collection.embedding != spec:
            raise ConflictError(
                f"collection {collection.name!r} is indexed with {collection.embedding.label} "
                f"but the configured embedder is {spec.label}; rebuild the index"
            )
        total = uow.chunks.count(collection.id, config_key)
        embedded = uow.embeddings.count(collection.id, config_key)
        if embedded < total:
            raise ConflictError(
                f"chunk set {config_key!r} is not fully indexed ({embedded}/{total} chunks); "
                "run indexing before vector or hybrid search"
            )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
