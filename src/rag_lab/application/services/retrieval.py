"""Retrieval use case: search one chunk set with the selected method.

Flow of :meth:`RetrievalService.search`::

    validate request -> check chunk set (and, for vector/hybrid, the embedding binding
    and index coverage) -> embed the query (vector/hybrid only) -> search -> fuse (hybrid)
    -> rerank (hybrid_rerank)

Hybrid search takes the top ``candidates`` of each method and fuses them with
Reciprocal Rank Fusion. ``hybrid_rerank`` then hands the best ``rerank_candidates``
fused chunks to a reranker and reorders them by its scores. Every hit keeps the rank
each stage gave it, so a result can be explained ("full-text found it at #1, vectors
at #7, the reranker moved it from hybrid #4 to #1").
"""

import time
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from uuid import UUID

from rag_lab.application.errors import (
    ConflictError,
    EmbeddingProviderError,
    InvalidRequestError,
    NotFoundError,
    RerankerError,
)
from rag_lab.application.ports.embeddings import Embedder
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.ports.reranking import Reranker
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.retrieval import ScoredChunk, SearchMethod, reciprocal_rank_fusion

MAX_TOP_K = 100


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int
    # 1-based rank per stage ("vector", "fulltext", and "hybrid" before reranking).
    ranks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchTimings:
    embed_ms: float
    search_ms: float
    total_ms: float
    rerank_ms: float = 0.0


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
        reranker: Reranker | None = None,
        rerank_candidates: int = 30,
    ) -> None:
        if candidates <= 0 or rrf_k <= 0 or rerank_candidates <= 0:
            raise ValueError("candidates, rrf_k and rerank_candidates must be positive")
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

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
        reranks = method is SearchMethod.HYBRID_RERANK
        if reranks and self._reranker is None:
            raise InvalidRequestError("no reranker is configured (RERANKER_PROVIDER)")
        uses_vectors = method is not SearchMethod.FULLTEXT

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
            expected = self._embedder.spec.dimensions
            if len(query_vector) != expected:
                raise EmbeddingProviderError(
                    f"provider returned a query vector of dimension {len(query_vector)}, "
                    f"expected {expected}"
                )

        search_started = time.perf_counter()
        # Hybrid needs a deeper candidate pool than the final top_k to fuse well,
        # and reranking needs at least rerank_candidates fused chunks to reorder.
        fused_limit = max(top_k, self._rerank_candidates) if reranks else top_k
        depth = (
            top_k
            if method in (SearchMethod.VECTOR, SearchMethod.FULLTEXT)
            else max(fused_limit, self._candidates)
        )
        rankings: dict[str, list[ScoredChunk]] = {}
        with self._uow_factory() as uow:
            if uses_vectors:
                rankings["vector"] = uow.search.vector(
                    collection_id, config_key, query_vector, depth, document_ids
                )
            if method is not SearchMethod.VECTOR:
                rankings["fulltext"] = uow.search.fulltext(
                    collection_id, config_key, query, depth, document_ids
                )
        hits = self._rank(method, rankings, fused_limit)
        search_ms = _elapsed_ms(search_started)

        rerank_ms = 0.0
        if reranks:
            rerank_started = time.perf_counter()
            hits = self._rerank(query, hits)
            rerank_ms = _elapsed_ms(rerank_started)
        hits = hits[:top_k]

        return SearchResult(
            collection_id=collection_id,
            config_key=config_key,
            query=query,
            method=method,
            hits=hits,
            timings=SearchTimings(
                embed_ms=embed_ms,
                search_ms=search_ms,
                rerank_ms=rerank_ms,
                total_ms=_elapsed_ms(started),
            ),
        )

    def _rank(
        self, method: SearchMethod, rankings: dict[str, list[ScoredChunk]], top_k: int
    ) -> tuple[SearchHit, ...]:
        if method in (SearchMethod.HYBRID, SearchMethod.HYBRID_RERANK):
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

    def _rerank(self, query: str, hits: tuple[SearchHit, ...]) -> tuple[SearchHit, ...]:
        """Reorder fused hits by reranker score; ties keep the hybrid order."""
        assert self._reranker is not None
        if not hits:
            return hits
        scores = self._reranker.score(query, [hit.chunk.text for hit in hits])
        if len(scores) != len(hits):
            raise RerankerError(f"reranker returned {len(scores)} scores for {len(hits)} texts")
        ordered = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return tuple(
            SearchHit(hit.chunk, float(score), rank, {**hit.ranks, "hybrid": hit.rank})
            for rank, (hit, score) in enumerate(ordered, start=1)
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
