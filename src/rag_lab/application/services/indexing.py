"""Indexing use cases: embed chunk sets and keep the collection bound to one model.

Flow of :meth:`IndexingService.index_chunk_set`::

    bind collection to the embedder spec (or rebuild) -> commit
    loop: read next batch of pending chunks -> embed (outside any transaction)
          -> validate dimensions -> save batch -> commit

Committing per batch makes indexing resumable: if the provider fails half-way,
the next run only embeds what is still pending.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from rag_lab.application.errors import ConflictError, EmbeddingProviderError, NotFoundError
from rag_lab.application.ports.embeddings import Embedder
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection, EmbeddingMismatchError
from rag_lab.domain.embeddings import EmbeddingSpec


@dataclass(frozen=True, slots=True)
class IndexingReport:
    collection_id: UUID
    config_key: str
    embedding: EmbeddingSpec
    total: int
    already_indexed: int
    embedded: int
    seconds: float


@dataclass(frozen=True, slots=True)
class ChunkSetStatus:
    config_key: str
    total: int
    embedded: int


@dataclass(frozen=True, slots=True)
class IndexStatus:
    collection_id: UUID
    embedding: EmbeddingSpec | None
    chunk_sets: tuple[ChunkSetStatus, ...]


class IndexingService:
    def __init__(
        self, uow_factory: Callable[[], UnitOfWork], embedder: Embedder, batch_size: int = 64
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._batch_size = batch_size

    def index_chunk_set(
        self, collection_id: UUID, config_key: str, rebuild: bool = False
    ) -> IndexingReport:
        started = time.perf_counter()
        spec = self._embedder.spec

        with self._uow_factory() as uow:
            collection = self._require_chunk_set(uow, collection_id, config_key)
            if rebuild:
                uow.embeddings.delete_collection(collection_id)
                collection = collection.unbind_embedding()
            try:
                uow.collections.update(collection.bind_embedding(spec))
            except EmbeddingMismatchError as error:
                raise ConflictError(str(error)) from error
            total = uow.chunks.count(collection_id, config_key)
            already_indexed = uow.embeddings.count(collection_id, config_key)
            uow.commit()

        embedded = 0
        # Upper bound on iterations: protects against a store that never drains.
        for _ in range(math.ceil(max(total - already_indexed, 0) / self._batch_size) + 1):
            with self._uow_factory() as uow:
                batch = uow.embeddings.pending(collection_id, config_key, self._batch_size)
            if not batch:
                break
            vectors = self._embed(batch, spec)
            with self._uow_factory() as uow:
                uow.embeddings.save(
                    collection_id, config_key, spec.model, list(zip(batch, vectors, strict=True))
                )
                uow.commit()
            embedded += len(batch)

        return IndexingReport(
            collection_id=collection_id,
            config_key=config_key,
            embedding=spec,
            total=total,
            already_indexed=already_indexed,
            embedded=embedded,
            seconds=round(time.perf_counter() - started, 3),
        )

    def status(self, collection_id: UUID) -> IndexStatus:
        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            chunk_sets = tuple(
                ChunkSetStatus(
                    config_key=config.key,
                    total=uow.chunks.count(collection_id, config.key),
                    embedded=uow.embeddings.count(collection_id, config.key),
                )
                for config in collection.chunking_configs
            )
        return IndexStatus(collection_id, collection.embedding, chunk_sets)

    def _embed(self, batch: list[Chunk], spec: EmbeddingSpec) -> list[list[float]]:
        vectors = self._embedder.embed_documents([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise EmbeddingProviderError(
                f"provider returned {len(vectors)} vectors for {len(batch)} texts"
            )
        wrong = {len(v) for v in vectors if len(v) != spec.dimensions}
        if wrong:
            raise EmbeddingProviderError(
                f"provider returned vectors of dimension {sorted(wrong)}, expected "
                f"{spec.dimensions}; check EMBEDDING_DIMENSIONS for model {spec.model!r}"
            )
        return vectors

    @staticmethod
    def _require_collection(uow: UnitOfWork, collection_id: UUID) -> Collection:
        collection = uow.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"collection {collection_id} not found")
        return collection

    def _require_chunk_set(
        self, uow: UnitOfWork, collection_id: UUID, config_key: str
    ) -> Collection:
        collection = self._require_collection(uow, collection_id)
        if config_key not in {c.key for c in collection.chunking_configs}:
            raise NotFoundError(f"chunk set {config_key!r} not found in this collection")
        return collection
