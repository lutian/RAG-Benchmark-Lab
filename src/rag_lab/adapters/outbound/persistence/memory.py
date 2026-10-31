"""In-memory unit of work: fast, dependency-free persistence for tests and demos.

It honours the same contract as the PostgreSQL adapter (verified by the shared
contract tests), including rollback when a transaction is not committed.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence, Set
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self
from uuid import UUID

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document
from rag_lab.domain.retrieval import ScoredChunk

ChunkKey = tuple[UUID, str, UUID]
VectorKey = tuple[UUID, str, UUID, int]


@dataclass
class _State:
    collections: dict[UUID, Collection] = field(default_factory=dict)
    documents: dict[UUID, Document] = field(default_factory=dict)
    chunks: dict[ChunkKey, list[Chunk]] = field(default_factory=dict)
    vectors: dict[VectorKey, tuple[str, list[float]]] = field(default_factory=dict)

    def copy(self) -> "_State":
        return _State(
            dict(self.collections),
            dict(self.documents),
            {key: list(value) for key, value in self.chunks.items()},
            dict(self.vectors),
        )


class InMemoryCollectionRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def add(self, collection: Collection) -> None:
        self._state.collections[collection.id] = collection

    def update(self, collection: Collection) -> None:
        self._state.collections[collection.id] = collection

    def get(self, collection_id: UUID) -> Collection | None:
        return self._state.collections.get(collection_id)

    def get_by_name(self, name: str) -> Collection | None:
        return next((c for c in self._state.collections.values() if c.name == name), None)

    def list(self) -> list[Collection]:
        return sorted(self._state.collections.values(), key=lambda c: c.created_at)


class InMemoryDocumentRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def add(self, document: Document) -> None:
        self._state.documents[document.id] = document

    def get(self, collection_id: UUID, document_id: UUID) -> Document | None:
        document = self._state.documents.get(document_id)
        return document if document and document.collection_id == collection_id else None

    def list(self, collection_id: UUID) -> list[Document]:
        return [d for d in self._state.documents.values() if d.collection_id == collection_id]

    def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        return next(
            (d for d in self.list(collection_id) if d.content_hash == content_hash),
            None,
        )


class InMemoryChunkRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    def replace(
        self, collection_id: UUID, config_key: str, document_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        self._state.chunks[(collection_id, config_key, document_id)] = list(chunks)
        # Replaced chunks lose their vectors, exactly like the SQL cascade.
        for key in [
            k for k in self._state.vectors if k[:3] == (collection_id, config_key, document_id)
        ]:
            del self._state.vectors[key]

    def count(self, collection_id: UUID, config_key: str) -> int:
        return len(self.list(collection_id, config_key))

    def list(
        self, collection_id: UUID, config_key: str, document_id: UUID | None = None
    ) -> list[Chunk]:
        return [
            chunk
            for (collection, key, document), chunks in self._state.chunks.items()
            if collection == collection_id
            and key == config_key
            and (document_id is None or document == document_id)
            for chunk in sorted(chunks, key=lambda c: c.index)
        ]


class InMemoryEmbeddingRepository:
    def __init__(self, state: _State) -> None:
        self._state = state

    @staticmethod
    def _key(collection_id: UUID, config_key: str, chunk: Chunk) -> VectorKey:
        return (collection_id, config_key, chunk.document_id, chunk.index)

    def pending(self, collection_id: UUID, config_key: str, limit: int) -> list[Chunk]:
        chunks = InMemoryChunkRepository(self._state).list(collection_id, config_key)
        missing = [
            c for c in chunks if self._key(collection_id, config_key, c) not in self._state.vectors
        ]
        return missing[:limit]

    def save(
        self,
        collection_id: UUID,
        config_key: str,
        model: str,
        items: Sequence[tuple[Chunk, Sequence[float]]],
    ) -> None:
        for chunk, vector in items:
            self._state.vectors[self._key(collection_id, config_key, chunk)] = (
                model,
                list(vector),
            )

    def count(self, collection_id: UUID, config_key: str) -> int:
        return sum(1 for key in self._state.vectors if key[:2] == (collection_id, config_key))

    def delete_collection(self, collection_id: UUID) -> None:
        for key in [k for k in self._state.vectors if k[0] == collection_id]:
            del self._state.vectors[key]


class InMemorySearchRepository:
    """Brute-force search with the same ranking contract as PostgreSQL.

    Full-text matching lowercases and splits on word characters; it ignores the
    collection language (no stemming or stop words). Contract tests only rely on
    behaviour both adapters share.
    """

    def __init__(self, state: _State) -> None:
        self._state = state

    def _chunks(
        self, collection_id: UUID, config_key: str, document_ids: Set[UUID] | None
    ) -> list[Chunk]:
        chunks = InMemoryChunkRepository(self._state).list(collection_id, config_key)
        return [c for c in chunks if document_ids is None or c.document_id in document_ids]

    def vector(
        self,
        collection_id: UUID,
        config_key: str,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        scored = []
        for chunk in self._chunks(collection_id, config_key, document_ids):
            stored = self._state.vectors.get(
                InMemoryEmbeddingRepository._key(collection_id, config_key, chunk)
            )
            if stored is not None:
                scored.append(ScoredChunk(chunk, _cosine(query_vector, stored[1])))
        return sorted(scored, key=lambda s: s.score, reverse=True)[:limit]

    def fulltext(
        self,
        collection_id: UUID,
        config_key: str,
        query: str,
        limit: int,
        document_ids: Set[UUID] | None = None,
    ) -> list[ScoredChunk]:
        terms = set(_words(query))
        scored = []
        for chunk in self._chunks(collection_id, config_key, document_ids):
            words = _words(chunk.text)
            counts = Counter(words)
            hits = sum(counts[term] for term in terms)
            if hits:
                # Same length normalisation as ts_rank_cd(..., 1): divide by 1 + log(length).
                scored.append(ScoredChunk(chunk, hits / (1 + math.log(len(words)))))
        return sorted(scored, key=lambda s: s.score, reverse=True)[:limit]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class InMemoryUnitOfWork:
    """Keeps a committed state; each ``with`` block works on a copy until commit."""

    def __init__(self) -> None:
        self._committed = _State()
        self._bind(self._committed)

    def _bind(self, state: _State) -> None:
        self._working = state
        self.collections = InMemoryCollectionRepository(state)
        self.documents = InMemoryDocumentRepository(state)
        self.chunks = InMemoryChunkRepository(state)
        self.embeddings = InMemoryEmbeddingRepository(state)
        self.search = InMemorySearchRepository(state)

    def __enter__(self) -> Self:
        self._bind(self._committed.copy())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.rollback()

    def commit(self) -> None:
        self._committed = self._working
        self._bind(self._committed.copy())

    def rollback(self) -> None:
        self._bind(self._committed)
