"""In-memory unit of work: fast, dependency-free persistence for tests and demos.

It honours the same contract as the PostgreSQL adapter (verified by the shared
contract tests), including rollback when a transaction is not committed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self
from uuid import UUID

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document

ChunkKey = tuple[UUID, str, UUID]


@dataclass
class _State:
    collections: dict[UUID, Collection] = field(default_factory=dict)
    documents: dict[UUID, Document] = field(default_factory=dict)
    chunks: dict[ChunkKey, list[Chunk]] = field(default_factory=dict)

    def copy(self) -> "_State":
        return _State(
            dict(self.collections),
            dict(self.documents),
            {key: list(value) for key, value in self.chunks.items()},
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
