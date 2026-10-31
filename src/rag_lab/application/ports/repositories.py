"""Persistence ports. Every read that returns documents or chunks takes a collection id:
the isolation boundary is part of the interface, not an optional filter.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from rag_lab.application.ports.embeddings import EmbeddingRepository
from rag_lab.application.ports.search import SearchRepository
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document


class CollectionRepository(Protocol):
    def add(self, collection: Collection) -> None: ...
    def update(self, collection: Collection) -> None: ...
    def get(self, collection_id: UUID) -> Collection | None: ...
    def get_by_name(self, name: str) -> Collection | None: ...
    def list(self) -> list[Collection]: ...


class DocumentRepository(Protocol):
    def add(self, document: Document) -> None: ...
    def get(self, collection_id: UUID, document_id: UUID) -> Document | None: ...
    def list(self, collection_id: UUID) -> list[Document]: ...
    def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None: ...


class ChunkRepository(Protocol):
    def replace(
        self, collection_id: UUID, config_key: str, document_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        """Store the chunks of one document for one chunk set, replacing previous ones."""
        ...

    def list(
        self, collection_id: UUID, config_key: str, document_id: UUID | None = None
    ) -> list[Chunk]:
        """Chunks of a chunk set, ordered by document then chunk index."""
        ...

    def count(self, collection_id: UUID, config_key: str) -> int: ...


class UnitOfWork(Protocol):
    """Transaction boundary: changes are visible only after :meth:`commit`.

    Leaving the ``with`` block without committing rolls everything back.
    """

    @property
    def collections(self) -> CollectionRepository: ...
    @property
    def documents(self) -> DocumentRepository: ...
    @property
    def chunks(self) -> ChunkRepository: ...
    @property
    def embeddings(self) -> EmbeddingRepository: ...
    @property
    def search(self) -> SearchRepository: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...
    def rollback(self) -> None: ...
