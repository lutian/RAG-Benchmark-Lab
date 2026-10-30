"""Repository implementations over a SQLAlchemy connection owned by the unit of work."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import Connection, Row, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from rag_lab.adapters.outbound.persistence.postgres import tables as t
from rag_lab.application.errors import ConflictError
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document, ElementKind, MediaType, StructuralElement
from rag_lab.domain.spans import Span


def _write(connection: Connection, statement: Any, rows: Any = None) -> None:
    """Execute a write, translating constraint violations into a ConflictError."""
    try:
        with connection.begin_nested():
            connection.execute(statement, rows) if rows is not None else connection.execute(
                statement
            )
    except IntegrityError as error:
        raise ConflictError(f"conflicting data: {error.orig}") from error


class PostgresCollectionRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, collection: Collection) -> None:
        _write(
            self._connection,
            insert(t.collections).values(
                id=collection.id, name=collection.name, created_at=collection.created_at
            ),
        )
        self._save_configs(collection)

    def update(self, collection: Collection) -> None:
        self._connection.execute(
            update(t.collections)
            .where(t.collections.c.id == collection.id)
            .values(name=collection.name)
        )
        self._save_configs(collection)

    def _save_configs(self, collection: Collection) -> None:
        existing = set(
            self._connection.scalars(
                select(t.chunking_configs.c.key).where(
                    t.chunking_configs.c.collection_id == collection.id
                )
            )
        )
        rows = [
            {
                "collection_id": collection.id,
                "key": config.key,
                "position": position,
                "strategy": config.strategy.value,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
            for position, config in enumerate(collection.chunking_configs)
            if config.key not in existing
        ]
        if rows:
            _write(self._connection, insert(t.chunking_configs), rows)

    def get(self, collection_id: UUID) -> Collection | None:
        row = self._connection.execute(
            select(t.collections).where(t.collections.c.id == collection_id)
        ).first()
        return self._to_domain(row) if row else None

    def get_by_name(self, name: str) -> Collection | None:
        row = self._connection.execute(
            select(t.collections).where(t.collections.c.name == name)
        ).first()
        return self._to_domain(row) if row else None

    def list(self) -> list[Collection]:
        rows = self._connection.execute(
            select(t.collections).order_by(t.collections.c.created_at, t.collections.c.id)
        )
        return [self._to_domain(row) for row in rows]

    def _to_domain(self, row: Row[Any]) -> Collection:
        configs = self._connection.execute(
            select(t.chunking_configs)
            .where(t.chunking_configs.c.collection_id == row.id)
            .order_by(t.chunking_configs.c.position)
        )
        return Collection(
            id=row.id,
            name=row.name,
            chunking_configs=tuple(
                ChunkingConfig(ChunkingStrategy(c.strategy), c.chunk_size, c.chunk_overlap)
                for c in configs
            ),
            created_at=row.created_at,
        )


class PostgresDocumentRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def add(self, document: Document) -> None:
        _write(
            self._connection,
            insert(t.documents).values(
                id=document.id,
                collection_id=document.collection_id,
                filename=document.filename,
                media_type=document.media_type.value,
                content_hash=document.content_hash,
                text=document.text,
                metadata=dict(document.metadata),
                created_at=document.created_at,
            ),
        )
        if document.elements:
            _write(
                self._connection,
                insert(t.document_elements),
                [
                    {
                        "document_id": document.id,
                        "position": position,
                        "kind": element.kind.value,
                        "start_offset": element.span.start,
                        "end_offset": element.span.end,
                        "label": element.label,
                        "level": element.level,
                        "path": list(element.path),
                    }
                    for position, element in enumerate(document.elements)
                ],
            )

    def get(self, collection_id: UUID, document_id: UUID) -> Document | None:
        row = self._connection.execute(
            select(t.documents).where(
                t.documents.c.id == document_id,
                t.documents.c.collection_id == collection_id,
            )
        ).first()
        return self._to_domain(row) if row else None

    def list(self, collection_id: UUID) -> list[Document]:
        rows = self._connection.execute(
            select(t.documents)
            .where(t.documents.c.collection_id == collection_id)
            .order_by(t.documents.c.created_at, t.documents.c.id)
        )
        return [self._to_domain(row) for row in rows]

    def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        row = self._connection.execute(
            select(t.documents).where(
                t.documents.c.collection_id == collection_id,
                t.documents.c.content_hash == content_hash,
            )
        ).first()
        return self._to_domain(row) if row else None

    def _to_domain(self, row: Row[Any]) -> Document:
        elements = self._connection.execute(
            select(t.document_elements)
            .where(t.document_elements.c.document_id == row.id)
            .order_by(t.document_elements.c.position)
        )
        return Document(
            id=row.id,
            collection_id=row.collection_id,
            filename=row.filename,
            media_type=MediaType(row.media_type),
            content_hash=row.content_hash,
            text=row.text,
            elements=tuple(
                StructuralElement(
                    kind=ElementKind(e.kind),
                    span=Span(e.start_offset, e.end_offset),
                    label=e.label,
                    level=e.level,
                    path=tuple(e.path),
                )
                for e in elements
            ),
            metadata=dict(row.metadata),
            created_at=row.created_at,
        )


class PostgresChunkRepository:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def replace(
        self, collection_id: UUID, config_key: str, document_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        self._connection.execute(
            delete(t.chunks).where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
                t.chunks.c.document_id == document_id,
            )
        )
        if not chunks:
            return
        _write(
            self._connection,
            insert(t.chunks),
            [
                {
                    "collection_id": collection_id,
                    "config_key": config_key,
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "start_offset": chunk.span.start,
                    "end_offset": chunk.span.end,
                    "text": chunk.text,
                    "page_numbers": list(chunk.page_numbers),
                    "section_path": list(chunk.section_path),
                }
                for chunk in chunks
            ],
        )

    def list(
        self, collection_id: UUID, config_key: str, document_id: UUID | None = None
    ) -> list[Chunk]:
        query = (
            select(t.chunks)
            .join(t.documents, t.documents.c.id == t.chunks.c.document_id)
            .where(
                t.chunks.c.collection_id == collection_id,
                t.chunks.c.config_key == config_key,
            )
            .order_by(t.documents.c.created_at, t.documents.c.id, t.chunks.c.chunk_index)
        )
        if document_id is not None:
            query = query.where(t.chunks.c.document_id == document_id)
        return [
            Chunk(
                document_id=row.document_id,
                collection_id=row.collection_id,
                index=row.chunk_index,
                span=Span(row.start_offset, row.end_offset),
                text=row.text,
                page_numbers=tuple(row.page_numbers),
                section_path=tuple(row.section_path),
            )
            for row in self._connection.execute(query)
        ]
