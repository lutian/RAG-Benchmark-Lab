"""Ingestion use cases: collections, document upload, chunk sets.

Flow of :meth:`IngestionService.ingest_document`::

    bytes -> size guard -> parser port -> Document (reference layer)
          -> chunk with every collection config -> persist atomically
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from rag_lab.application.errors import (
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    UploadTooLargeError,
)
from rag_lab.application.ports.parsing import DocumentParser
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.domain.chunking import ChunkingConfig, chunk_document
from rag_lab.domain.chunks import Chunk
from rag_lab.domain.collections import Collection
from rag_lab.domain.documents import Document


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document: Document
    duplicate: bool
    chunk_counts: dict[str, int]


class IngestionService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        parser: DocumentParser,
        max_upload_bytes: int | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._parser = parser
        self._max_upload_bytes = max_upload_bytes

    # --- collections -------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        chunking_configs: tuple[ChunkingConfig, ...] = (),
        fulltext_language: str = "english",
    ) -> Collection:
        try:
            collection = Collection.create(name, chunking_configs, fulltext_language)
        except ValueError as error:
            raise InvalidRequestError(str(error)) from error
        with self._uow_factory() as uow:
            if uow.collections.get_by_name(collection.name):
                raise ConflictError(f"collection {collection.name!r} already exists")
            uow.collections.add(collection)
            uow.commit()
        return collection

    def list_collections(self) -> list[Collection]:
        with self._uow_factory() as uow:
            return uow.collections.list()

    def get_collection(self, collection_id: UUID) -> Collection:
        with self._uow_factory() as uow:
            return self._require_collection(uow, collection_id)

    def add_chunking_config(self, collection_id: UUID, config: ChunkingConfig) -> Collection:
        """Register a new chunk set and chunk every existing document with it."""
        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            try:
                updated = collection.with_chunking_config(config)
            except ValueError as error:
                raise ConflictError(str(error)) from error
            uow.collections.update(updated)
            for document in uow.documents.list(collection_id):
                uow.chunks.replace(
                    collection_id, config.key, document.id, chunk_document(document, config)
                )
            uow.commit()
        return updated

    # --- documents ---------------------------------------------------------------------

    def ingest_document(self, collection_id: UUID, filename: str, data: bytes) -> IngestionResult:
        if self._max_upload_bytes is not None and len(data) > self._max_upload_bytes:
            raise UploadTooLargeError(
                f"file {filename!r} exceeds the upload limit of {self._max_upload_bytes} bytes"
            )
        content_hash = hashlib.sha256(data).hexdigest()

        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            existing = uow.documents.find_by_hash(collection_id, content_hash)
            if existing is not None:
                counts = {
                    config.key: len(uow.chunks.list(collection_id, config.key, existing.id))
                    for config in collection.chunking_configs
                }
                return IngestionResult(existing, duplicate=True, chunk_counts=counts)

            parsed = self._parser.parse(filename, data)
            document = Document(
                id=uuid4(),
                collection_id=collection_id,
                filename=filename,
                media_type=parsed.media_type,
                content_hash=content_hash,
                text=parsed.text,
                elements=parsed.elements,
                metadata=dict(parsed.metadata),
            )
            uow.documents.add(document)
            counts = {}
            for config in collection.chunking_configs:
                chunks = chunk_document(document, config)
                uow.chunks.replace(collection_id, config.key, document.id, chunks)
                counts[config.key] = len(chunks)
            uow.commit()

        return IngestionResult(document, duplicate=False, chunk_counts=counts)

    def list_documents(self, collection_id: UUID) -> list[Document]:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            return uow.documents.list(collection_id)

    def get_document(self, collection_id: UUID, document_id: UUID) -> Document:
        with self._uow_factory() as uow:
            document = uow.documents.get(collection_id, document_id)
            if document is None:
                raise NotFoundError(f"document {document_id} not found in this collection")
            return document

    def list_chunks(
        self, collection_id: UUID, config_key: str, document_id: UUID | None = None
    ) -> list[Chunk]:
        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            if config_key not in {c.key for c in collection.chunking_configs}:
                raise NotFoundError(f"chunk set {config_key!r} not found in this collection")
            return uow.chunks.list(collection_id, config_key, document_id)

    @staticmethod
    def _require_collection(uow: UnitOfWork, collection_id: UUID) -> Collection:
        collection = uow.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"collection {collection_id} not found")
        return collection
