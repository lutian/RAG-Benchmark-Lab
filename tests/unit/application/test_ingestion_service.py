"""Ingestion use cases, exercised through the in-memory unit of work."""

from uuid import uuid4

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import ConflictError, DocumentParsingError, NotFoundError
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy

FIXED = ChunkingConfig(ChunkingStrategy.FIXED, 40, 0)
RECURSIVE = ChunkingConfig(ChunkingStrategy.RECURSIVE, 40, 10)
MARKDOWN = b"# Intro\nRetrieval matters.\n\n# Chunking\nChunk size changes recall a lot.\n"


@pytest.fixture
def service() -> IngestionService:
    uow = InMemoryUnitOfWork()
    return IngestionService(
        uow_factory=lambda: uow, parser=ParserRegistry(), max_upload_bytes=1_000
    )


def test_ingesting_chunks_the_document_with_every_collection_config(
    service: IngestionService,
) -> None:
    collection = service.create_collection("docs", (FIXED, RECURSIVE))

    result = service.ingest_document(collection.id, "guide.md", MARKDOWN)

    assert result.duplicate is False
    assert set(result.chunk_counts) == {FIXED.key, RECURSIVE.key}
    chunks = service.list_chunks(collection.id, RECURSIVE.key, result.document.id)
    assert chunks and all(c.document_id == result.document.id for c in chunks)
    assert result.document.metadata["title"] == "Intro"


def test_ingesting_the_same_bytes_twice_returns_the_existing_document(
    service: IngestionService,
) -> None:
    collection = service.create_collection("docs", (FIXED,))
    first = service.ingest_document(collection.id, "a.md", MARKDOWN)

    second = service.ingest_document(collection.id, "copy.md", MARKDOWN)

    assert second.duplicate is True
    assert second.document.id == first.document.id
    assert len(service.list_documents(collection.id)) == 1


def test_adding_a_chunking_config_chunks_existing_documents(service: IngestionService) -> None:
    collection = service.create_collection("docs", (FIXED,))
    document = service.ingest_document(collection.id, "a.md", MARKDOWN).document

    service.add_chunking_config(collection.id, RECURSIVE)

    assert service.list_chunks(collection.id, RECURSIVE.key, document.id)
    assert [c.key for c in service.get_collection(collection.id).chunking_configs] == [
        FIXED.key,
        RECURSIVE.key,
    ]


def test_collection_names_are_unique(service: IngestionService) -> None:
    service.create_collection("docs")

    with pytest.raises(ConflictError, match="docs"):
        service.create_collection(" docs ")


def test_rejects_files_over_the_upload_limit(service: IngestionService) -> None:
    collection = service.create_collection("docs")

    with pytest.raises(DocumentParsingError, match="exceeds"):
        service.ingest_document(collection.id, "big.txt", b"x" * 1_001)


def test_unknown_collection_is_not_found(service: IngestionService) -> None:
    with pytest.raises(NotFoundError):
        service.ingest_document(uuid4(), "a.txt", b"text")


def test_unknown_chunk_set_is_not_found(service: IngestionService) -> None:
    collection = service.create_collection("docs", (FIXED,))

    with pytest.raises(NotFoundError, match="recursive-40-10"):
        service.list_chunks(collection.id, RECURSIVE.key)


def test_failed_parsing_persists_nothing(service: IngestionService) -> None:
    collection = service.create_collection("docs", (FIXED,))

    with pytest.raises(DocumentParsingError):
        service.ingest_document(collection.id, "fake.pdf", b"not a pdf")

    assert service.list_documents(collection.id) == []
