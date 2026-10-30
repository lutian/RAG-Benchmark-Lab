"""Transaction contract shared by every persistence adapter."""

from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.errors import ConflictError, DocumentParsingError
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.collections import Collection

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 50, 0)


@pytest.fixture
def service(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def test_uncommitted_changes_are_discarded(uow_factory: Callable[[], UnitOfWork]) -> None:
    with uow_factory() as uow:
        uow.collections.add(Collection.create("ghost"))

    with uow_factory() as uow:
        assert uow.collections.get_by_name("ghost") is None


def test_duplicate_collection_name_is_a_conflict(service: IngestionService) -> None:
    service.create_collection("dup", (CONFIG,))

    with pytest.raises(ConflictError):
        service.create_collection("dup")


def test_failed_ingestion_leaves_no_partial_document(service: IngestionService) -> None:
    collection = service.create_collection("atomic", (CONFIG,))

    with pytest.raises(DocumentParsingError):
        service.ingest_document(collection.id, "broken.pdf", b"%PDF-1.7 garbage")

    assert service.list_documents(collection.id) == []


def test_chunking_configs_keep_their_order(service: IngestionService) -> None:
    second = ChunkingConfig(ChunkingStrategy.STRUCTURE, 300, 0)
    collection = service.create_collection("ordered", (CONFIG,))
    service.add_chunking_config(collection.id, second)

    assert service.get_collection(collection.id).chunking_configs == (CONFIG, second)
