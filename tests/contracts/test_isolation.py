"""Isolation contract: nothing from one collection is ever visible from another.

Runs against every persistence adapter (see ``uow_factory`` in conftest).
"""

from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.errors import NotFoundError
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy

CONFIG = ChunkingConfig(ChunkingStrategy.RECURSIVE, 60, 10)
SAME_CONTENT = b"Shared paragraph about vector search.\n\nAnother paragraph about BM25."


@pytest.fixture
def service(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def test_identical_documents_in_two_collections_stay_separate(service: IngestionService) -> None:
    alpha = service.create_collection("alpha", (CONFIG,))
    beta = service.create_collection("beta", (CONFIG,))
    doc_alpha = service.ingest_document(alpha.id, "a.txt", SAME_CONTENT).document
    doc_beta = service.ingest_document(beta.id, "b.txt", SAME_CONTENT).document

    assert doc_alpha.id != doc_beta.id
    assert [d.id for d in service.list_documents(alpha.id)] == [doc_alpha.id]
    assert [d.id for d in service.list_documents(beta.id)] == [doc_beta.id]
    assert {c.collection_id for c in service.list_chunks(alpha.id, CONFIG.key)} == {alpha.id}
    assert {c.document_id for c in service.list_chunks(beta.id, CONFIG.key)} == {doc_beta.id}


def test_documents_cannot_be_read_through_another_collection(service: IngestionService) -> None:
    alpha = service.create_collection("alpha", (CONFIG,))
    beta = service.create_collection("beta", (CONFIG,))
    doc_beta = service.ingest_document(beta.id, "b.txt", SAME_CONTENT).document

    with pytest.raises(NotFoundError):
        service.get_document(alpha.id, doc_beta.id)
    assert service.list_chunks(alpha.id, CONFIG.key, doc_beta.id) == []


def test_round_trip_preserves_reference_layer(service: IngestionService) -> None:
    collection = service.create_collection("gamma", (CONFIG,))
    markdown = b"# Title\nBody text.\n## Sub\nMore body.\n"
    stored = service.ingest_document(collection.id, "t.md", markdown).document

    loaded = service.get_document(collection.id, stored.id)

    assert loaded.text == stored.text
    assert loaded.elements == stored.elements
    assert dict(loaded.metadata) == dict(stored.metadata)
    for chunk in service.list_chunks(collection.id, CONFIG.key, stored.id):
        assert chunk.text == loaded.text[chunk.span.start : chunk.span.end]
