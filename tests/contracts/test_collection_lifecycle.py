"""Renaming and deleting collections, against every persistence adapter."""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.errors import ConflictError, NotFoundError
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    BenchmarkRun,
    Evidence,
    GoldenQuestion,
    QuestionOrigin,
)
from rag_lab.domain.retrieval import SearchMethod
from rag_lab.domain.spans import Span
from tests.fakes import FakeEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 25, 0)
TEXT = b"Vectors capture meaning. BM25 captures exact terms. Hybrid search mixes both."


@pytest.fixture
def ingestion(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def populate(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService, name: str
) -> tuple[UUID, UUID, UUID, UUID]:
    """A collection with a document, chunks, vectors, a golden question and a run."""
    collection = ingestion.create_collection(name, (CONFIG,))
    document = ingestion.ingest_document(collection.id, "a.txt", TEXT).document
    IndexingService(uow_factory=uow_factory, embedder=FakeEmbedder()).index_chunk_set(
        collection.id, CONFIG.key
    )
    question = GoldenQuestion.create(
        collection.id, "Q?", (Evidence(document.id, Span(0, 7)),), QuestionOrigin.MANUAL
    )
    config = BenchmarkConfig((CONFIG.key,), (SearchMethod.HYBRID,), ks=(1,), generation=False)
    run = BenchmarkRun(
        id=uuid4(),
        collection_id=collection.id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config=config,
        environment={},
        question_count=0,
        results=(),
        summaries=(),
    )
    with uow_factory() as uow:
        uow.golden.add(question)
        uow.runs.add(run)
        uow.commit()
    return collection.id, document.id, question.id, run.id


def test_rename_persists_and_keeps_everything_else(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection_id, document_id, _, _ = populate(uow_factory, ingestion, "alpha")

    ingestion.rename_collection(collection_id, "renamed")

    stored = ingestion.get_collection(collection_id)
    assert stored.name == "renamed"
    assert [c.key for c in stored.chunking_configs] == [CONFIG.key]
    assert stored.embedding is not None
    assert [d.id for d in ingestion.list_documents(collection_id)] == [document_id]
    with uow_factory() as uow:
        assert uow.collections.get_by_name("alpha") is None


def test_rename_to_a_name_in_use_is_a_conflict(ingestion: IngestionService) -> None:
    ingestion.create_collection("alpha", (CONFIG,))
    beta = ingestion.create_collection("beta", (CONFIG,))

    with pytest.raises(ConflictError):
        ingestion.rename_collection(beta.id, "alpha")
    assert ingestion.get_collection(beta.id).name == "beta"


def test_delete_removes_the_collection_and_all_its_data_only(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    gone, gone_document, gone_question, gone_run = populate(uow_factory, ingestion, "alpha")
    kept, kept_document, kept_question, kept_run = populate(uow_factory, ingestion, "beta")

    ingestion.delete_collection(gone)

    with pytest.raises(NotFoundError):
        ingestion.get_collection(gone)
    with uow_factory() as uow:
        assert uow.documents.get(gone, gone_document) is None
        assert uow.chunks.count(gone, CONFIG.key) == 0
        assert uow.embeddings.count(gone, CONFIG.key) == 0
        assert uow.golden.get(gone, gone_question) is None
        assert uow.runs.get(gone, gone_run) is None
        # The other collection is untouched.
        assert uow.documents.get(kept, kept_document) is not None
        assert uow.chunks.count(kept, CONFIG.key) > 0
        assert uow.embeddings.count(kept, CONFIG.key) > 0
        assert uow.golden.get(kept, kept_question) is not None
        assert uow.runs.get(kept, kept_run) is not None
    # The name can be reused.
    ingestion.create_collection("alpha", (CONFIG,))


def test_deleting_an_unknown_collection_is_not_found(ingestion: IngestionService) -> None:
    with pytest.raises(NotFoundError):
        ingestion.delete_collection(uuid4())
