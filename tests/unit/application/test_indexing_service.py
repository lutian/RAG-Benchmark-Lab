"""Indexing use cases: embed chunk sets, bind the collection, resume after failures."""

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import ConflictError, EmbeddingProviderError, NotFoundError
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.embeddings import EmbeddingSpec
from tests.fakes import FakeEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 20, 0)
TEXT = b"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"


@pytest.fixture
def uow() -> InMemoryUnitOfWork:
    return InMemoryUnitOfWork()


@pytest.fixture
def ingestion(uow: InMemoryUnitOfWork) -> IngestionService:
    return IngestionService(uow_factory=lambda: uow, parser=ParserRegistry())


def indexing(uow: InMemoryUnitOfWork, embedder: FakeEmbedder, batch: int = 2) -> IndexingService:
    return IndexingService(uow_factory=lambda: uow, embedder=embedder, batch_size=batch)


def seeded(ingestion: IngestionService) -> str:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    return str(collection.id)


def test_indexes_every_chunk_in_batches_and_binds_the_collection(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    total = len(ingestion.list_chunks(collection.id, CONFIG.key))
    embedder = FakeEmbedder()

    report = indexing(uow, embedder).index_chunk_set(collection.id, CONFIG.key)

    assert report.embedded == total
    assert report.total == total
    assert report.already_indexed == 0
    assert all(len(batch) <= 2 for batch in embedder.batches)
    assert ingestion.get_collection(collection.id).embedding == EmbeddingSpec("fake-embed", 8)


def test_reindexing_only_embeds_new_chunks(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    service = indexing(uow, FakeEmbedder())
    first = service.index_chunk_set(collection.id, CONFIG.key)
    ingestion.ingest_document(collection.id, "b.txt", b"brand new document text here")

    second = service.index_chunk_set(collection.id, CONFIG.key)

    assert second.already_indexed == first.total
    assert second.embedded == second.total - first.total
    assert second.embedded > 0


def test_status_reports_pending_chunks_per_chunk_set(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    other = ChunkingConfig(ChunkingStrategy.RECURSIVE, 30, 0)
    collection = ingestion.create_collection("docs", (CONFIG, other))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    service = indexing(uow, FakeEmbedder())
    service.index_chunk_set(collection.id, CONFIG.key)

    status = service.status(collection.id)

    assert status.embedding == EmbeddingSpec("fake-embed", 8)
    by_key = {s.config_key: s for s in status.chunk_sets}
    assert by_key[CONFIG.key].embedded == by_key[CONFIG.key].total > 0
    assert by_key[other.key].embedded == 0
    assert by_key[other.key].total > 0


def test_a_failed_batch_keeps_earlier_batches_and_the_next_run_resumes(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)

    with pytest.raises(EmbeddingProviderError):
        indexing(uow, FakeEmbedder(fail_on_call=2)).index_chunk_set(collection.id, CONFIG.key)
    resumed = indexing(uow, FakeEmbedder()).index_chunk_set(collection.id, CONFIG.key)

    assert resumed.already_indexed == 2
    assert resumed.embedded == resumed.total - 2


def test_vectors_with_the_wrong_dimension_are_rejected_and_not_stored(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)

    with pytest.raises(EmbeddingProviderError, match="dimension"):
        indexing(uow, FakeEmbedder(wrong_dimensions=True)).index_chunk_set(
            collection.id, CONFIG.key
        )

    assert indexing(uow, FakeEmbedder()).status(collection.id).chunk_sets[0].embedded == 0


def test_a_different_embedding_model_is_a_conflict(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    indexing(uow, FakeEmbedder()).index_chunk_set(collection.id, CONFIG.key)

    with pytest.raises(ConflictError, match="rebuild"):
        indexing(uow, FakeEmbedder(model="other")).index_chunk_set(collection.id, CONFIG.key)


def test_rebuild_drops_old_vectors_and_rebinds(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    indexing(uow, FakeEmbedder()).index_chunk_set(collection.id, CONFIG.key)

    report = indexing(uow, FakeEmbedder(model="other", dimensions=4)).index_chunk_set(
        collection.id, CONFIG.key, rebuild=True
    )

    assert report.already_indexed == 0
    assert report.embedded == report.total
    assert ingestion.get_collection(collection.id).embedding == EmbeddingSpec("other", 4)


def test_unknown_chunk_set_is_not_found(
    uow: InMemoryUnitOfWork, ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("docs", (CONFIG,))

    with pytest.raises(NotFoundError):
        indexing(uow, FakeEmbedder()).index_chunk_set(collection.id, "fixed-1-0")
