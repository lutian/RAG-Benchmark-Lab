"""Embedding storage contract, run against every persistence adapter."""

from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.embeddings import EmbeddingSpec
from tests.fakes import FakeEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 25, 0)
TEXT = b"Vectors capture meaning. BM25 captures exact terms. Hybrid search mixes both."


@pytest.fixture
def ingestion(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def indexing(factory: Callable[[], UnitOfWork], embedder: FakeEmbedder) -> IndexingService:
    return IndexingService(uow_factory=factory, embedder=embedder, batch_size=2)


def test_embeddings_are_isolated_per_collection(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    alpha = ingestion.create_collection("alpha", (CONFIG,))
    beta = ingestion.create_collection("beta", (CONFIG,))
    ingestion.ingest_document(alpha.id, "a.txt", TEXT)
    ingestion.ingest_document(beta.id, "b.txt", TEXT)

    indexing(uow_factory, FakeEmbedder()).index_chunk_set(alpha.id, CONFIG.key)

    service = indexing(uow_factory, FakeEmbedder())
    assert service.status(alpha.id).chunk_sets[0].embedded > 0
    assert service.status(beta.id).chunk_sets[0].embedded == 0
    assert service.status(beta.id).embedding is None


def test_binding_and_progress_survive_a_round_trip(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("gamma", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)

    with pytest.raises(Exception, match="outage"):
        indexing(uow_factory, FakeEmbedder(fail_on_call=2)).index_chunk_set(
            collection.id, CONFIG.key
        )

    status = indexing(uow_factory, FakeEmbedder()).status(collection.id)
    assert status.embedding == EmbeddingSpec("fake-embed", 8)
    assert status.chunk_sets[0].embedded == 2
    assert ingestion.get_collection(collection.id).embedding == EmbeddingSpec("fake-embed", 8)


def test_rebuild_replaces_vectors_with_another_dimension(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("delta", (CONFIG,))
    ingestion.ingest_document(collection.id, "a.txt", TEXT)
    indexing(uow_factory, FakeEmbedder()).index_chunk_set(collection.id, CONFIG.key)

    report = indexing(uow_factory, FakeEmbedder(model="m2", dimensions=3)).index_chunk_set(
        collection.id, CONFIG.key, rebuild=True
    )

    assert report.embedded == report.total
    assert ingestion.get_collection(collection.id).embedding == EmbeddingSpec("m2", 3)
