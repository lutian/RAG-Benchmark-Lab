"""Search contract, run against every persistence adapter."""

from collections.abc import Callable

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from tests.fakes import ConceptEmbedder

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)
OTHER = ChunkingConfig(ChunkingStrategy.FIXED, 300, 0)
CONCEPTS = {"automobile": 0, "car": 0, "bank": 1, "savings": 2, "river": 3}
EMBEDDER = ConceptEmbedder(CONCEPTS, dimensions=4)


@pytest.fixture
def ingestion(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def test_full_text_is_scoped_to_collection_and_chunk_set(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    alpha = ingestion.create_collection("alpha", (CONFIG, OTHER))
    beta = ingestion.create_collection("beta", (CONFIG,))
    ingestion.ingest_document(alpha.id, "a.txt", b"The bank approved the savings plan.")
    ingestion.ingest_document(beta.id, "b.txt", b"The bank approved the savings plan.")

    with uow_factory() as uow:
        hits = uow.search.fulltext(alpha.id, CONFIG.key, "bank", 10)

    assert len(hits) == 1
    assert hits[0].chunk.collection_id == alpha.id
    assert hits[0].score > 0


def test_full_text_matches_any_query_term_and_ranks_coverage_first(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("gamma", (CONFIG,))
    ingestion.ingest_document(collection.id, "one.txt", b"The bank is closed today.")
    both = ingestion.ingest_document(collection.id, "two.txt", b"The bank savings rate rose.")
    ingestion.ingest_document(collection.id, "three.txt", b"A quiet river at dawn.")

    with uow_factory() as uow:
        hits = uow.search.fulltext(collection.id, CONFIG.key, "bank savings unicorn", 10)
        none = uow.search.fulltext(collection.id, CONFIG.key, "unicorn", 10)

    assert len(hits) == 2
    assert hits[0].chunk.document_id == both.document.id
    assert none == []


def test_vector_search_returns_nearest_chunks_of_the_collection_only(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    alpha = ingestion.create_collection("delta", (CONFIG,))
    beta = ingestion.create_collection("epsilon", (CONFIG,))
    car = ingestion.ingest_document(alpha.id, "car.txt", b"An automobile on the road.")
    ingestion.ingest_document(alpha.id, "river.txt", b"A river in the valley.")
    ingestion.ingest_document(beta.id, "car.txt", b"An automobile on the road.")
    for collection in (alpha, beta):
        IndexingService(uow_factory=uow_factory, embedder=EMBEDDER).index_chunk_set(
            collection.id, CONFIG.key
        )

    with uow_factory() as uow:
        hits = uow.search.vector(alpha.id, CONFIG.key, EMBEDDER.embed_query("car"), 10)

    assert [hit.chunk.collection_id for hit in hits] == [alpha.id, alpha.id]
    assert hits[0].chunk.document_id == car.document.id
    assert hits[0].score == pytest.approx(1.0, abs=0.01)
    assert hits[0].score > hits[1].score


def test_document_filter_and_limit_apply_to_both_methods(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection = ingestion.create_collection("zeta", (CONFIG,))
    first = ingestion.ingest_document(collection.id, "1.txt", b"bank one")
    second = ingestion.ingest_document(collection.id, "2.txt", b"bank two")
    IndexingService(uow_factory=uow_factory, embedder=EMBEDDER).index_chunk_set(
        collection.id, CONFIG.key
    )
    query = EMBEDDER.embed_query("bank")

    with uow_factory() as uow:
        only_second = {second.document.id}
        assert [
            h.chunk.document_id
            for h in uow.search.fulltext(collection.id, CONFIG.key, "bank", 10, only_second)
        ] == [second.document.id]
        assert [
            h.chunk.document_id
            for h in uow.search.vector(collection.id, CONFIG.key, query, 10, only_second)
        ] == [second.document.id]
        assert len(uow.search.fulltext(collection.id, CONFIG.key, "bank", 1)) == 1
        assert len(uow.search.vector(collection.id, CONFIG.key, query, 1)) == 1
        assert uow.search.vector(collection.id, CONFIG.key, query, 10, set()) == []
    assert first.document.id != second.document.id
