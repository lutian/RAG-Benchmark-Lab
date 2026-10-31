"""PostgreSQL-only full-text behaviour: stemming and stop words per collection language."""

import os
from collections.abc import Callable, Iterator

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy

pytestmark = pytest.mark.integration
CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)


@pytest.fixture
def uow_factory() -> Iterator[Callable[[], UnitOfWork]]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from tests.integration.database import postgres_uow_factory

    yield from postgres_uow_factory(url)


def search(factory: Callable[[], UnitOfWork], language: str, text: bytes, query: str) -> int:
    ingestion = IngestionService(uow_factory=factory, parser=ParserRegistry())
    collection = ingestion.create_collection(f"docs-{language}", (CONFIG,), language)
    ingestion.ingest_document(collection.id, "a.txt", text)
    with factory() as uow:
        return len(uow.search.fulltext(collection.id, CONFIG.key, query, 10))


def test_english_stems_terms_and_ignores_stop_words(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    assert search(uow_factory, "english", b"The bank raised its rate.", "rates raising") == 1
    assert search(uow_factory, "simple", b"The bank raised its rate.", "rates") == 0


def test_portuguese_collection_uses_portuguese_stemming(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    assert search(uow_factory, "portuguese", b"Os bancos subiram os juros.", "banco") == 1


def test_a_query_made_only_of_stop_words_matches_nothing(
    uow_factory: Callable[[], UnitOfWork],
) -> None:
    assert search(uow_factory, "english", b"The bank raised its rate.", "the of and") == 0
