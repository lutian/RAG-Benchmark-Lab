"""PostgreSQL keeps golden evidence consistent when documents or collections are deleted."""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import Evidence, GoldenQuestion, QuestionOrigin
from rag_lab.domain.spans import Span
from tests.integration.database import migrate

pytestmark = pytest.mark.integration


@pytest.fixture
def engine() -> Iterator[Engine]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    migrate(url)
    engine = create_db_engine(url)
    yield engine
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE collections CASCADE"))
    engine.dispose()


def test_a_labelled_document_cannot_be_deleted_alone_but_its_collection_can(
    engine: Engine,
) -> None:
    ingestion = IngestionService(
        uow_factory=lambda: SqlAlchemyUnitOfWork(engine), parser=ParserRegistry()
    )
    collection = ingestion.create_collection(
        "labelled", (ChunkingConfig(ChunkingStrategy.FIXED, 50, 0),)
    )
    document = ingestion.ingest_document(collection.id, "a.txt", b"Refunds take 14 days.").document
    with SqlAlchemyUnitOfWork(engine) as uow:
        uow.golden.add(
            GoldenQuestion.create(
                collection.id, "Q?", (Evidence(document.id, Span(0, 7)),), QuestionOrigin.MANUAL
            )
        )
        uow.commit()

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text("DELETE FROM documents WHERE id = :d"), {"d": document.id})
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM collections WHERE id = :c"), {"c": collection.id})
        remaining = connection.scalar(text("SELECT count(*) FROM golden_evidence"))
    assert remaining == 0
