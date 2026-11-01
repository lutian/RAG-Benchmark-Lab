"""Migrations keep existing data usable: 0003 backfills full-text vectors for old chunks."""

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import Engine, text

from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from tests.integration.database import alembic_config, migrate

pytestmark = pytest.mark.integration


@pytest.fixture
def engine() -> Iterator[Engine]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    migrate(url)
    engine = create_db_engine(url)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE collections CASCADE"))
    try:
        yield engine
    finally:
        migrate(url)  # always leave the schema at head for the other tests
        with engine.begin() as connection:
            connection.execute(text("TRUNCATE collections CASCADE"))
        engine.dispose()


def test_upgrade_to_0003_makes_existing_chunks_searchable(engine: Engine) -> None:
    url = os.environ["TEST_DATABASE_URL"]
    command.downgrade(alembic_config(url), "0002")
    collection, document = uuid4(), uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO collections (id, name, created_at) VALUES (:c, 'old', now())"),
            {"c": collection},
        )
        connection.execute(
            text("INSERT INTO chunking_configs VALUES (:c, 'fixed-200-0', 0, 'fixed', 200, 0)"),
            {"c": collection},
        )
        connection.execute(
            text(
                "INSERT INTO documents VALUES (:d, :c, 'a.txt', 'text/plain', 'h', "
                "'Banks raised rates', '{}', now())"
            ),
            {"c": collection, "d": document},
        )
        connection.execute(
            text(
                "INSERT INTO chunks (collection_id, config_key, document_id, chunk_index, "
                "start_offset, end_offset, text, page_numbers, section_path) "
                "VALUES (:c, 'fixed-200-0', :d, 0, 0, 18, 'Banks raised rates', '{}', '{}')"
            ),
            {"c": collection, "d": document},
        )

    command.upgrade(alembic_config(url), "head")

    with SqlAlchemyUnitOfWork(engine) as uow:
        hits = uow.search.fulltext(collection, "fixed-200-0", "bank rate", 10)
        language = uow.collections.get(collection)
    assert [hit.chunk.text for hit in hits] == ["Banks raised rates"]
    assert language is not None and language.fulltext_language == "english"
