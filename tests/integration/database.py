"""Helpers to run tests against a disposable PostgreSQL database (TEST_DATABASE_URL)."""

from collections.abc import Callable, Iterator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from rag_lab.application.ports.repositories import UnitOfWork

ROOT = Path(__file__).resolve().parents[2]


def migrate(url: str) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, "head")


def postgres_uow_factory(url: str) -> Iterator[Callable[[], UnitOfWork]]:
    """Migrate, hand out unit-of-work factories, then wipe every table."""
    migrate(url)
    engine = create_db_engine(url)
    try:
        yield lambda: SqlAlchemyUnitOfWork(engine)
    finally:
        with engine.begin() as connection:
            connection.execute(text("TRUNCATE collections CASCADE"))
        engine.dispose()
