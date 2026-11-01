"""Alembic environment: URL from the ``sqlalchemy.url`` option or the app settings."""

from alembic import context
from sqlalchemy import create_engine

from rag_lab.adapters.outbound.persistence.postgres.tables import metadata
from rag_lab.config import get_settings

config = context.config
url = config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
