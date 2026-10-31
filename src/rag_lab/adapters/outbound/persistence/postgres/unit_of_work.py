"""SQLAlchemy unit of work: one connection and one transaction per ``with`` block."""

from types import TracebackType
from typing import Self

from sqlalchemy import Connection, Engine, RootTransaction, create_engine

from rag_lab.adapters.outbound.persistence.postgres.repositories import (
    PostgresChunkRepository,
    PostgresCollectionRepository,
    PostgresDocumentRepository,
    PostgresEmbeddingRepository,
    PostgresSearchRepository,
)


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


class SqlAlchemyUnitOfWork:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._connection: Connection | None = None
        self._transaction: RootTransaction | None = None

    def __enter__(self) -> Self:
        self._connection = self._engine.connect()
        self._transaction = self._connection.begin()
        self.collections = PostgresCollectionRepository(self._connection)
        self.documents = PostgresDocumentRepository(self._connection)
        self.chunks = PostgresChunkRepository(self._connection)
        self.embeddings = PostgresEmbeddingRepository(self._connection)
        self.search = PostgresSearchRepository(self._connection)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._transaction is not None and self._transaction.is_active:
                self._transaction.rollback()
        finally:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._transaction = None

    def commit(self) -> None:
        connection, transaction = self._require_open()
        transaction.commit()
        self._transaction = connection.begin()

    def rollback(self) -> None:
        connection, transaction = self._require_open()
        transaction.rollback()
        self._transaction = connection.begin()

    def _require_open(self) -> tuple[Connection, RootTransaction]:
        if self._connection is None or self._transaction is None:
            raise RuntimeError("unit of work used outside of a 'with' block")
        return self._connection, self._transaction
