"""Composition root: the only place that knows which adapter implements which port.

Inbound adapters (API, CLI, tests) ask for a :class:`Container` and never build
adapters themselves, so swapping PostgreSQL for the in-memory store is one line here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.config import Settings
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    ingestion: IngestionService
    default_chunking_configs: tuple[ChunkingConfig, ...]


def default_chunking_configs(settings: Settings) -> tuple[ChunkingConfig, ...]:
    """One config per strategy with the default size/overlap: the baseline comparison."""
    return tuple(
        ChunkingConfig(strategy, settings.default_chunk_size, settings.default_chunk_overlap)
        for strategy in ChunkingStrategy
    )


def build_container(settings: Settings) -> Container:
    """Production wiring: PostgreSQL persistence."""
    engine = create_db_engine(settings.database_url)
    return _assemble(settings, lambda: SqlAlchemyUnitOfWork(engine))


def build_in_memory_container(settings: Settings) -> Container:
    """Wiring without external services, for tests and quick demos."""
    uow = InMemoryUnitOfWork()
    return _assemble(settings, lambda: uow)


def _assemble(settings: Settings, uow_factory: Callable[[], UnitOfWork]) -> Container:
    ingestion = IngestionService(
        uow_factory=uow_factory,
        parser=ParserRegistry(),
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    return Container(settings, ingestion, default_chunking_configs(settings))
