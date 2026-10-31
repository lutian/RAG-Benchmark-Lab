"""Composition root: the only place that knows which adapter implements which port.

Inbound adapters (API, CLI, tests) ask for a :class:`Container` and never build
adapters themselves, so swapping PostgreSQL for the in-memory store, or one
embeddings provider for another, is one line here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag_lab.adapters.outbound.embeddings.openai_compatible import OpenAICompatibleEmbedder
from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from rag_lab.application.ports.embeddings import Embedder
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.application.services.retrieval import RetrievalService
from rag_lab.config import Settings
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    ingestion: IngestionService
    indexing: IndexingService
    retrieval: RetrievalService
    default_chunking_configs: tuple[ChunkingConfig, ...]


def default_chunking_configs(settings: Settings) -> tuple[ChunkingConfig, ...]:
    """One config per strategy with the default size/overlap: the baseline comparison."""
    return tuple(
        ChunkingConfig(strategy, settings.default_chunk_size, settings.default_chunk_overlap)
        for strategy in ChunkingStrategy
    )


def build_embedder(settings: Settings) -> Embedder:
    """Embeddings provider from ``EMBEDDING_*`` settings (no network call until used)."""
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def build_container(settings: Settings) -> Container:
    """Production wiring: PostgreSQL persistence and the configured embeddings provider."""
    engine = create_db_engine(settings.database_url)
    return _assemble(settings, lambda: SqlAlchemyUnitOfWork(engine), build_embedder(settings))


def build_in_memory_container(settings: Settings, embedder: Embedder | None = None) -> Container:
    """Wiring without a database, for tests and quick demos."""
    uow = InMemoryUnitOfWork()
    return _assemble(settings, lambda: uow, embedder or build_embedder(settings))


def _assemble(
    settings: Settings, uow_factory: Callable[[], UnitOfWork], embedder: Embedder
) -> Container:
    ingestion = IngestionService(
        uow_factory=uow_factory,
        parser=ParserRegistry(),
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    indexing = IndexingService(
        uow_factory=uow_factory, embedder=embedder, batch_size=settings.embedding_batch_size
    )
    retrieval = RetrievalService(
        uow_factory=uow_factory,
        embedder=embedder,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.hybrid_rrf_k,
    )
    return Container(settings, ingestion, indexing, retrieval, default_chunking_configs(settings))
