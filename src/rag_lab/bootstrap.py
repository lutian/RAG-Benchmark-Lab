"""Composition root: the only place that knows which adapter implements which port.

Inbound adapters (API, CLI, tests) ask for a :class:`Container` and never build
adapters themselves, so swapping PostgreSQL for the in-memory store, or one
embeddings provider for another, is one line here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag_lab.adapters.outbound.embeddings.openai_compatible import OpenAICompatibleEmbedder
from rag_lab.adapters.outbound.generation.openai_compatible import OpenAICompatibleChatModel
from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.adapters.outbound.persistence.postgres import SqlAlchemyUnitOfWork, create_db_engine
from rag_lab.adapters.outbound.reranking.cross_encoder import CrossEncoderReranker
from rag_lab.adapters.outbound.reranking.http_api import HttpReranker
from rag_lab.application.ports.embeddings import Embedder
from rag_lab.application.ports.generation import ChatModel
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.ports.reranking import Reranker
from rag_lab.application.services.answering import AnswerService
from rag_lab.application.services.benchmark import BenchmarkService
from rag_lab.application.services.golden import GoldenDatasetService
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
    answering: AnswerService
    golden: GoldenDatasetService
    benchmark: BenchmarkService
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


def build_chat_model(settings: Settings) -> ChatModel:
    """Chat LLM from ``LLM_*`` settings; only the configured generation options are sent."""
    return OpenAICompatibleChatModel(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        options=settings.llm_generation_options(),
    )


def build_reranker(settings: Settings) -> Reranker:
    """Reranker from ``RERANKER_*`` settings (a local model is loaded on first use)."""
    if settings.reranker_provider == "api":
        return HttpReranker(
            base_url=settings.reranker_base_url,
            model=settings.reranker_model,
            api_key=settings.reranker_api_key.get_secret_value(),
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return CrossEncoderReranker(settings.reranker_model)


def environment(settings: Settings, reranker: Reranker, chat: ChatModel) -> dict[str, str]:
    """Models and retrieval settings recorded with every benchmark run."""
    generation = ", ".join(f"{k}={v}" for k, v in settings.llm_generation_options().items())
    return {
        "embedding": f"{settings.embedding_model} ({settings.embedding_dimensions}d)",
        "reranker": f"{settings.reranker_provider}: {reranker.name}",
        "llm": chat.name + (f" ({generation})" if generation else ""),
        "hybrid": f"RRF k={settings.hybrid_rrf_k}, {settings.retrieval_candidates} candidates",
        "rerank_candidates": str(settings.rerank_candidates),
    }


def build_container(settings: Settings) -> Container:
    """Production wiring: PostgreSQL persistence and the configured providers."""
    engine = create_db_engine(settings.database_url)
    return _assemble(
        settings,
        lambda: SqlAlchemyUnitOfWork(engine),
        build_embedder(settings),
        build_reranker(settings),
        build_chat_model(settings),
    )


def build_in_memory_container(
    settings: Settings,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chat: ChatModel | None = None,
) -> Container:
    """Wiring without a database, for tests and quick demos."""
    uow = InMemoryUnitOfWork()
    return _assemble(
        settings,
        lambda: uow,
        embedder or build_embedder(settings),
        reranker or build_reranker(settings),
        chat or build_chat_model(settings),
    )


def _assemble(
    settings: Settings,
    uow_factory: Callable[[], UnitOfWork],
    embedder: Embedder,
    reranker: Reranker,
    chat: ChatModel,
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
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )
    answering = AnswerService(retrieval=retrieval, chat=chat)
    golden = GoldenDatasetService(uow_factory=uow_factory, chat=chat)
    benchmark = BenchmarkService(
        uow_factory=uow_factory,
        retrieval=retrieval,
        answering=answering,
        chat=chat,
        environment=environment(settings, reranker, chat),
    )
    return Container(
        settings,
        ingestion,
        indexing,
        retrieval,
        answering,
        golden,
        benchmark,
        default_chunking_configs(settings),
    )
