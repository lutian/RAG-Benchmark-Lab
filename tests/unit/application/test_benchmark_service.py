"""Benchmark use case: every approved question x chunk set x method, scored and stored."""

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.errors import InvalidRequestError, NotFoundError
from rag_lab.application.services.answering import AnswerService
from rag_lab.application.services.benchmark import BenchmarkService
from rag_lab.application.services.golden import GoldenDatasetService
from rag_lab.application.services.indexing import IndexingService
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.application.services.retrieval import RetrievalService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import BenchmarkConfig, QuestionStatus
from rag_lab.domain.retrieval import SearchMethod
from tests.fakes import ConceptEmbedder, FakeChatModel, FakeReranker

SMALL = ChunkingConfig(ChunkingStrategy.FIXED, 60, 0)
LARGE = ChunkingConfig(ChunkingStrategy.FIXED, 400, 0)
CONCEPTS = {"refund": 0, "refunds": 0, "money": 0, "shipping": 1, "delivery": 1, "days": 2}
TEXT = (
    b"Refunds are paid back within fourteen days of the request. "
    b"Shipping takes three to five business days for standard delivery. "
    b"Gift cards never expire and can be used in every store."
)
ALL_METHODS = tuple(SearchMethod)


def chat_reply(prompt: str) -> str:
    if "Answer to check" in prompt:
        return "SUPPORTED: the source states it."
    return "It takes fourteen days [1]."


class Lab:
    def __init__(self) -> None:
        self.uow = InMemoryUnitOfWork()
        self.embedder = ConceptEmbedder(CONCEPTS, dimensions=3)
        self.reranker = FakeReranker(prefer="refund")
        self.chat = FakeChatModel(chat_reply)
        factory = lambda: self.uow  # noqa: E731
        ingestion = IngestionService(uow_factory=factory, parser=ParserRegistry())
        self.collection = ingestion.create_collection("docs", (SMALL, LARGE))
        self.document = ingestion.ingest_document(self.collection.id, "faq.txt", TEXT).document
        for config in (SMALL, LARGE):
            IndexingService(uow_factory=factory, embedder=self.embedder).index_chunk_set(
                self.collection.id, config.key
            )
        retrieval = RetrievalService(
            uow_factory=factory, embedder=self.embedder, reranker=self.reranker
        )
        golden = GoldenDatasetService(uow_factory=factory, chat=self.chat)
        refunds = golden.add_manual(
            self.collection.id,
            "How long do refunds take?",
            self.document.id,
            "within fourteen days",
        )
        shipping = golden.add_manual(
            self.collection.id, "How long is shipping?", self.document.id, "three to five business"
        )
        golden.add_manual(self.collection.id, "Draft?", self.document.id, "Gift cards")
        for question in (refunds, shipping):
            golden.update(self.collection.id, question.id, status=QuestionStatus.APPROVED)
        self.service = BenchmarkService(
            uow_factory=factory,
            retrieval=retrieval,
            answering=AnswerService(retrieval=retrieval, chat=self.chat),
            chat=self.chat,
            environment={"embedding": "concepts (3d)"},
        )


def config(**overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "chunk_sets": (SMALL.key, LARGE.key),
        "methods": ALL_METHODS,
        "ks": (1, 5),
        "mrr_cutoff": 10,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)  # type: ignore[arg-type]


def test_runs_the_full_matrix_on_approved_questions_and_stores_the_run() -> None:
    lab = Lab()

    run = lab.service.run(lab.collection.id, config())

    assert run.question_count == 2
    assert len(run.results) == 2 * 2 * len(ALL_METHODS)
    assert [(s.chunk_set, s.method) for s in run.summaries] == [
        (cs, m) for cs in (SMALL.key, LARGE.key) for m in ALL_METHODS
    ]
    assert all(r.latency_ms >= 0 and r.cpu_ms >= 0 for r in run.results)
    assert run.environment == {"embedding": "concepts (3d)"}
    assert lab.service.runs(lab.collection.id) == [run]
    assert lab.service.get(lab.collection.id, run.id) == run


def test_relevance_is_scored_from_evidence_overlap() -> None:
    lab = Lab()

    run = lab.service.run(lab.collection.id, config(methods=(SearchMethod.FULLTEXT,)))

    large = next(s for s in run.summaries if s.chunk_set == LARGE.key)
    # One 400-char chunk holds the whole document: every question finds it at rank 1.
    assert large.recall == {1: 1.0, 5: 1.0}
    assert large.mrr == 1.0


def test_warm_up_runs_once_per_method_before_measuring() -> None:
    lab = Lab()

    lab.service.run(
        lab.collection.id, config(chunk_sets=(SMALL.key,), methods=(SearchMethod.HYBRID_RERANK,))
    )

    # 1 warm-up + 2 measured questions.
    assert len(lab.reranker.calls) == 3


def test_generation_and_judge_metrics() -> None:
    lab = Lab()

    run = lab.service.run(
        lab.collection.id,
        config(
            chunk_sets=(LARGE.key,), methods=(SearchMethod.HYBRID,), generation=True, judge=True
        ),
    )

    [summary] = run.summaries
    assert summary.abstention_rate == 0.0
    assert summary.citation_valid_rate == 1.0
    assert summary.supported_rate == 1.0
    assert summary.tokens is not None and summary.tokens > 0
    assert all(r.supported is True for r in run.results)


def test_rejects_runs_without_approved_questions_or_with_unknown_chunk_sets() -> None:
    lab = Lab()
    with lab.uow as uow:
        for question in uow.golden.list(lab.collection.id, QuestionStatus.APPROVED):
            uow.golden.update(question.with_status(QuestionStatus.DRAFT))
        uow.commit()
    other = Lab()

    with pytest.raises(InvalidRequestError, match="approved"):
        lab.service.run(lab.collection.id, config())
    with pytest.raises(NotFoundError, match="chunk set"):
        other.service.run(other.collection.id, config(chunk_sets=("nope",)))
