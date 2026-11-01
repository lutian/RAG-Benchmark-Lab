"""Golden questions and benchmark runs, stored by every persistence adapter."""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.application.errors import ConflictError
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.services.ingestion import IngestionService
from rag_lab.domain.chunking import ChunkingConfig, ChunkingStrategy
from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    BenchmarkRun,
    Evidence,
    GoldenQuestion,
    QueryResult,
    QuestionOrigin,
    QuestionStatus,
    summarize,
)
from rag_lab.domain.retrieval import SearchMethod
from rag_lab.domain.spans import Span

CONFIG = ChunkingConfig(ChunkingStrategy.FIXED, 200, 0)


@pytest.fixture
def ingestion(uow_factory: Callable[[], UnitOfWork]) -> IngestionService:
    return IngestionService(uow_factory=uow_factory, parser=ParserRegistry())


def seeded(ingestion: IngestionService, name: str) -> tuple[UUID, UUID]:
    collection = ingestion.create_collection(name, (CONFIG,))
    document = ingestion.ingest_document(collection.id, "a.txt", b"Rates rose. Rivers flow.")
    return collection.id, document.document.id


def test_golden_questions_round_trip_with_ordered_evidence(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection_id, document_id = seeded(ingestion, "alpha")
    question = GoldenQuestion.create(
        collection_id,
        "What rose?",
        (Evidence(document_id, Span(0, 11)), Evidence(document_id, Span(12, 24))),
        QuestionOrigin.SYNTHETIC,
    )

    with uow_factory() as uow:
        uow.golden.add(question)
        uow.commit()
    with uow_factory() as uow:
        stored = uow.golden.get(collection_id, question.id)

    assert stored == question


def test_golden_questions_update_filter_delete_and_stay_isolated(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    alpha, document = seeded(ingestion, "beta")
    other, _ = seeded(ingestion, "gamma")
    evidence = (Evidence(document, Span(0, 5)),)
    first = GoldenQuestion.create(alpha, "Q1?", evidence, QuestionOrigin.MANUAL)
    second = GoldenQuestion.create(alpha, "Q2?", evidence, QuestionOrigin.MANUAL)

    with uow_factory() as uow:
        uow.golden.add(first)
        uow.golden.add(second)
        uow.golden.update(first.with_text("Q1 edited?").approve())
        uow.commit()
    with uow_factory() as uow:
        approved = uow.golden.list(alpha, QuestionStatus.APPROVED)
        everything = uow.golden.list(alpha)
        assert uow.golden.list(other) == []
        assert uow.golden.get(other, first.id) is None
        assert uow.golden.delete(other, second.id) is False
        assert uow.golden.delete(alpha, second.id) is True
        uow.commit()
    with uow_factory() as uow:
        remaining = uow.golden.list(alpha)

    assert [q.text for q in approved] == ["Q1 edited?"]
    assert [q.text for q in everything] == ["Q1 edited?", "Q2?"]
    assert [q.id for q in remaining] == [first.id]


def test_benchmark_runs_round_trip_and_list_newest_first(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection_id, _ = seeded(ingestion, "delta")
    config = BenchmarkConfig((CONFIG.key,), (SearchMethod.HYBRID,), ks=(1, 5), generation=True)
    results = (
        QueryResult(uuid4(), CONFIG.key, SearchMethod.HYBRID, {1: 0.0, 5: 1.0}, 0.5, 2, 12.5,
                    3.0, abstained=False, citations_valid=True, prompt_tokens=100,
                    completion_tokens=20),
    )  # fmt: skip

    def run(created: datetime) -> BenchmarkRun:
        return BenchmarkRun(
            id=uuid4(),
            collection_id=collection_id,
            created_at=created,
            config=config,
            environment={"embedding": "stub (4d)"},
            question_count=1,
            results=results,
            summaries=summarize(results, config),
        )

    older, newer = run(datetime(2026, 1, 1, tzinfo=UTC)), run(datetime(2026, 2, 1, tzinfo=UTC))
    with uow_factory() as uow:
        uow.runs.add(older)
        uow.runs.add(newer)
        uow.commit()
    with uow_factory() as uow:
        assert uow.runs.get(collection_id, older.id) == older
        assert [r.id for r in uow.runs.list(collection_id)] == [newer.id, older.id]
        assert uow.runs.get(uuid4(), older.id) is None


def test_evidence_must_belong_to_the_question_collection_and_update_never_inserts(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    alpha, _ = seeded(ingestion, "epsilon")
    _, foreign_document = seeded(ingestion, "zeta")
    stray = GoldenQuestion.create(
        alpha, "Q?", (Evidence(foreign_document, Span(0, 5)),), QuestionOrigin.MANUAL
    )

    with uow_factory() as uow, pytest.raises(ConflictError, match="collection"):
        uow.golden.add(stray)
    with uow_factory() as uow:
        uow.golden.update(stray)
        uow.commit()
    with uow_factory() as uow:
        assert uow.golden.list(alpha) == []


def test_update_changes_text_and_status_but_never_evidence(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection_id, document_id = seeded(ingestion, "eta")
    original = GoldenQuestion.create(
        collection_id, "Q?", (Evidence(document_id, Span(0, 5)),), QuestionOrigin.MANUAL
    )
    moved = GoldenQuestion(
        **{**{f: getattr(original, f) for f in GoldenQuestion.__dataclass_fields__},
           "text": "Q2?", "evidence": (Evidence(document_id, Span(6, 11)),)}
    ).approve()  # fmt: skip

    with uow_factory() as uow:
        uow.golden.add(original)
        uow.golden.update(moved)
        uow.commit()
    with uow_factory() as uow:
        stored = uow.golden.get(collection_id, original.id)

    assert stored is not None
    assert (stored.text, stored.status) == ("Q2?", QuestionStatus.APPROVED)
    assert stored.evidence == original.evidence


def test_runs_with_the_same_timestamp_list_by_id(
    uow_factory: Callable[[], UnitOfWork], ingestion: IngestionService
) -> None:
    collection_id, _ = seeded(ingestion, "theta")
    config = BenchmarkConfig((CONFIG.key,), (SearchMethod.VECTOR,), ks=(1,))
    created = datetime(2026, 3, 1, tzinfo=UTC)
    runs = [
        BenchmarkRun(uuid4(), collection_id, created, config, {}, 0, (), summarize((), config))
        for _ in range(3)
    ]

    with uow_factory() as uow:
        for run in runs:
            uow.runs.add(run)
        uow.commit()
    with uow_factory() as uow:
        listed = [r.id for r in uow.runs.list(collection_id)]

    assert listed == sorted((r.id for r in runs), key=str)
