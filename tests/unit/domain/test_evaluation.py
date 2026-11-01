"""Golden questions, span-overlap relevance and ranking metrics."""

from uuid import uuid4

import pytest

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    Evidence,
    GoldenQuestion,
    QueryResult,
    QuestionOrigin,
    QuestionStatus,
    is_relevant,
    percentile,
    score_ranking,
    summarize,
)
from rag_lab.domain.retrieval import SearchMethod
from rag_lab.domain.spans import Span

DOC, OTHER, COLLECTION = uuid4(), uuid4(), uuid4()


def chunk(start: int, end: int, document: object = DOC, index: int = 0) -> Chunk:
    return Chunk(
        DOC if document is DOC else OTHER,
        COLLECTION,
        index,
        Span(start, end),
        "x" * (end - start),
        (),
        (),
    )  # type: ignore[arg-type]


def test_a_chunk_is_relevant_when_it_overlaps_evidence_of_the_same_document() -> None:
    evidence = Evidence(DOC, Span(100, 200))

    assert is_relevant(chunk(150, 300), evidence)
    assert not is_relevant(chunk(200, 300), evidence)  # touching is not overlapping
    assert not is_relevant(chunk(150, 300, document=OTHER), evidence)


def test_recall_counts_evidence_spans_touched_by_the_top_k_chunks() -> None:
    evidence = (Evidence(DOC, Span(0, 10)), Evidence(DOC, Span(100, 110)))
    ranking = [chunk(50, 60), chunk(0, 20), chunk(70, 80), chunk(105, 120)]

    scores = score_ranking(ranking, evidence, ks=(1, 2, 5))

    assert scores.recall == {1: 0.0, 2: 0.5, 5: 1.0}
    assert scores.first_relevant_rank == 2
    assert scores.reciprocal_rank == 0.5


def test_reciprocal_rank_is_zero_beyond_the_cutoff_or_without_relevant_chunks() -> None:
    evidence = (Evidence(DOC, Span(0, 10)),)
    late = [chunk(50, 60, index=i) for i in range(10)] + [chunk(0, 5)]

    assert score_ranking(late, evidence, ks=(5,), mrr_cutoff=10).reciprocal_rank == 0.0
    assert score_ranking(late, evidence, ks=(5,), mrr_cutoff=10).first_relevant_rank == 11
    assert score_ranking([], evidence, ks=(5,)).first_relevant_rank is None


def test_golden_questions_validate_and_move_through_review() -> None:
    evidence = (Evidence(DOC, Span(0, 10)),)
    question = GoldenQuestion.create(COLLECTION, "  What is X?  ", evidence, QuestionOrigin.MANUAL)

    assert question.text == "What is X?"
    assert question.status is QuestionStatus.DRAFT
    assert question.approve().status is QuestionStatus.APPROVED
    assert question.reject().status is QuestionStatus.REJECTED
    assert question.with_text("What is Y?").text == "What is Y?"
    with pytest.raises(ValueError, match="question"):
        GoldenQuestion.create(COLLECTION, " ", evidence, QuestionOrigin.MANUAL)
    with pytest.raises(ValueError, match="evidence"):
        GoldenQuestion.create(COLLECTION, "Q?", (), QuestionOrigin.MANUAL)
    with pytest.raises(ValueError, match="question"):
        question.with_text("")


def result(
    method: SearchMethod, recall5: float, rr: float, latency: float, **extra: object
) -> QueryResult:
    return QueryResult(
        question_id=uuid4(),
        chunk_set="fixed-800-0",
        method=method,
        recall={1: 0.0, 5: recall5},
        reciprocal_rank=rr,
        first_relevant_rank=None,
        latency_ms=latency,
        cpu_ms=1.0,
        **extra,  # type: ignore[arg-type]
    )


def test_summaries_average_metrics_per_strategy_and_report_latency_percentiles() -> None:
    config = BenchmarkConfig(
        ("fixed-800-0",), (SearchMethod.VECTOR, SearchMethod.FULLTEXT), ks=(1, 5)
    )
    results = [
        result(SearchMethod.VECTOR, 1.0, 1.0, 10.0),
        result(SearchMethod.VECTOR, 0.0, 0.0, 30.0),
        result(SearchMethod.FULLTEXT, 1.0, 0.5, 5.0),
    ]

    vector, fulltext = summarize(results, config)

    assert (vector.method, vector.questions) == (SearchMethod.VECTOR, 2)
    assert vector.recall == {1: 0.0, 5: 0.5}
    assert vector.mrr == 0.5
    assert (vector.latency_p50_ms, vector.latency_p95_ms) == (10.0, 30.0)
    assert fulltext.mrr == 0.5
    assert vector.abstention_rate is None and vector.citation_valid_rate is None


def test_generation_rates_exclude_abstentions_from_citation_validity() -> None:
    config = BenchmarkConfig(("fixed-800-0",), (SearchMethod.HYBRID,), ks=(5,), generation=True)

    def answered(abstained: bool, valid: bool, completion_tokens: int) -> QueryResult:
        return result(
            SearchMethod.HYBRID,
            0.0 if abstained else 1.0,
            0.0 if abstained else 1.0,
            1.0,
            abstained=abstained,
            citations_valid=valid,
            prompt_tokens=10,
            completion_tokens=completion_tokens,
        )

    results = [answered(False, True, 2), answered(False, False, 2), answered(True, False, 1)]

    [summary] = summarize(results, config)

    assert summary.abstention_rate == 1 / 3
    assert summary.citation_valid_rate == 0.5
    assert summary.tokens == 35


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0


def test_benchmark_config_validation() -> None:
    assert BenchmarkConfig(("a",), (SearchMethod.VECTOR,), ks=(1, 5), mrr_cutoff=10).depth == 10
    with pytest.raises(ValueError, match="chunk set"):
        BenchmarkConfig((), (SearchMethod.VECTOR,))
    with pytest.raises(ValueError, match="judge"):
        BenchmarkConfig(("a",), (SearchMethod.VECTOR,), judge=True)
    with pytest.raises(ValueError, match="positive"):
        BenchmarkConfig(("a",), (SearchMethod.VECTOR,), ks=(0,))


def test_locate_excerpt_matches_across_whitespace_differences() -> None:
    from rag_lab.domain.evaluation import locate_excerpt

    text = "Refunds are issued\nwithin 14  days."

    assert locate_excerpt(text, "issued within 14 days") == Span(12, 34)
    assert locate_excerpt(text, "issued (within)") is None
    assert locate_excerpt(text, "   ") is None


def test_head_to_head_counts_per_question_wins_by_reciprocal_rank() -> None:
    from rag_lab.domain.evaluation import head_to_head

    q1, q2, q3 = uuid4(), uuid4(), uuid4()

    def row(question: object, method: SearchMethod, rr: float) -> QueryResult:
        return QueryResult(question, "cs", method, {5: 0.0}, rr, None, 1.0, 1.0)  # type: ignore[arg-type]

    results = [
        row(q1, SearchMethod.VECTOR, 1.0), row(q1, SearchMethod.FULLTEXT, 0.5),
        row(q2, SearchMethod.VECTOR, 0.0), row(q2, SearchMethod.FULLTEXT, 1.0),
        row(q3, SearchMethod.VECTOR, 0.5), row(q3, SearchMethod.FULLTEXT, 0.5),
    ]  # fmt: skip

    duel = head_to_head(results, "cs", SearchMethod.FULLTEXT, SearchMethod.VECTOR)

    assert duel is not None
    assert (duel.first_wins, duel.second_wins, duel.ties) == (1, 1, 1)
    assert head_to_head(results, "cs", SearchMethod.HYBRID, SearchMethod.VECTOR) is None


def test_failures_are_attributed_to_retrieval_or_generation() -> None:
    from rag_lab.domain.evaluation import diagnose_failures

    def row(rank: int | None, **extra: object) -> QueryResult:
        return QueryResult(
            uuid4(), "cs", SearchMethod.HYBRID, {5: 0.0}, 0.0, rank, 1.0, 1.0, **extra
        )  # type: ignore[arg-type]

    results = [
        row(1, abstained=False, citations_valid=True, supported=True),  # good answer
        row(None, abstained=True),  # evidence never retrieved
        row(9, abstained=False, citations_valid=True, supported=False),  # retrieved too late
        row(2, abstained=False, citations_valid=False),  # evidence shown, bad citations
        row(3),  # no generation in this row
    ]

    diagnosis = diagnose_failures(results, answer_top_k=5)

    assert (diagnosis.bad_answers, diagnosis.retrieval_misses, diagnosis.generation_errors) == (
        3,
        2,
        1,
    )
