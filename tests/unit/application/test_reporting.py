"""The Markdown report summarises a run against the project targets."""

from datetime import UTC, datetime
from uuid import uuid4

from rag_lab.application.reporting import markdown_report, missed_questions
from rag_lab.domain.evaluation import BenchmarkConfig, BenchmarkRun, QueryResult, summarize
from rag_lab.domain.retrieval import SearchMethod

Q1, Q2 = uuid4(), uuid4()


def make_run(generation: bool = False) -> BenchmarkRun:
    config = BenchmarkConfig(
        ("recursive-800-120",), (SearchMethod.VECTOR, SearchMethod.FULLTEXT), generation=generation
    )

    def row(q: object, method: SearchMethod, rank: int | None, **extra: object) -> QueryResult:
        hit = 1.0 if rank is not None and rank <= 5 else 0.0
        return QueryResult(q, "recursive-800-120", method, {1: hit, 5: hit, 10: hit},  # type: ignore[arg-type]
                           1 / rank if rank else 0.0, rank, 12.0, 3.0, **extra)  # type: ignore[arg-type]  # fmt: skip

    gen = {"abstained": False, "citations_valid": True} if generation else {}
    results = [
        row(Q1, SearchMethod.VECTOR, 1, **gen),
        row(Q2, SearchMethod.VECTOR, None, **({"abstained": True} if generation else {})),
        row(Q1, SearchMethod.FULLTEXT, 2, **gen),
        row(Q2, SearchMethod.FULLTEXT, 1, **gen),
    ]
    return BenchmarkRun(uuid4(), uuid4(), datetime(2026, 11, 1, tzinfo=UTC), config,
                        {"embedding": "nomic-embed-text (768d)"}, 2, tuple(results),
                        summarize(results, config))  # fmt: skip


def test_report_has_setup_matrix_with_targets_and_duels() -> None:
    report = markdown_report(make_run(), "Handbook")

    assert report.startswith("# Benchmark report — Handbook")
    assert "2 approved questions (target 50)" in report
    assert "| embedding | `nomic-embed-text (768d)` |" in report
    assert "**Best strategy:** `recursive-800-120` + `fulltext`" in report
    assert "| `recursive-800-120` | `vector` | 50.0% | 50.0% ❌ | 50.0% | 0.500 ❌ |" in report
    assert "| `recursive-800-120` | `fulltext` | 100.0% | 100.0% ✅ | 100.0% | 0.750 ✅ |" in report
    assert "Does lexical beat vectors? (`fulltext` vs `vector`) | 1 | 1 | 0 |" in report
    assert "## Generation" not in report


def test_report_includes_generation_and_the_failure_diagnosis() -> None:
    report = markdown_report(make_run(generation=True), "Handbook")

    assert "## Generation" in report
    assert "| `recursive-800-120` | `vector` | 50.0% | 100.0% ✅ |" in report
    assert "1 bad answers" in report
    assert "**1** had no evidence" in report


def test_missed_questions_lists_questions_no_strategy_found() -> None:
    from dataclasses import replace

    run = make_run()
    only_q2_vector = tuple(
        r for r in run.results if r.question_id == Q2 and r.method is SearchMethod.VECTOR
    )

    missed = missed_questions(replace(run, results=only_q2_vector), {str(Q2): "Where?"})

    assert missed == ["Where?"]
