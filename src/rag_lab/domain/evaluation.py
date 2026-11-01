"""Evaluation model: golden questions labelled with evidence spans, and ranking metrics.

Labels point at the *reference layer* (document + character span), never at chunk
ids, so one golden dataset scores every chunking configuration (ADR-002). A retrieved
chunk is relevant when it overlaps an evidence span of the same document.

Metrics per question, for a ranked list of chunks:

* ``recall[k]``: share of the question's evidence spans touched by the top-k chunks;
* ``reciprocal_rank``: ``1 / rank`` of the first relevant chunk within ``mrr_cutoff``
  (0 when there is none), averaged into MRR@cutoff.
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from rag_lab.domain.chunks import Chunk
from rag_lab.domain.retrieval import SearchMethod
from rag_lab.domain.spans import Span


class QuestionStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class QuestionOrigin(StrEnum):
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Where the answer lives: a span of one document's canonical text."""

    document_id: UUID
    span: Span


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: UUID
    collection_id: UUID
    text: str
    evidence: tuple[Evidence, ...]
    origin: QuestionOrigin
    status: QuestionStatus = QuestionStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        collection_id: UUID,
        text: str,
        evidence: tuple[Evidence, ...],
        origin: QuestionOrigin,
    ) -> "GoldenQuestion":
        if not evidence:
            raise ValueError("a golden question needs at least one evidence span")
        return cls(uuid4(), collection_id, _clean(text), evidence, origin)

    def with_text(self, text: str) -> "GoldenQuestion":
        return replace(self, text=_clean(text))

    def with_status(self, status: QuestionStatus) -> "GoldenQuestion":
        return replace(self, status=status)

    def approve(self) -> "GoldenQuestion":
        return self.with_status(QuestionStatus.APPROVED)

    def reject(self) -> "GoldenQuestion":
        return self.with_status(QuestionStatus.REJECTED)


def _clean(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("question text is required")
    return cleaned


def is_relevant(chunk: Chunk, evidence: Evidence) -> bool:
    return chunk.document_id == evidence.document_id and chunk.span.overlap(evidence.span) > 0


@dataclass(frozen=True, slots=True)
class RankingScores:
    recall: dict[int, float]
    reciprocal_rank: float
    # 1-based rank of the first relevant chunk in the whole list (None if absent).
    first_relevant_rank: int | None


def score_ranking(
    ranking: Sequence[Chunk],
    evidence: Sequence[Evidence],
    ks: Iterable[int] = (1, 5, 10),
    mrr_cutoff: int = 10,
) -> RankingScores:
    relevant = [any(is_relevant(c, e) for e in evidence) for c in ranking]
    first = next((rank for rank, hit in enumerate(relevant, start=1) if hit), None)
    recall = {
        k: sum(any(is_relevant(c, e) for c in ranking[:k]) for e in evidence) / len(evidence)
        for k in ks
    }
    reciprocal = 1 / first if first is not None and first <= mrr_cutoff else 0.0
    return RankingScores(recall, reciprocal, first)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """What one run measures: every chunk set x every method, on approved questions."""

    chunk_sets: tuple[str, ...]
    methods: tuple[SearchMethod, ...]
    ks: tuple[int, ...] = (1, 5, 10)
    mrr_cutoff: int = 10
    # Generation metrics call the LLM once per question x chunk set x method.
    generation: bool = False
    judge: bool = False
    answer_top_k: int = 5

    def __post_init__(self) -> None:
        if not self.chunk_sets or not self.methods:
            raise ValueError("a benchmark needs at least one chunk set and one method")
        if not self.ks or min(self.ks) <= 0 or self.mrr_cutoff <= 0 or self.answer_top_k <= 0:
            raise ValueError("ks, mrr_cutoff and answer_top_k must be positive")
        if self.judge and not self.generation:
            raise ValueError("the LLM judge needs generation enabled")

    @property
    def depth(self) -> int:
        """How many chunks to retrieve so every metric can be computed."""
        return max(*self.ks, self.mrr_cutoff)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One question answered by one strategy (chunk set + method)."""

    question_id: UUID
    chunk_set: str
    method: SearchMethod
    recall: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    latency_ms: float
    cpu_ms: float
    # Generation metrics; None when generation was not part of the run.
    abstained: bool | None = None
    citations_valid: bool | None = None
    supported: bool | None = None  # LLM judge verdict
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StrategySummary:
    chunk_set: str
    method: SearchMethod
    questions: int
    recall: dict[int, float]
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    cpu_ms_mean: float
    abstention_rate: float | None = None
    # Shares among answers that did not abstain.
    citation_valid_rate: float | None = None
    supported_rate: float | None = None
    tokens: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    id: UUID
    collection_id: UUID
    created_at: datetime
    config: BenchmarkConfig
    # Models and settings in effect (embedding model, reranker, LLM, ...).
    environment: dict[str, str]
    question_count: int
    results: tuple[QueryResult, ...]
    summaries: tuple[StrategySummary, ...]


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (0 for no values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(flags: Sequence[bool | None]) -> float | None:
    known = [flag for flag in flags if flag is not None]
    return sum(known) / len(known) if known else None


def summarize(
    results: Sequence[QueryResult], config: BenchmarkConfig
) -> tuple[StrategySummary, ...]:
    """One summary per chunk set x method, in config order."""
    summaries = []
    for chunk_set in config.chunk_sets:
        for method in config.methods:
            rows = [r for r in results if r.chunk_set == chunk_set and r.method == method]
            answered = [r for r in rows if r.abstained is False]
            tokens = [
                (r.prompt_tokens or 0) + (r.completion_tokens or 0)
                for r in rows
                if r.prompt_tokens is not None or r.completion_tokens is not None
            ]
            latencies = [r.latency_ms for r in rows]
            summaries.append(
                StrategySummary(
                    chunk_set=chunk_set,
                    method=method,
                    questions=len(rows),
                    recall={k: _mean([r.recall[k] for r in rows]) for k in config.ks},
                    mrr=_mean([r.reciprocal_rank for r in rows]),
                    latency_p50_ms=percentile(latencies, 50),
                    latency_p95_ms=percentile(latencies, 95),
                    cpu_ms_mean=_mean([r.cpu_ms for r in rows]),
                    abstention_rate=_rate([r.abstained for r in rows]),
                    citation_valid_rate=_rate([r.citations_valid for r in answered]),
                    supported_rate=_rate([r.supported for r in answered]),
                    tokens=sum(tokens) if tokens else None,
                )
            )
    return tuple(summaries)


def locate_excerpt(text: str, excerpt: str) -> Span | None:
    """Find ``excerpt`` in ``text``, treating any run of whitespace as equal.

    Lets a reviewer paste evidence copied from a PDF viewer (different line breaks)
    and still get exact offsets into the canonical text. Returns the first match.
    """
    words = excerpt.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, text)
    return Span(match.start(), match.end()) if match else None


@dataclass(frozen=True, slots=True)
class HeadToHead:
    """Per-question comparison of two methods on one chunk set, by reciprocal rank."""

    first: SearchMethod
    second: SearchMethod
    first_wins: int
    second_wins: int
    ties: int


def head_to_head(
    results: Sequence[QueryResult], chunk_set: str, first: SearchMethod, second: SearchMethod
) -> HeadToHead | None:
    """How often ``first`` ranked the evidence higher than ``second`` (None if one is missing)."""
    by_question: dict[UUID, dict[SearchMethod, float]] = {}
    for r in results:
        if r.chunk_set == chunk_set and r.method in (first, second):
            by_question.setdefault(r.question_id, {})[r.method] = r.reciprocal_rank
    pairs = [(q[first], q[second]) for q in by_question.values() if len(q) == 2]
    if not pairs:
        return None
    return HeadToHead(
        first,
        second,
        first_wins=sum(a > b for a, b in pairs),
        second_wins=sum(b > a for a, b in pairs),
        ties=sum(a == b for a, b in pairs),
    )


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    """Bad answers split by cause: was the evidence among the sources the LLM saw?"""

    bad_answers: int
    retrieval_misses: int  # evidence not in the top answer_top_k: a retrieval problem
    generation_errors: int  # evidence was there: the generation went wrong


def diagnose_failures(results: Sequence[QueryResult], answer_top_k: int) -> FailureDiagnosis:
    """A bad answer is an abstention, invalid citations, or a judge verdict of unsupported."""
    bad = [
        r
        for r in results
        if r.abstained is not None
        and (r.abstained or r.citations_valid is False or r.supported is False)
    ]
    misses = sum(r.first_relevant_rank is None or r.first_relevant_rank > answer_top_k for r in bad)
    return FailureDiagnosis(len(bad), misses, len(bad) - misses)
