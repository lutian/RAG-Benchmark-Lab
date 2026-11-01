"""Answer use case: the full question flow.

    question -> RetrievalService (selected method, filters, reranking) -> numbered
    context -> ChatModel -> citations parsed and checked against the sources

Two abstention paths are kept apart on purpose, because they point to different
culprits when diagnosing a bad answer:

* ``no_evidence_retrieved``: retrieval returned nothing, the LLM is not called;
* ``model_abstained``: sources were retrieved but the LLM judged them insufficient.
"""

import time
from collections.abc import Set
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from rag_lab.application.ports.generation import ChatModel, Completion
from rag_lab.application.prompts import answer_messages
from rag_lab.application.services.retrieval import (
    RetrievalService,
    SearchHit,
    SearchTimings,
)
from rag_lab.domain.answers import CitationCheck, check_citations, is_abstention
from rag_lab.domain.retrieval import SearchMethod

AbstentionReason = Literal["no_evidence_retrieved", "model_abstained"]


@dataclass(frozen=True, slots=True)
class Citation:
    """A source the answer cites: its number in the prompt and the retrieved hit."""

    number: int
    hit: SearchHit


@dataclass(frozen=True, slots=True)
class AnswerTimings:
    retrieval: SearchTimings
    generation_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    method: SearchMethod
    config_key: str
    text: str
    abstained: bool
    abstention_reason: AbstentionReason | None
    sources: tuple[SearchHit, ...]
    citations: tuple[Citation, ...]
    citation_check: CitationCheck
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    timings: AnswerTimings


class AnswerService:
    def __init__(self, retrieval: RetrievalService, chat: ChatModel) -> None:
        self._retrieval = retrieval
        self._chat = chat

    def answer(
        self,
        collection_id: UUID,
        config_key: str,
        question: str,
        method: SearchMethod = SearchMethod.HYBRID,
        top_k: int = 5,
        document_ids: Set[UUID] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        result = self._retrieval.search(
            collection_id, config_key, question, method, top_k=top_k, document_ids=document_ids
        )
        sources = result.hits
        if not sources:
            return self._answer(
                result.query, method, config_key, "", sources, None, result.timings, 0.0, started
            )

        generation_started = time.perf_counter()
        completion = self._chat.complete(
            answer_messages(result.query, [hit.chunk for hit in sources])
        )
        generation_ms = _elapsed_ms(generation_started)
        return self._answer(
            result.query,
            method,
            config_key,
            completion.text.strip(),
            sources,
            completion,
            result.timings,
            generation_ms,
            started,
        )

    @staticmethod
    def _answer(
        question: str,
        method: SearchMethod,
        config_key: str,
        text: str,
        sources: tuple[SearchHit, ...],
        completion: Completion | None,
        retrieval: SearchTimings,
        generation_ms: float,
        started: float,
    ) -> Answer:
        reason: AbstentionReason | None = None
        if completion is None:
            reason = "no_evidence_retrieved"
        elif is_abstention(text):
            reason = "model_abstained"
        check = check_citations("" if reason else text, len(sources))
        valid_numbers = [n for n in check.cited if n not in check.invalid]
        return Answer(
            question=question,
            method=method,
            config_key=config_key,
            text=text,
            abstained=reason is not None,
            abstention_reason=reason,
            sources=sources,
            citations=tuple(Citation(n, sources[n - 1]) for n in valid_numbers),
            citation_check=check,
            model=completion.model if completion else None,
            prompt_tokens=completion.prompt_tokens if completion else None,
            completion_tokens=completion.completion_tokens if completion else None,
            timings=AnswerTimings(retrieval, generation_ms, _elapsed_ms(started)),
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
