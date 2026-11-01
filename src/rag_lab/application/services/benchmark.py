"""Benchmark use case: the chunking x retrieval matrix on the approved golden questions.

Flow of :meth:`BenchmarkService.run`::

    validate (chunk sets exist, approved questions exist)
    warm up every method once, unmeasured (loads the reranker, opens connections)
    for chunk set, method, question:
        retrieve top `depth` chunks -> wall time + CPU time -> Recall@k, reciprocal rank
        [generation] answer with `answer_top_k` sources -> abstention, citation validity
        [judge]      ask the LLM whether the answer is supported by its sources
    summarize per strategy -> store the run

Warm-up matters: the first reranked query loads a model (seconds), which would
otherwise be charged to whichever question happens to come first.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from rag_lab.application.errors import InvalidRequestError, NotFoundError
from rag_lab.application.ports.generation import ChatModel
from rag_lab.application.ports.repositories import UnitOfWork
from rag_lab.application.prompts import judge_messages, parse_verdict
from rag_lab.application.services.answering import AnswerService
from rag_lab.application.services.retrieval import RetrievalService
from rag_lab.domain.collections import Collection
from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    BenchmarkRun,
    GoldenQuestion,
    QueryResult,
    QuestionStatus,
    score_ranking,
    summarize,
)
from rag_lab.domain.retrieval import SearchMethod

Progress = Callable[[int, int], None]


class BenchmarkService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        retrieval: RetrievalService,
        answering: AnswerService,
        chat: ChatModel,
        environment: Mapping[str, str],
    ) -> None:
        self._uow_factory = uow_factory
        self._retrieval = retrieval
        self._answering = answering
        self._chat = chat
        self._environment = dict(environment)

    def run(
        self, collection_id: UUID, config: BenchmarkConfig, progress: Progress | None = None
    ) -> BenchmarkRun:
        questions = self._prepare(collection_id, config)
        self._warm_up(collection_id, config, questions[0])

        total = len(config.chunk_sets) * len(config.methods) * len(questions)
        results: list[QueryResult] = []
        for chunk_set in config.chunk_sets:
            for method in config.methods:
                for question in questions:
                    results.append(
                        self._measure(collection_id, config, chunk_set, method, question)
                    )
                    if progress is not None:
                        progress(len(results), total)

        run = BenchmarkRun(
            id=uuid4(),
            collection_id=collection_id,
            created_at=datetime.now(UTC),
            config=config,
            environment=self._environment,
            question_count=len(questions),
            results=tuple(results),
            summaries=summarize(results, config),
        )
        with self._uow_factory() as uow:
            uow.runs.add(run)
            uow.commit()
        return run

    def runs(self, collection_id: UUID) -> list[BenchmarkRun]:
        with self._uow_factory() as uow:
            self._require_collection(uow, collection_id)
            return uow.runs.list(collection_id)

    def get(self, collection_id: UUID, run_id: UUID) -> BenchmarkRun:
        with self._uow_factory() as uow:
            run = uow.runs.get(collection_id, run_id)
        if run is None:
            raise NotFoundError(f"benchmark run {run_id} not found")
        return run

    # --- steps ----------------------------------------------------------------------
    def _prepare(self, collection_id: UUID, config: BenchmarkConfig) -> list[GoldenQuestion]:
        with self._uow_factory() as uow:
            collection = self._require_collection(uow, collection_id)
            known = {c.key for c in collection.chunking_configs}
            missing = [key for key in config.chunk_sets if key not in known]
            if missing:
                raise NotFoundError(f"chunk set(s) {missing} not found in this collection")
            questions = uow.golden.list(collection_id, QuestionStatus.APPROVED)
        if not questions:
            raise InvalidRequestError(
                "the collection has no approved golden questions; review some first"
            )
        return questions

    def _warm_up(
        self, collection_id: UUID, config: BenchmarkConfig, question: GoldenQuestion
    ) -> None:
        # Also fails fast (e.g. 409 for an unindexed chunk set) before any measurement.
        for chunk_set in config.chunk_sets:
            for method in config.methods:
                self._retrieval.search(collection_id, chunk_set, question.text, method, top_k=1)

    def _measure(
        self,
        collection_id: UUID,
        config: BenchmarkConfig,
        chunk_set: str,
        method: SearchMethod,
        question: GoldenQuestion,
    ) -> QueryResult:
        wall, cpu = time.perf_counter(), time.process_time()
        result = self._retrieval.search(
            collection_id,
            chunk_set,
            question.text,
            method,
            top_k=config.depth,
        )
        latency_ms = (time.perf_counter() - wall) * 1000
        cpu_ms = (time.process_time() - cpu) * 1000
        scores = score_ranking(
            [hit.chunk for hit in result.hits], question.evidence, config.ks, config.mrr_cutoff
        )
        measured = QueryResult(
            question_id=question.id,
            chunk_set=chunk_set,
            method=result.method,
            recall=scores.recall,
            reciprocal_rank=scores.reciprocal_rank,
            first_relevant_rank=scores.first_relevant_rank,
            latency_ms=round(latency_ms, 3),
            cpu_ms=round(cpu_ms, 3),
        )
        if not config.generation:
            return measured
        return self._with_generation(collection_id, config, chunk_set, question, measured)

    def _with_generation(
        self,
        collection_id: UUID,
        config: BenchmarkConfig,
        chunk_set: str,
        question: GoldenQuestion,
        measured: QueryResult,
    ) -> QueryResult:
        answer = self._answering.answer(
            collection_id, chunk_set, question.text, measured.method, top_k=config.answer_top_k
        )
        prompt_tokens, completion_tokens = answer.prompt_tokens, answer.completion_tokens
        supported = None
        if config.judge and not answer.abstained:
            verdict = self._chat.complete(
                judge_messages(question.text, answer.text, [h.chunk for h in answer.sources])
            )
            supported = parse_verdict(verdict.text)
            prompt_tokens = (prompt_tokens or 0) + (verdict.prompt_tokens or 0)
            completion_tokens = (completion_tokens or 0) + (verdict.completion_tokens or 0)
        return replace(
            measured,
            abstained=answer.abstained,
            citations_valid=None if answer.abstained else answer.citation_check.valid,
            supported=supported,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _require_collection(uow: UnitOfWork, collection_id: UUID) -> Collection:
        collection = uow.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"collection {collection_id} not found")
        return collection
