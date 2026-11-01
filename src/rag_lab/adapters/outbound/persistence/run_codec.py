"""JSON encoding of benchmark runs (stored as one JSONB document, exported as files).

Kept explicit instead of ``dataclasses.asdict`` so the stored format is visible and
versioned: JSON object keys are strings, so ``recall`` keys (k) are converted back.
"""

from typing import Any
from uuid import UUID

from rag_lab.domain.evaluation import (
    BenchmarkConfig,
    BenchmarkRun,
    QueryResult,
    StrategySummary,
)
from rag_lab.domain.retrieval import SearchMethod

FORMAT_VERSION = 1


def _recall_out(recall: dict[int, float]) -> dict[str, float]:
    return {str(k): v for k, v in recall.items()}


def _recall_in(recall: dict[str, float]) -> dict[int, float]:
    return {int(k): float(v) for k, v in recall.items()}


def run_to_payload(run: BenchmarkRun) -> dict[str, Any]:
    config = run.config
    return {
        "format_version": FORMAT_VERSION,
        "config": {
            "chunk_sets": list(config.chunk_sets),
            "methods": [m.value for m in config.methods],
            "ks": list(config.ks),
            "mrr_cutoff": config.mrr_cutoff,
            "generation": config.generation,
            "judge": config.judge,
            "answer_top_k": config.answer_top_k,
        },
        "environment": dict(run.environment),
        "question_count": run.question_count,
        "results": [
            {
                "question_id": str(r.question_id),
                "chunk_set": r.chunk_set,
                "method": r.method.value,
                "recall": _recall_out(r.recall),
                "reciprocal_rank": r.reciprocal_rank,
                "first_relevant_rank": r.first_relevant_rank,
                "latency_ms": r.latency_ms,
                "cpu_ms": r.cpu_ms,
                "abstained": r.abstained,
                "citations_valid": r.citations_valid,
                "supported": r.supported,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
            }
            for r in run.results
        ],
        "summaries": [
            {
                "chunk_set": s.chunk_set,
                "method": s.method.value,
                "questions": s.questions,
                "recall": _recall_out(s.recall),
                "mrr": s.mrr,
                "latency_p50_ms": s.latency_p50_ms,
                "latency_p95_ms": s.latency_p95_ms,
                "cpu_ms_mean": s.cpu_ms_mean,
                "abstention_rate": s.abstention_rate,
                "citation_valid_rate": s.citation_valid_rate,
                "supported_rate": s.supported_rate,
                "tokens": s.tokens,
            }
            for s in run.summaries
        ],
    }


def run_from_payload(
    run_id: UUID, collection_id: UUID, created_at: Any, payload: dict[str, Any]
) -> BenchmarkRun:
    c = payload["config"]
    config = BenchmarkConfig(
        chunk_sets=tuple(c["chunk_sets"]),
        methods=tuple(SearchMethod(m) for m in c["methods"]),
        ks=tuple(c["ks"]),
        mrr_cutoff=c["mrr_cutoff"],
        generation=c["generation"],
        judge=c["judge"],
        answer_top_k=c["answer_top_k"],
    )
    results = tuple(
        QueryResult(
            question_id=UUID(r["question_id"]),
            chunk_set=r["chunk_set"],
            method=SearchMethod(r["method"]),
            recall=_recall_in(r["recall"]),
            reciprocal_rank=r["reciprocal_rank"],
            first_relevant_rank=r["first_relevant_rank"],
            latency_ms=r["latency_ms"],
            cpu_ms=r["cpu_ms"],
            abstained=r["abstained"],
            citations_valid=r["citations_valid"],
            supported=r["supported"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
        )
        for r in payload["results"]
    )
    summaries = tuple(
        StrategySummary(
            chunk_set=s["chunk_set"],
            method=SearchMethod(s["method"]),
            questions=s["questions"],
            recall=_recall_in(s["recall"]),
            mrr=s["mrr"],
            latency_p50_ms=s["latency_p50_ms"],
            latency_p95_ms=s["latency_p95_ms"],
            cpu_ms_mean=s["cpu_ms_mean"],
            abstention_rate=s["abstention_rate"],
            citation_valid_rate=s["citation_valid_rate"],
            supported_rate=s["supported_rate"],
            tokens=s["tokens"],
        )
        for s in payload["summaries"]
    )
    return BenchmarkRun(
        id=run_id,
        collection_id=collection_id,
        created_at=created_at,
        config=config,
        environment=dict(payload["environment"]),
        question_count=payload["question_count"],
        results=results,
        summaries=summaries,
    )
