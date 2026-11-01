"""Markdown report of a benchmark run: the artifact that makes an experiment shareable.

Sections: setup (models, settings, questions), the retrieval matrix with the
project's targets, head-to-head duels that answer the interview questions ("when
does lexical beat vectors?", "when does reranking help?"), generation quality, and
the retrieval-vs-generation diagnosis of bad answers.
"""

from collections.abc import Mapping, Sequence

from rag_lab.domain.evaluation import (
    BenchmarkRun,
    StrategySummary,
    diagnose_failures,
    head_to_head,
)
from rag_lab.domain.retrieval import SearchMethod

# Initial validation targets from the project brief (proposed, not achieved results).
TARGET_QUESTIONS = 50
TARGET_RECALL_AT_5 = 0.85
TARGET_MRR_AT_10 = 0.75
TARGET_VALID_CITATIONS = 0.95

DUELS = (
    (SearchMethod.FULLTEXT, SearchMethod.VECTOR, "Does lexical beat vectors?"),
    (SearchMethod.HYBRID, SearchMethod.VECTOR, "Does fusion help vectors?"),
    (SearchMethod.HYBRID_RERANK, SearchMethod.HYBRID, "Does reranking help hybrid?"),
)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _mark(value: float | None, target: float) -> str:
    if value is None:
        return "—"
    return f"{value:.1%} {'✅' if value >= target else '❌'}"


def best_strategy(summaries: Sequence[StrategySummary], k: int = 5) -> StrategySummary:
    """Highest Recall@k, then MRR, then the lowest median latency."""
    return max(summaries, key=lambda s: (s.recall.get(k, 0.0), s.mrr, -s.latency_p50_ms))


def markdown_report(run: BenchmarkRun, collection_name: str) -> str:
    config = run.config
    k = 5 if 5 in config.ks else max(config.ks)
    best = best_strategy(run.summaries, k)
    lines = [
        f"# Benchmark report — {collection_name}",
        "",
        f"Run `{run.id}` · {run.created_at:%Y-%m-%d %H:%M} UTC · "
        f"{run.question_count} approved questions (target {TARGET_QUESTIONS})",
        "",
        f"**Best strategy:** `{best.chunk_set}` + `{best.method}` — Recall@{k} "
        f"{best.recall.get(k, 0.0):.1%}, MRR@{config.mrr_cutoff} {best.mrr:.3f}, "
        f"p50 {best.latency_p50_ms:.1f} ms.",
        "",
        "## Setup",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        *(f"| {key} | `{value}` |" for key, value in sorted(run.environment.items())),
        f"| chunk sets | {', '.join(f'`{c}`' for c in config.chunk_sets)} |",
        f"| methods | {', '.join(f'`{m}`' for m in config.methods)} |",
        f"| generation / judge | {config.generation} / {config.judge} |",
        "",
        "## Retrieval",
        "",
        f"Targets: Recall@5 ≥ {TARGET_RECALL_AT_5:.0%}, MRR@10 ≥ {TARGET_MRR_AT_10:.2f}. "
        "A chunk is relevant when it overlaps a labelled evidence span.",
        "",
        "| Chunk set | Method | "
        + " | ".join(f"Recall@{x}" for x in config.ks)
        + f" | MRR@{config.mrr_cutoff} | p50 ms | p95 ms | CPU ms |",
        "|---|---|" + "---|" * (len(config.ks) + 4),
    ]
    for s in run.summaries:
        recall_cells = [
            _mark(s.recall[x], TARGET_RECALL_AT_5) if x == 5 else _pct(s.recall[x])
            for x in config.ks
        ]
        mrr = f"{s.mrr:.3f} {'✅' if s.mrr >= TARGET_MRR_AT_10 else '❌'}"
        lines.append(
            f"| `{s.chunk_set}` | `{s.method}` | {' | '.join(recall_cells)} | {mrr} | "
            f"{s.latency_p50_ms:.1f} | {s.latency_p95_ms:.1f} | {s.cpu_ms_mean:.1f} |"
        )

    duels = [
        (chunk_set, label, duel)
        for chunk_set in config.chunk_sets
        for first, second, label in DUELS
        if (duel := head_to_head(run.results, chunk_set, first, second)) is not None
    ]
    if duels:
        lines += [
            "",
            "## Head to head",
            "",
            "Per question, which method ranked the evidence higher (by reciprocal rank).",
            "",
            "| Chunk set | Question | Wins | Losses | Ties |",
            "|---|---|---|---|---|",
            *(
                f"| `{cs}` | {label} (`{d.first}` vs `{d.second}`) | {d.first_wins} | "
                f"{d.second_wins} | {d.ties} |"
                for cs, label, d in duels
            ),
        ]

    if config.generation:
        diagnosis = diagnose_failures(run.results, config.answer_top_k)
        lines += [
            "",
            "## Generation",
            "",
            f"Target: valid citations ≥ {TARGET_VALID_CITATIONS:.0%} of non-abstained answers. "
            f"Each answer used the top {config.answer_top_k} chunks.",
            "",
            "| Chunk set | Method | Abstained | Valid citations | Supported (judge) | Tokens |",
            "|---|---|---|---|---|---|",
            *(
                f"| `{s.chunk_set}` | `{s.method}` | {_pct(s.abstention_rate)} | "
                f"{_mark(s.citation_valid_rate, TARGET_VALID_CITATIONS)} | "
                f"{_pct(s.supported_rate)} | {s.tokens if s.tokens is not None else '—'} |"
                for s in run.summaries
            ),
            "",
            "### Retrieval or generation?",
            "",
            f"{diagnosis.bad_answers} bad answers (abstained, invalid citations or unsupported): "
            f"**{diagnosis.retrieval_misses}** had no evidence in the sources given to the LLM "
            f"(retrieval problem), **{diagnosis.generation_errors}** had it and still failed "
            "(generation problem).",
        ]
    return "\n".join(lines) + "\n"


def missed_questions(run: BenchmarkRun, texts: Mapping[str, str]) -> list[str]:
    """Questions no strategy retrieved evidence for (worth reviewing the label)."""
    found = {str(r.question_id) for r in run.results if r.first_relevant_rank is not None}
    every = {str(r.question_id) for r in run.results}
    return sorted(texts.get(q, q) for q in every - found)
