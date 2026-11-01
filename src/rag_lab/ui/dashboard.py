"""Pure data shaping for the benchmark dashboard (no Streamlit calls, unit-tested).

The page renders what these functions return, so the logic behind every chart and
table is testable without a browser.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

METHOD_LABELS = {
    "vector": "Vector",
    "fulltext": "Full-text",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + rerank",
}


def strategy_label(summary: Mapping[str, Any]) -> str:
    return f"{summary['chunk_set']} · {METHOD_LABELS.get(summary['method'], summary['method'])}"


def metric_options(summaries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Selectable heatmap metrics: label -> key ("recall:5" or "mrr")."""
    ks = sorted({int(k) for s in summaries for k in s["recall"]})
    return {**{f"Recall@{k}": f"recall:{k}" for k in ks}, "MRR": "mrr"}


def metric_value(summary: Mapping[str, Any], key: str) -> float:
    if key == "mrr":
        return float(summary["mrr"])
    return float(summary["recall"][key.split(":")[1]])


def matrix_frame(summaries: Sequence[Mapping[str, Any]], key: str) -> pd.DataFrame:
    """One row per chunk set x method: the heatmap of the chosen metric."""
    return pd.DataFrame(
        [
            {
                "chunk_set": s["chunk_set"],
                "method": METHOD_LABELS.get(s["method"], s["method"]),
                "value": metric_value(s, key),
            }
            for s in summaries
        ]
    )


def latency_frame(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": strategy_label(s),
                "p50_ms": s["latency_p50_ms"],
                "p95_ms": s["latency_p95_ms"],
                "cpu_ms": s["cpu_ms_mean"],
            }
            for s in summaries
        ]
    )


def summary_table(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The accessible table view of every chart: one row per strategy."""
    rows = []
    for s in summaries:
        row: dict[str, Any] = {"Strategy": strategy_label(s)}
        row.update(
            {f"Recall@{k}": v for k, v in sorted(s["recall"].items(), key=lambda kv: int(kv[0]))}
        )
        row.update(
            {
                "MRR": s["mrr"],
                "p50 ms": s["latency_p50_ms"],
                "p95 ms": s["latency_p95_ms"],
                "CPU ms": s["cpu_ms_mean"],
                "Abstained": s["abstention_rate"],
                "Valid citations": s["citation_valid_rate"],
                "Supported": s["supported_rate"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def drilldown_frame(
    results: Sequence[Mapping[str, Any]], questions: Mapping[str, str]
) -> pd.DataFrame:
    """Per question, the rank of the first relevant chunk for every strategy.

    Empty cells mean the evidence was not retrieved at all: the rows to look at
    when diagnosing a strategy (or a questionable label).
    """
    rows: dict[str, dict[str, Any]] = {}
    for r in results:
        row = rows.setdefault(
            r["question_id"], {"Question": questions.get(r["question_id"], r["question_id"])}
        )
        row[strategy_label(r)] = r["first_relevant_rank"]
    frame = pd.DataFrame(list(rows.values()))
    return frame.sort_values("Question").reset_index(drop=True) if not frame.empty else frame
