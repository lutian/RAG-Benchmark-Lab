"""Data shaping behind the benchmark dashboard."""

from rag_lab.ui.dashboard import (
    drilldown_frame,
    latency_frame,
    matrix_frame,
    metric_options,
    summary_table,
)

SUMMARIES = [
    {"chunk_set": "fixed-800-120", "method": "vector", "questions": 2,
     "recall": {"1": 0.5, "5": 1.0}, "mrr": 0.75, "latency_p50_ms": 10.0,
     "latency_p95_ms": 20.0, "cpu_ms_mean": 3.0, "abstention_rate": None,
     "citation_valid_rate": None, "supported_rate": None, "tokens": None},
    {"chunk_set": "fixed-800-120", "method": "hybrid_rerank", "questions": 2,
     "recall": {"1": 1.0, "5": 1.0}, "mrr": 1.0, "latency_p50_ms": 40.0,
     "latency_p95_ms": 60.0, "cpu_ms_mean": 30.0, "abstention_rate": None,
     "citation_valid_rate": None, "supported_rate": None, "tokens": None},
]  # fmt: skip


def test_metric_options_follow_the_measured_ks() -> None:
    assert metric_options(SUMMARIES) == {
        "Recall@1": "recall:1",
        "Recall@5": "recall:5",
        "MRR": "mrr",
    }


def test_matrix_and_latency_frames() -> None:
    matrix = matrix_frame(SUMMARIES, "recall:1")
    latency = latency_frame(SUMMARIES)

    assert matrix.to_dict("records") == [
        {"chunk_set": "fixed-800-120", "method": "Vector", "value": 0.5},
        {"chunk_set": "fixed-800-120", "method": "Hybrid + rerank", "value": 1.0},
    ]
    assert list(matrix_frame(SUMMARIES, "mrr")["value"]) == [0.75, 1.0]
    assert latency["strategy"].tolist() == [
        "fixed-800-120 · Vector",
        "fixed-800-120 · Hybrid + rerank",
    ]
    assert latency["p95_ms"].tolist() == [20.0, 60.0]
    assert summary_table(SUMMARIES).columns.tolist()[:4] == [
        "Strategy",
        "Recall@1",
        "Recall@5",
        "MRR",
    ]


def test_drilldown_pivots_first_relevant_rank_per_strategy() -> None:
    results = [
        {"question_id": "q1", "chunk_set": "cs", "method": "vector", "first_relevant_rank": 3},
        {"question_id": "q1", "chunk_set": "cs", "method": "fulltext", "first_relevant_rank": None},
        {"question_id": "q2", "chunk_set": "cs", "method": "vector", "first_relevant_rank": 1},
    ]

    frame = drilldown_frame(results, {"q1": "Where?", "q2": "Ask?"})

    assert frame["Question"].tolist() == ["Ask?", "Where?"]
    assert frame.loc[1, "cs · Vector"] == 3
