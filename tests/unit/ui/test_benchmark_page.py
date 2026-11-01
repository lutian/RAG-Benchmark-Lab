"""The benchmark dashboard renders a stored run and can start a new one (API faked)."""

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import rag_lab.ui
from rag_lab.ui import api_client

PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "benchmark.py")
CHUNK_SET = {"key": "fixed-800-120", "strategy": "fixed", "chunk_size": 800, "chunk_overlap": 120}
COLLECTION = {
    "id": "c1",
    "name": "Docs",
    "fulltext_language": "english",
    "chunking_configs": [CHUNK_SET],
}


def summary(method: str, recall5: float, mrr: float) -> dict[str, Any]:
    return {"chunk_set": "fixed-800-120", "method": method, "questions": 2,
            "recall": {"1": recall5 / 2, "5": recall5, "10": recall5}, "mrr": mrr,
            "latency_p50_ms": 10.0, "latency_p95_ms": 15.0, "cpu_ms_mean": 2.0,
            "abstention_rate": None, "citation_valid_rate": None, "supported_rate": None,
            "tokens": None}  # fmt: skip


RUN = {
    "id": "r1", "created_at": "2026-11-01T10:00:00+00:00", "question_count": 2,
    "config": {"mrr_cutoff": 10}, "environment": {},
    "summaries": [summary("vector", 0.5, 0.4), summary("hybrid_rerank", 1.0, 0.9)],
}  # fmt: skip


class FakeClient:
    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self.runs = runs
        self.started: list[dict[str, Any]] = []

    def list_collections(self) -> list[dict[str, Any]]:
        return [COLLECTION]

    def benchmark_runs(self, collection_id: str) -> list[dict[str, Any]]:
        return self.runs

    def benchmark_run(self, collection_id: str, run_id: str) -> dict[str, Any]:
        return {"results": [{"question_id": "q1", "chunk_set": "fixed-800-120",
                             "method": "vector", "first_relevant_rank": 2}]}  # fmt: skip

    def golden_questions(
        self, collection_id: str, status: str | None = None
    ) -> list[dict[str, Any]]:
        return [{"id": "q1", "text": "Refund delay?"}]

    def run_benchmark(self, collection_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.started.append(body)
        return RUN


def test_shows_the_best_strategy_against_the_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "get_client", lambda: FakeClient([RUN]))

    app = AppTest.from_file(PAGE).run()

    assert not app.exception
    rendered = " ".join(md.value for md in app.markdown)
    assert "fixed-800-120 · Hybrid + rerank" in rendered
    assert app.metric[0].value == "2 / 50"
    assert app.metric[1].value == "100.0%"
    assert "+15.0% vs target 85%" in app.metric[1].delta
    assert len(app.dataframe) == 2


def test_runs_a_benchmark_with_the_chosen_options(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient([])
    monkeypatch.setattr(api_client, "get_client", lambda: fake)

    app = AppTest.from_file(PAGE).run()
    assert any("No benchmark runs yet" in i.value for i in app.info)
    app.checkbox[0].check()
    app.button[0].click().run()

    assert fake.started == [
        {
            "chunk_sets": ["fixed-800-120"],
            "methods": ["vector", "fulltext", "hybrid", "hybrid_rerank"],
            "generation": True,
            "judge": False,
        }
    ]
