"""The ask page shows a cited answer next to its evidence (API is faked)."""

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import rag_lab.ui
from rag_lab.ui import api_client

PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "ask.py")
COLLECTION = {
    "id": "c1",
    "name": "Docs",
    "created_at": "2026-10-30T00:00:00Z",
    "fulltext_language": "english",
    "chunking_configs": [
        {"key": "fixed-200-0", "strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0}
    ],
}
SOURCE = {
    "document_id": "d1",
    "index": 0,
    "start": 0,
    "end": 22,
    "text": "The bank raised rates.",
    "page_numbers": [2],
    "section_path": [],
    "rank": 1,
    "score": 0.8,
    "ranks": {"fulltext": 1},
}


def answer(**overrides: Any) -> dict[str, Any]:
    return {
        "question": "Who raised rates?",
        "chunk_set": "fixed-200-0",
        "method": "hybrid",
        "answer": "The bank did [1].",
        "abstained": False,
        "abstention_reason": None,
        "citations": [{"number": 1, "source": SOURCE}],
        "citations_valid": True,
        "invalid_citations": [],
        "sources": [SOURCE, {**SOURCE, "text": "A quiet river.", "index": 1}],
        "model": "gpt-test",
        "prompt_tokens": 120,
        "completion_tokens": 9,
        "timings": {
            "retrieval": {"embed_ms": 1.0, "search_ms": 2.0, "rerank_ms": 0.0, "total_ms": 3.0},
            "generation_ms": 450.0,
            "total_ms": 455.0,
        },
        **overrides,
    }


class FakeClient:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, int]] = []

    def list_collections(self) -> list[dict[str, Any]]:
        return [COLLECTION]

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        return [{"id": "d1", "filename": "rates.txt"}]

    def answer(
        self,
        collection_id: str,
        question: str,
        chunk_set: str,
        method: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((question, method, top_k))
        return self.reply


def run(monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any]) -> tuple[AppTest, FakeClient]:
    fake = FakeClient(reply)
    monkeypatch.setattr(api_client, "get_client", lambda: fake)
    app = AppTest.from_file(PAGE).run()
    app.text_input[0].input("Who raised rates?")
    app.button[0].click().run()
    return app, fake


def test_shows_the_cited_answer_next_to_its_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    app, fake = run(monkeypatch, answer())

    assert not app.exception
    assert fake.calls == [("Who raised rates?", "hybrid", 5)]
    rendered = " ".join(md.value for md in app.markdown)
    assert '<span class="rl-cite">[1]</span>' in rendered
    assert "citations valid" in rendered
    assert '<div class="rl-card selected">' in rendered  # cited source is marked
    assert "[2] rates.txt" in rendered  # uncited sources are still shown
    assert "120 + 9 tokens" in rendered
    assert "generation 450 ms" in rendered


def test_explains_why_the_answer_abstained(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = run(
        monkeypatch,
        answer(
            answer="",
            abstained=True,
            abstention_reason="no_evidence_retrieved",
            citations=[],
            sources=[],
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
        ),
    )

    assert not app.exception
    assert any("Retrieval found no evidence" in w.value for w in app.warning)


def test_abstained_text_is_shown_without_citation_badges(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = run(
        monkeypatch,
        answer(
            answer="NO_ANSWER: source [9] is about rivers.",
            abstained=True,
            abstention_reason="model_abstained",
            citations=[],
        ),
    )

    rendered = " ".join(md.value for md in app.markdown)
    assert "source [9] is about rivers" in rendered
    assert '<span class="rl-cite' not in rendered
