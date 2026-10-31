"""The retrieval page compares the three search methods side by side (API is faked)."""

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import rag_lab.ui
from rag_lab.ui import api_client

PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "retrieval.py")
COLLECTION = {
    "id": "c1",
    "name": "Docs",
    "created_at": "2026-10-30T00:00:00Z",
    "fulltext_language": "english",
    "chunking_configs": [
        {"key": "fixed-200-0", "strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0}
    ],
}
HIT = {
    "document_id": "d1",
    "index": 0,
    "start": 0,
    "end": 22,
    "text": "The bank raised rates.",
    "page_numbers": [3],
    "section_path": ["Rates"],
    "rank": 1,
    "score": 0.5,
}


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, list[str] | None]] = []

    def list_collections(self) -> list[dict[str, Any]]:
        return [COLLECTION]

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        return [{"id": "d1", "filename": "rates.txt"}]

    def search(
        self,
        collection_id: str,
        query: str,
        chunk_set: str,
        method: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, top_k, document_ids))
        if method == "vector":
            raise api_client.ApiError("chunk set 'fixed-200-0' is not fully indexed (0/1 chunks)")
        ranks = {"fulltext": 1} if method == "fulltext" else {"vector": 4, "fulltext": 1}
        return {
            "query": query,
            "chunk_set": chunk_set,
            "method": method,
            "hits": [{**HIT, "ranks": ranks}],
            "timings": {"embed_ms": 0.0, "search_ms": 1.5, "total_ms": 2.0},
        }


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(api_client, "get_client", lambda: client)
    return client


def test_compares_every_method_for_one_query(fake: FakeClient) -> None:
    app = AppTest.from_file(PAGE).run()
    app.text_input[0].input("bank rates")
    app.button[0].click().run()

    assert not app.exception
    assert [call[0] for call in fake.calls] == ["vector", "fulltext", "hybrid"]
    rendered = " ".join(md.value for md in app.markdown)
    assert "rates.txt" in rendered
    assert '<mark class="rl-evidence">bank</mark>' in rendered
    assert "Vector #4" in rendered
    assert "p. 3 · Rates" in rendered
    assert any("not fully indexed" in warning.value for warning in app.warning)


def test_waits_for_a_query_before_searching(fake: FakeClient) -> None:
    app = AppTest.from_file(PAGE).run()

    assert not app.exception
    assert fake.calls == []
    assert any("Full-text language: english" in caption.value for caption in app.caption)


def test_empty_state_without_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    class Empty(FakeClient):
        def list_collections(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(api_client, "get_client", lambda: Empty())

    app = AppTest.from_file(PAGE).run()

    assert any("Ingestion page" in info.value for info in app.info)
