"""The ingestion page lists documents and previews chunk boundaries (API is faked)."""

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import rag_lab.ui
from rag_lab.ui import api_client

PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "ingestion.py")
COLLECTION = {
    "id": "c1",
    "name": "Docs",
    "created_at": "2026-10-30T00:00:00Z",
    "chunking_configs": [
        {
            "key": "structure-800-120",
            "strategy": "structure",
            "chunk_size": 800,
            "chunk_overlap": 120,
        }
    ],
}
DOCUMENT = {
    "id": "d1",
    "collection_id": "c1",
    "filename": "guide.md",
    "media_type": "text/markdown",
    "content_hash": "h",
    "characters": 20,
    "metadata": {"title": "Guide"},
    "created_at": "2026-10-30T00:00:00Z",
}


class FakeClient:
    def list_collections(self) -> list[dict[str, Any]]:
        return [COLLECTION]

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        return [DOCUMENT]

    def get_document(self, collection_id: str, document_id: str) -> dict[str, Any]:
        return {**DOCUMENT, "text": "# Guide\nHello chunks", "elements": []}

    def list_chunks(
        self, collection_id: str, config_key: str, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "document_id": "d1",
                "index": 0,
                "start": 0,
                "end": 20,
                "text": "x",
                "page_numbers": [],
                "section_path": ["Guide"],
            }
        ]

    def index_status(self, collection_id: str) -> dict[str, Any]:
        return {
            "embedding": {"model": "nomic-embed-text", "dimensions": 768},
            "chunk_sets": [{"config_key": "structure-800-120", "total": 40, "embedded": 12}],
        }


@pytest.fixture(autouse=True)
def fake_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_client, "get_client", lambda: FakeClient())


def test_lists_documents_and_previews_chunks() -> None:
    app = AppTest.from_file(PAGE).run()

    assert not app.exception
    rendered = " ".join(md.value for md in app.markdown)
    assert "guide.md" in rendered
    assert 'class="rl-chunk-a"' in rendered


def test_shows_empty_state_without_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    class Empty(FakeClient):
        def list_collections(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(api_client, "get_client", lambda: Empty())

    app = AppTest.from_file(PAGE).run()

    assert not app.exception
    assert any("Create a collection" in info.value for info in app.info)


def test_shows_indexing_progress_and_embedding_binding() -> None:
    app = AppTest.from_file(PAGE).run()

    rendered = " ".join(md.value for md in app.markdown)
    assert "12 / 40 chunks embedded" in rendered
    assert "nomic-embed-text (768d)" in rendered
    assert any(button.label == "Index chunk set" for button in app.button)
