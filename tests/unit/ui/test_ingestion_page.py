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


class ManagedClient(FakeClient):
    """Keeps collections in memory so rename and delete are visible on the next run."""

    def __init__(self) -> None:
        self.collections = [dict(COLLECTION), {**COLLECTION, "id": "c2", "name": "Other"}]
        self.calls: list[tuple[str, ...]] = []

    def list_collections(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self.collections]

    def rename_collection(self, collection_id: str, name: str) -> dict[str, Any]:
        self.calls.append(("rename", collection_id, name))
        if any(c["name"] == name and c["id"] != collection_id for c in self.collections):
            raise api_client.ApiError(f"collection {name!r} already exists")
        target = next(c for c in self.collections if c["id"] == collection_id)
        target["name"] = name
        return dict(target)

    def delete_collection(self, collection_id: str) -> None:
        self.calls.append(("delete", collection_id))
        self.collections = [c for c in self.collections if c["id"] != collection_id]


@pytest.fixture
def managed(monkeypatch: pytest.MonkeyPatch) -> ManagedClient:
    client = ManagedClient()
    monkeypatch.setattr(api_client, "get_client", lambda: client)
    return client


def dialog_button(app: AppTest, label: str) -> Any:
    """The dialog's button: drawn after the sidebar button with the same label."""
    return [b for b in app.button if b.label == label][-1]


def selected(app: AppTest) -> Any:
    return app.selectbox(key="ingestion-collection")


def test_rename_requires_confirmation_and_selects_the_new_name(managed: ManagedClient) -> None:
    app = AppTest.from_file(PAGE).run()
    dialog_button(app, "Rename").click().run()
    assert managed.calls == []

    next(t for t in app.text_input if t.label == "New name").set_value("Manuals").run()
    dialog_button(app, "Rename").click().run()

    assert not app.exception
    assert managed.calls == [("rename", "c1", "Manuals")]
    assert selected(app).value == "Manuals"
    assert not any(t.label == "New name" for t in app.text_input)


def test_rename_error_is_shown_inside_the_open_dialog(managed: ManagedClient) -> None:
    app = AppTest.from_file(PAGE).run()
    dialog_button(app, "Rename").click().run()
    next(t for t in app.text_input if t.label == "New name").set_value("Other").run()

    dialog_button(app, "Rename").click().run()

    assert any("already exists" in e.value for e in app.error)
    assert any(t.label == "New name" for t in app.text_input)
    assert any(b.label == "Cancel" for b in app.button)
    assert selected(app).value == "Docs"


def test_delete_requires_confirmation_and_removes_the_collection(managed: ManagedClient) -> None:
    app = AppTest.from_file(PAGE).run()
    selected(app).set_value("Other").run()
    dialog_button(app, "Delete").click().run()
    assert any("cannot be undone" in w.value for w in app.warning)
    assert managed.calls == []

    dialog_button(app, "Delete").click().run()

    assert not app.exception
    assert managed.calls == [("delete", "c2")]
    assert selected(app).options == ["Docs"]
    assert not any("cannot be undone" in w.value for w in app.warning)


@pytest.mark.parametrize("action", ["Rename", "Delete"])
def test_cancel_closes_the_dialog_without_calling_the_api(
    managed: ManagedClient, action: str
) -> None:
    app = AppTest.from_file(PAGE).run()
    dialog_button(app, action).click().run()

    dialog_button(app, "Cancel").click().run()

    assert not app.exception
    assert managed.calls == []
    assert not any(b.label == "Cancel" for b in app.button)
