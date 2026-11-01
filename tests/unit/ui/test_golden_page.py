"""The golden dataset page lists questions with their evidence and drives review (API faked)."""

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import rag_lab.ui
from rag_lab.ui import api_client

PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "golden.py")
COLLECTION = {"id": "c1", "name": "Docs", "chunking_configs": [], "fulltext_language": "english"}


def question(qid: str, status: str) -> dict[str, Any]:
    return {
        "id": qid,
        "text": f"Question {qid}?",
        "status": status,
        "origin": "synthetic",
        "created_at": "2026-11-01T00:00:00Z",
        "evidence": [
            {"document_id": "d1", "filename": "policy.txt", "start": 0, "end": 14,
             "excerpt": "Refunds <fast>"}
        ],
    }  # fmt: skip


class FakeClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str | None, str | None]] = []

    def list_collections(self) -> list[dict[str, Any]]:
        return [COLLECTION]

    def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        return [{"id": "d1", "filename": "policy.txt"}]

    def golden_questions(
        self, collection_id: str, status: str | None = None
    ) -> list[dict[str, Any]]:
        return [question("q1", "draft"), question("q2", "approved")]

    def golden_update(
        self, collection_id: str, qid: str, text: str | None, status: str | None
    ) -> dict[str, Any]:
        self.updates.append((qid, text, status))
        return question(qid, status or "draft")

    def golden_export(self, collection_id: str) -> dict[str, Any]:
        return {"format_version": 1, "questions": []}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(api_client, "get_client", lambda: client)
    return client


def test_lists_questions_with_escaped_evidence_and_progress(fake: FakeClient) -> None:
    app = AppTest.from_file(PAGE).run()

    assert not app.exception
    rendered = " ".join(md.value for md in app.markdown)
    assert "Question q1?" in rendered and "Question q2?" in rendered
    assert "Refunds &lt;fast&gt;" in rendered
    assert app.metric[0].value == "1 / 50"
    assert app.metric[1].value == "1"


def test_approve_button_updates_the_status(fake: FakeClient) -> None:
    app = AppTest.from_file(PAGE).run()

    next(b for b in app.button if b.key == "approve-q1").click().run()

    assert ("q1", None, "approved") in fake.updates
