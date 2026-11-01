"""HTTP contract of the golden dataset endpoints."""

from fastapi.testclient import TestClient

from tests.fakes import FakeChatModel
from tests.unit.adapters.conftest import make_client

TEXT = (
    b"Refunds are issued within 14 days of a return request, provided the item is unused "
    b"and in its original packaging. Shipping costs are not refunded unless the item "
    b"arrived damaged or the wrong item was sent."
)


def seeded(client: TestClient) -> tuple[str, str]:
    configs = [{"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0}]
    collection_id = client.post(
        "/collections", json={"name": "docs", "chunking_configs": configs}
    ).json()["id"]
    document = client.post(
        f"/collections/{collection_id}/documents", files={"file": ("policy.txt", TEXT)}
    ).json()
    return collection_id, document["id"]


def test_manual_question_review_and_listing() -> None:
    client = make_client()
    collection_id, document_id = seeded(client)
    url = f"/collections/{collection_id}/golden"

    created = client.post(
        url,
        json={"question": "How long do refunds take?", "document_id": document_id,
              "excerpt": "within  14 days"},
    )  # fmt: skip
    question_id = created.json()["id"]
    approved = client.patch(f"{url}/{question_id}", json={"status": "approved"})
    edited = client.patch(f"{url}/{question_id}", json={"text": "Refund delay?"})

    assert created.status_code == 201
    [evidence] = created.json()["evidence"]
    assert (evidence["excerpt"], evidence["filename"]) == ("within 14 days", "policy.txt")
    assert created.json()["status"] == "draft"
    assert approved.json()["status"] == "approved"
    assert edited.json()["text"] == "Refund delay?"
    assert [q["id"] for q in client.get(url, params={"status": "approved"}).json()] == [question_id]
    assert client.get(url, params={"status": "draft"}).json() == []
    assert client.delete(f"{url}/{question_id}").status_code == 204
    assert client.delete(f"{url}/{question_id}").status_code == 404


def test_generate_creates_drafts_with_passage_evidence() -> None:
    client = make_client(chat=FakeChatModel("How long do refunds take?"))
    collection_id, _ = seeded(client)

    response = client.post(f"/collections/{collection_id}/golden/generate", json={"count": 1})

    assert response.status_code == 201
    [question] = response.json()
    assert question["origin"] == "synthetic"
    assert question["text"] == "How long do refunds take?"
    assert question["evidence"][0]["excerpt"].startswith("Refunds are issued")


def test_export_import_round_trip_and_errors() -> None:
    client = make_client()
    collection_id, document_id = seeded(client)
    url = f"/collections/{collection_id}/golden"
    client.post(url, json={"question": "Q?", "document_id": document_id, "excerpt": "14 days"})

    exported = client.get(f"{url}/export").json()
    other = make_client()
    other_id, _ = seeded(other)
    report = other.post(f"/collections/{other_id}/golden/import", json=exported)

    assert report.json() == {"imported": 1, "skipped": 0}
    assert other.get(f"/collections/{other_id}/golden").json()[0]["text"] == "Q?"
    missing = client.post(
        url, json={"question": "Q?", "document_id": document_id, "excerpt": "unicorn"}
    )
    assert missing.status_code == 422
    assert client.post(f"{url}/import", json={"questions": []}).status_code == 422
    assert (
        client.patch(f"{url}/00000000-0000-0000-0000-000000000000", json={"text": "x"}).status_code
        == 404
    )
