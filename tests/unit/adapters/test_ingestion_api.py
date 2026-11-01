"""HTTP contract of the ingestion endpoints."""

from fastapi.testclient import TestClient

MARKDOWN = b"# Intro\nRetrieval quality depends on chunking.\n\n# Usage\nUpload files.\n"


def create_collection(client: TestClient, name: str = "docs", **extra: object) -> str:
    response = client.post("/collections", json={"name": name, **extra})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def upload(client: TestClient, collection_id: str, name: str, data: bytes) -> dict[str, object]:
    response = client.post(f"/collections/{collection_id}/documents", files={"file": (name, data)})
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_new_collection_gets_one_default_config_per_strategy(client: TestClient) -> None:
    response = client.post("/collections", json={"name": "docs"})

    assert response.status_code == 201
    keys = [c["key"] for c in response.json()["chunking_configs"]]
    assert keys == ["fixed-800-120", "recursive-800-120", "structure-800-120"]


def test_custom_chunking_configs_are_accepted(client: TestClient) -> None:
    configs = [{"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 30}]
    response = client.post("/collections", json={"name": "c", "chunking_configs": configs})

    assert [c["key"] for c in response.json()["chunking_configs"]] == ["recursive-300-30"]


def test_invalid_chunking_config_is_rejected(client: TestClient) -> None:
    configs = [{"strategy": "fixed", "chunk_size": 100, "chunk_overlap": 100}]
    response = client.post("/collections", json={"name": "c", "chunking_configs": configs})

    assert response.status_code == 422


def test_duplicate_collection_name_returns_409(client: TestClient) -> None:
    create_collection(client)

    assert client.post("/collections", json={"name": "docs"}).status_code == 409


def test_upload_returns_document_summary_and_chunk_counts(client: TestClient) -> None:
    collection_id = create_collection(client)

    body = upload(client, collection_id, "guide.md", MARKDOWN)

    assert body["filename"] == "guide.md"
    assert body["media_type"] == "text/markdown"
    assert body["duplicate"] is False
    assert set(body["chunk_counts"]) == {  # type: ignore[arg-type]
        "fixed-800-120",
        "recursive-800-120",
        "structure-800-120",
    }


def test_document_detail_exposes_reference_text_and_structure(client: TestClient) -> None:
    collection_id = create_collection(client)
    document_id = upload(client, collection_id, "guide.md", MARKDOWN)["id"]

    body = client.get(f"/collections/{collection_id}/documents/{document_id}").json()

    assert body["text"] == MARKDOWN.decode()
    assert [e["label"] for e in body["elements"]] == ["Intro", "Usage"]
    assert body["metadata"]["title"] == "Intro"


def test_chunks_endpoint_returns_spans_over_the_reference_text(client: TestClient) -> None:
    collection_id = create_collection(client)
    document_id = upload(client, collection_id, "guide.md", MARKDOWN)["id"]

    response = client.get(
        f"/collections/{collection_id}/chunk-sets/structure-800-120/chunks",
        params={"document_id": document_id},
    )

    chunks = response.json()
    assert [c["section_path"] for c in chunks] == [["Intro"], ["Usage"]]
    text = MARKDOWN.decode()
    assert all(text[c["start"] : c["end"]] == c["text"] for c in chunks)


def test_adding_a_chunking_config_returns_the_updated_collection(client: TestClient) -> None:
    collection_id = create_collection(client)

    response = client.post(
        f"/collections/{collection_id}/chunking-configs",
        json={"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0},
    )

    assert response.status_code == 201
    assert "fixed-200-0" in [c["key"] for c in response.json()["chunking_configs"]]


def test_unsupported_file_type_returns_415(client: TestClient) -> None:
    collection_id = create_collection(client)

    response = client.post(
        f"/collections/{collection_id}/documents", files={"file": ("a.docx", b"PK..")}
    )

    assert response.status_code == 415
    assert ".docx" in response.json()["detail"]


def test_unparseable_file_returns_422(client: TestClient) -> None:
    collection_id = create_collection(client)

    response = client.post(
        f"/collections/{collection_id}/documents", files={"file": ("a.pdf", b"nope")}
    )

    assert response.status_code == 422


def test_file_over_upload_limit_returns_413(client: TestClient) -> None:
    collection_id = create_collection(client)

    response = client.post(
        f"/collections/{collection_id}/documents",
        files={"file": ("big.txt", b"x" * (1024 * 1024 + 1))},
    )

    assert response.status_code == 413


def test_unknown_collection_returns_404(client: TestClient) -> None:
    response = client.get("/collections/00000000-0000-0000-0000-000000000000/documents")

    assert response.status_code == 404


def test_document_from_another_collection_returns_404(client: TestClient) -> None:
    alpha = create_collection(client, "alpha")
    beta = create_collection(client, "beta")
    document_id = upload(client, beta, "guide.md", MARKDOWN)["id"]

    assert client.get(f"/collections/{alpha}/documents/{document_id}").status_code == 404
