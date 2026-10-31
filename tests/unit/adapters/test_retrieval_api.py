"""HTTP contract of the search endpoint and the collection full-text language."""

from fastapi.testclient import TestClient

KEY = "fixed-200-0"


def seeded(client: TestClient, **extra: object) -> str:
    configs = [{"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0}]
    body = {"name": "docs", "chunking_configs": configs, **extra}
    collection_id = str(client.post("/collections", json=body).json()["id"])
    for name, text in {"a.txt": b"The bank raised rates.", "b.txt": b"A quiet river."}.items():
        client.post(f"/collections/{collection_id}/documents", files={"file": (name, text)})
    return collection_id


def test_full_text_search_returns_ranked_hits_with_spans_and_timings(client: TestClient) -> None:
    collection_id = seeded(client)

    response = client.post(
        f"/collections/{collection_id}/search",
        json={"query": "bank", "chunk_set": KEY, "method": "fulltext"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "fulltext"
    assert body["chunk_set"] == KEY
    [hit] = body["hits"]
    assert hit["text"] == "The bank raised rates."
    assert (hit["start"], hit["end"], hit["rank"]) == (0, 22, 1)
    assert hit["ranks"] == {"fulltext": 1}
    assert hit["score"] > 0
    assert set(body["timings"]) == {"embed_ms", "search_ms", "total_ms"}


def test_hybrid_is_the_default_and_needs_an_indexed_chunk_set(client: TestClient) -> None:
    collection_id = seeded(client)
    url = f"/collections/{collection_id}/search"

    before = client.post(url, json={"query": "bank", "chunk_set": KEY})
    client.post(f"/collections/{collection_id}/chunk-sets/{KEY}/index")
    after = client.post(url, json={"query": "bank", "chunk_set": KEY, "top_k": 1})

    assert before.status_code == 409
    assert "not indexed" in before.json()["detail"]
    assert after.status_code == 200
    assert after.json()["method"] == "hybrid"
    assert len(after.json()["hits"]) == 1


def test_document_filter_limits_hits(client: TestClient) -> None:
    collection_id = seeded(client)
    documents = client.get(f"/collections/{collection_id}/documents").json()
    river = next(d["id"] for d in documents if d["filename"] == "b.txt")

    response = client.post(
        f"/collections/{collection_id}/search",
        json={
            "query": "bank river",
            "chunk_set": KEY,
            "method": "fulltext",
            "document_ids": [river],
        },
    )

    assert [hit["document_id"] for hit in response.json()["hits"]] == [river]


def test_invalid_search_requests(client: TestClient) -> None:
    collection_id = seeded(client)
    url = f"/collections/{collection_id}/search"

    assert client.post(url, json={"query": "", "chunk_set": KEY}).status_code == 422
    assert (
        client.post(url, json={"query": "   ", "chunk_set": KEY, "method": "fulltext"}).status_code
        == 422
    )
    assert client.post(url, json={"query": "x", "chunk_set": KEY, "top_k": 0}).status_code == 422
    assert (
        client.post(url, json={"query": "x", "chunk_set": KEY, "method": "magic"}).status_code
        == 422
    )
    assert client.post(url, json={"query": "x", "chunk_set": "nope"}).status_code == 404
    other = "00000000-0000-0000-0000-000000000000"
    assert (
        client.post(
            f"/collections/{other}/search", json={"query": "x", "chunk_set": KEY}
        ).status_code
        == 404
    )


def test_collection_language_defaults_from_settings_and_is_validated(client: TestClient) -> None:
    default = client.post("/collections", json={"name": "en"}).json()
    portuguese = client.post(
        "/collections", json={"name": "pt", "fulltext_language": "portuguese"}
    ).json()
    invalid = client.post("/collections", json={"name": "xx", "fulltext_language": "klingon"})

    assert default["fulltext_language"] == "english"
    assert portuguese["fulltext_language"] == "portuguese"
    assert invalid.status_code == 422
    assert "full-text language" in invalid.json()["detail"]


def test_blank_collection_name_is_a_validation_error(client: TestClient) -> None:
    assert client.post("/collections", json={"name": "   "}).status_code == 422
