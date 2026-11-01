"""HTTP contract of the indexing endpoints."""

from collections.abc import Callable

from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder

TEXT = b"Embeddings capture meaning. " * 20
KEY = "recursive-100-0"


def seeded(client: TestClient) -> str:
    configs = [{"strategy": "recursive", "chunk_size": 100, "chunk_overlap": 0}]
    collection_id = str(
        client.post("/collections", json={"name": "docs", "chunking_configs": configs}).json()["id"]
    )
    client.post(f"/collections/{collection_id}/documents", files={"file": ("a.txt", TEXT)})
    return collection_id


def test_index_endpoint_embeds_the_chunk_set_and_reports_progress(client: TestClient) -> None:
    collection_id = seeded(client)

    response = client.post(f"/collections/{collection_id}/chunk-sets/{KEY}/index")

    assert response.status_code == 200
    body = response.json()
    assert body["embedded"] == body["total"] > 0
    assert body["already_indexed"] == 0
    assert body["embedding"] == {"model": "fake-embed", "dimensions": 8}
    assert body["seconds"] >= 0


def test_status_endpoint_shows_binding_and_progress(client: TestClient) -> None:
    collection_id = seeded(client)
    before = client.get(f"/collections/{collection_id}/index").json()
    client.post(f"/collections/{collection_id}/chunk-sets/{KEY}/index")

    after = client.get(f"/collections/{collection_id}/index").json()

    assert before["embedding"] is None
    assert before["chunk_sets"][0]["embedded"] == 0
    assert after["embedding"] == {"model": "fake-embed", "dimensions": 8}
    assert after["chunk_sets"][0]["embedded"] == after["chunk_sets"][0]["total"]
    collection = client.get(f"/collections/{collection_id}").json()
    assert collection["embedding"] == {"model": "fake-embed", "dimensions": 8}


def test_rebuild_flag_reindexes_everything(client: TestClient) -> None:
    collection_id = seeded(client)
    client.post(f"/collections/{collection_id}/chunk-sets/{KEY}/index")

    body = client.post(
        f"/collections/{collection_id}/chunk-sets/{KEY}/index", params={"rebuild": True}
    ).json()

    assert body["already_indexed"] == 0
    assert body["embedded"] == body["total"]


def test_unknown_chunk_set_returns_404(client: TestClient) -> None:
    collection_id = seeded(client)

    response = client.post(f"/collections/{collection_id}/chunk-sets/fixed-1-0/index")

    assert response.status_code == 404


def test_provider_failure_returns_502_with_detail(
    client_factory: Callable[[FakeEmbedder], TestClient],
) -> None:
    client = client_factory(FakeEmbedder(fail_on_call=1))
    collection_id = seeded(client)

    response = client.post(f"/collections/{collection_id}/chunk-sets/{KEY}/index")

    assert response.status_code == 502
    assert "outage" in response.json()["detail"]
