"""HTTP contract of the benchmark endpoints."""

from fastapi.testclient import TestClient

from tests.unit.adapters.conftest import make_client

TEXT = b"Refunds are issued within 14 days. Shipping takes three business days."


def seeded(client: TestClient, approve: bool = True) -> str:
    configs = [{"strategy": "fixed", "chunk_size": 40, "chunk_overlap": 0}]
    collection_id = client.post(
        "/collections", json={"name": "docs", "chunking_configs": configs}
    ).json()["id"]
    document_id = client.post(
        f"/collections/{collection_id}/documents", files={"file": ("faq.txt", TEXT)}
    ).json()["id"]
    client.post(f"/collections/{collection_id}/chunk-sets/fixed-40-0/index")
    question = client.post(
        f"/collections/{collection_id}/golden",
        json={"question": "How long do refunds take?", "document_id": document_id,
              "excerpt": "within 14 days"},
    ).json()  # fmt: skip
    if approve:
        client.patch(
            f"/collections/{collection_id}/golden/{question['id']}", json={"status": "approved"}
        )
    return collection_id


def test_run_list_get_and_report() -> None:
    client = make_client()
    collection_id = seeded(client)
    url = f"/collections/{collection_id}/benchmarks"

    created = client.post(url, json={"methods": ["fulltext", "vector"], "ks": [1, 5]})
    run_id = created.json()["id"]
    full = client.get(f"{url}/{run_id}").json()
    report = client.get(f"{url}/{run_id}/report")

    assert created.status_code == 201
    body = created.json()
    assert body["question_count"] == 1
    assert [(s["chunk_set"], s["method"]) for s in body["summaries"]] == [
        ("fixed-40-0", "fulltext"),
        ("fixed-40-0", "vector"),
    ]
    assert body["summaries"][0]["recall"] == {"1": 1.0, "5": 1.0}
    assert set(body["environment"]) >= {"embedding", "reranker", "llm"}
    assert [r["id"] for r in client.get(url).json()] == [run_id]
    assert len(full["results"]) == 2
    assert full["format_version"] == 1
    assert report.headers["content-type"].startswith("text/markdown")
    assert report.text.startswith("# Benchmark report — docs")


def test_defaults_cover_every_method_and_validation_errors() -> None:
    client = make_client()
    collection_id = seeded(client)
    url = f"/collections/{collection_id}/benchmarks"

    every = client.post(url, json={})
    judge_only = client.post(url, json={"judge": True})

    assert [s["method"] for s in every.json()["summaries"]] == [
        "vector", "fulltext", "hybrid", "hybrid_rerank"
    ]  # fmt: skip
    assert judge_only.status_code == 422
    assert client.post(url, json={"chunk_sets": ["nope"]}).status_code == 404
    unapproved = make_client()
    assert (
        unapproved.post(
            f"/collections/{seeded(unapproved, approve=False)}/benchmarks", json={}
        ).status_code
        == 422
    )
    assert client.get(f"{url}/00000000-0000-0000-0000-000000000000").status_code == 404
