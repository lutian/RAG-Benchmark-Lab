"""HTTP contract of the answer endpoint."""

from fastapi.testclient import TestClient

from rag_lab.domain.answers import ABSTENTION
from tests.fakes import FakeChatModel
from tests.unit.adapters.conftest import make_client

KEY = "fixed-200-0"


def seeded(client: TestClient) -> str:
    configs = [{"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0}]
    body = {"name": "docs", "chunking_configs": configs}
    collection_id = str(client.post("/collections", json=body).json()["id"])
    for name, text in {"a.txt": b"The bank raised rates.", "b.txt": b"A quiet river."}.items():
        client.post(f"/collections/{collection_id}/documents", files={"file": (name, text)})
    return collection_id


def ask(client: TestClient, collection_id: str, question: str, **extra: object) -> dict:  # type: ignore[type-arg]
    body = {"question": question, "chunk_set": KEY, "method": "fulltext", **extra}
    response = client.post(f"/collections/{collection_id}/answer", json=body)
    return {"status": response.status_code, **response.json()}


def test_answer_with_valid_citations_sources_usage_and_timings() -> None:
    client = make_client(chat=FakeChatModel("The bank raised rates [1]."))
    collection_id = seeded(client)

    body = ask(client, collection_id, "Who raised rates?")

    assert body["status"] == 200
    assert body["answer"] == "The bank raised rates [1]."
    assert body["abstained"] is False
    assert body["citations_valid"] is True
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["source"]["text"] == "The bank raised rates."
    assert [s["text"] for s in body["sources"]] == ["The bank raised rates."]
    assert (body["prompt_tokens"], body["completion_tokens"]) == (10, 5)
    assert body["model"] == "fake-chat-1"
    assert {"embed_ms", "search_ms", "rerank_ms", "total_ms", "generation_ms"} <= set(
        body["timings"]
    )


def test_invalid_citations_and_abstentions_are_reported() -> None:
    invalid = make_client(chat=FakeChatModel("Rates [9]."))
    abstaining = make_client(chat=FakeChatModel(f"{ABSTENTION}: nothing about dates."))

    collection_id = seeded(abstaining)

    bad = ask(invalid, seeded(invalid), "rates?")
    model = ask(abstaining, collection_id, "When were rates raised?")
    empty = ask(abstaining, collection_id, "zzz")

    assert (bad["citations_valid"], bad["invalid_citations"]) == (False, [9])
    assert (model["abstained"], model["abstention_reason"]) == (True, "model_abstained")
    assert (empty["abstained"], empty["abstention_reason"]) == (True, "no_evidence_retrieved")
    assert empty["sources"] == []


def test_llm_failure_is_a_bad_gateway_and_validation_applies() -> None:
    client = make_client(chat=FakeChatModel(fail=True))
    collection_id = seeded(client)

    assert ask(client, collection_id, "rates?")["status"] == 502
    assert ask(client, collection_id, "")["status"] == 422
    assert ask(client, collection_id, "x", top_k=0)["status"] == 422
    assert ask(client, collection_id, "x", chunk_set="nope")["status"] == 404
