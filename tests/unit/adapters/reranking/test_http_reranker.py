"""The /rerank API adapter, exercised against a real local HTTP stub."""

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rag_lab.adapters.outbound.reranking.http_api import HttpReranker
from rag_lab.application.errors import RerankerError

Responder = Callable[[dict[str, Any]], tuple[int, Any]]


def jina_style(body: dict[str, Any]) -> tuple[int, Any]:
    # Results sorted by score (not input order), as real providers return them.
    scores = [float(len(text)) for text in body["documents"]]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return 200, {"results": [{"index": i, "relevance_score": scores[i]} for i in order]}


class StubRerankAPI:
    def __init__(self, responder: Responder) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, **body})
                stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                status, payload = responder(body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def stub_factory() -> Iterator[Callable[[Responder], StubRerankAPI]]:
    stubs: list[StubRerankAPI] = []

    def make(responder: Responder) -> StubRerankAPI:
        stubs.append(StubRerankAPI(responder))
        return stubs[-1]

    yield make
    for stub in stubs:
        stub.server.shutdown()


def test_sends_query_and_documents_and_maps_scores_back_to_input_order(
    stub_factory: Callable[[Responder], StubRerankAPI],
) -> None:
    stub = stub_factory(jina_style)
    reranker = HttpReranker(stub.url, "rerank-model", api_key="secret")

    scores = reranker.score("q", ["a", "ccc", "bb"])

    assert scores == [1.0, 3.0, 2.0]
    [request] = stub.requests
    assert request["path"] == "/v1/rerank"
    assert request["model"] == "rerank-model"
    assert request["query"] == "q"
    assert request["documents"] == ["a", "ccc", "bb"]
    assert request["top_n"] == 3
    assert stub.headers[0]["authorization"] == "Bearer secret"
    assert reranker.name == "rerank-model"


def test_accepts_a_bare_list_with_score_fields(
    stub_factory: Callable[[Responder], StubRerankAPI],
) -> None:
    stub = stub_factory(
        lambda body: (200, [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])
    )

    assert HttpReranker(stub.url, "m").score("q", ["a", "b"]) == [0.1, 0.9]
    assert "authorization" not in stub.headers[0]


def test_empty_input_makes_no_request(stub_factory: Callable[[Responder], StubRerankAPI]) -> None:
    stub = stub_factory(jina_style)

    assert HttpReranker(stub.url, "m").score("q", []) == []
    assert stub.requests == []


@pytest.mark.parametrize(
    ("responder", "message"),
    [
        (lambda body: (500, {"error": "boom"}), "HTTP 500"),
        (lambda body: (200, {"results": [{"index": 0, "relevance_score": 1.0}]}), "missing"),
        (lambda body: (200, {"unexpected": True}), "unexpected"),
    ],
)
def test_failures_are_reported_as_reranker_errors(
    stub_factory: Callable[[Responder], StubRerankAPI], responder: Responder, message: str
) -> None:
    stub = stub_factory(responder)

    with pytest.raises(RerankerError, match=message):
        HttpReranker(stub.url, "m").score("q", ["a", "b"])


def test_unreachable_endpoint_is_a_reranker_error() -> None:
    with pytest.raises(RerankerError, match=r"127\.0\.0\.1:9"):
        HttpReranker("http://127.0.0.1:9/v1", "m", timeout_seconds=1).score("q", ["a"])


def test_base_url_is_required() -> None:
    with pytest.raises(ValueError, match="RERANKER_BASE_URL"):
        HttpReranker("", "m")
