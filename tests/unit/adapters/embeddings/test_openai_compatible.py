"""The OpenAI-compatible embedder, exercised against a real local HTTP stub."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rag_lab.adapters.outbound.embeddings.openai_compatible import OpenAICompatibleEmbedder
from rag_lab.application.errors import EmbeddingProviderError
from rag_lab.domain.embeddings import EmbeddingSpec


class StubProvider:
    """Minimal ``POST /v1/embeddings`` server that records every request."""

    def __init__(self, dimensions: int = 3, status: int = 200) -> None:
        self.dimensions = dimensions
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                if stub.status != 200:
                    self._send(stub.status, {"error": {"message": "boom", "type": "server_error"}})
                    return
                data = [
                    {"object": "embedding", "index": i, "embedding": [float(i)] * stub.dimensions}
                    for i, _ in enumerate(body["input"])
                ]
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": data,
                        "model": body["model"],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    },
                )

            def _send(self, status: int, payload: dict[str, Any]) -> None:
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

    def __enter__(self) -> "StubProvider":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider() -> Iterator[StubProvider]:
    with StubProvider() as stub:
        yield stub


def embedder(url: str, **overrides: Any) -> OpenAICompatibleEmbedder:
    options: dict[str, Any] = {
        "base_url": url,
        "api_key": "sk-test",
        "model": "nomic-embed-text",
        "dimensions": 3,
        "batch_size": 2,
        "timeout_seconds": 5,
        "max_retries": 0,
    }
    options.update(overrides)
    return OpenAICompatibleEmbedder(**options)


def test_sends_raw_text_in_batches_with_model_and_key(provider: StubProvider) -> None:
    vectors = embedder(provider.url).embed_documents(["one", "two", "three"])

    assert vectors == [[0.0] * 3, [1.0] * 3, [0.0] * 3]
    assert [r["input"] for r in provider.requests] == [["one", "two"], ["three"]]
    assert {r["model"] for r in provider.requests} == {"nomic-embed-text"}
    assert provider.headers[0]["authorization"] == "Bearer sk-test"


def test_does_not_send_the_dimensions_parameter(provider: StubProvider) -> None:
    embedder(provider.url).embed_query("hello")

    assert "dimensions" not in provider.requests[0]


def test_exposes_its_spec() -> None:
    assert embedder("http://unused/v1").spec == EmbeddingSpec("nomic-embed-text", 3)


def test_empty_api_key_still_reaches_local_providers(provider: StubProvider) -> None:
    embedder(provider.url, api_key="").embed_query("hello")

    assert provider.requests


def test_provider_errors_become_embedding_provider_errors() -> None:
    with (
        StubProvider(status=500) as failing,
        pytest.raises(EmbeddingProviderError, match="nomic-embed-text"),
    ):
        embedder(failing.url).embed_documents(["x"])


def test_unreachable_provider_is_reported() -> None:
    with pytest.raises(EmbeddingProviderError, match=r"127\.0\.0\.1:9"):
        embedder("http://127.0.0.1:9/v1").embed_documents(["x"])
