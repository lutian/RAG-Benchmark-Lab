"""API test client wired to in-memory adapters through the composition root."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from rag_lab.adapters.inbound.api.app import create_app
from rag_lab.bootstrap import build_in_memory_container
from rag_lab.config import Settings
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker


def make_client(
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
    chat: FakeChatModel | None = None,
) -> TestClient:
    settings = Settings(_env_file=None, max_upload_mb=1)  # type: ignore[call-arg]
    container = build_in_memory_container(
        settings,
        embedder=embedder or FakeEmbedder(),
        reranker=reranker or FakeReranker(),
        chat=chat or FakeChatModel("ok [1]"),
    )
    return TestClient(create_app(container))


@pytest.fixture
def client() -> TestClient:
    return make_client()


@pytest.fixture
def client_factory() -> Callable[[FakeEmbedder], TestClient]:
    return make_client
