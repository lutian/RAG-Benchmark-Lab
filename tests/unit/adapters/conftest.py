"""API test client wired to in-memory adapters through the composition root."""

import pytest
from fastapi.testclient import TestClient

from rag_lab.adapters.inbound.api.app import create_app
from rag_lab.bootstrap import build_in_memory_container
from rag_lab.config import Settings


@pytest.fixture
def client() -> TestClient:
    settings = Settings(_env_file=None, max_upload_mb=1)  # type: ignore[call-arg]
    return TestClient(create_app(build_in_memory_container(settings)))
