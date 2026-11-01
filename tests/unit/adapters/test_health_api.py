"""The API exposes a liveness endpoint."""

from fastapi.testclient import TestClient

from rag_lab.adapters.inbound.api.app import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
