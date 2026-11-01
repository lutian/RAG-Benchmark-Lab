"""The UI's HTTP client: the answer timeout follows the configured LLM timeout."""

import pytest

from rag_lab.config import Settings
from rag_lab.ui import api_client


def test_answer_timeout_covers_two_llm_attempts_plus_a_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, llm_timeout_seconds=300)  # type: ignore[call-arg]
    monkeypatch.setattr(api_client, "get_settings", lambda: settings)
    api_client._cached_client.clear()  # type: ignore[attr-defined]

    client = api_client.get_client()

    assert client._generation_timeout == 2 * 300 + api_client.GENERATION_MARGIN_SECONDS


def test_non_json_error_pages_become_api_errors() -> None:
    import httpx

    def proxy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad gateway</html>")

    client = api_client.ApiClient("http://api")
    client._http = httpx.Client(base_url="http://api", transport=httpx.MockTransport(proxy))

    with pytest.raises(api_client.ApiError, match="Bad gateway"):
        client.benchmark_report("c1", "r1")
