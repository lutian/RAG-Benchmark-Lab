"""The OpenAI-compatible chat model, exercised against a real local HTTP stub."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rag_lab.adapters.outbound.generation.openai_compatible import OpenAICompatibleChatModel
from rag_lab.application.errors import GenerationProviderError
from rag_lab.application.ports.generation import ChatMessage

MESSAGES = [ChatMessage("system", "rules"), ChatMessage("user", "question")]


class StubChat:
    """Minimal ``POST /v1/chat/completions`` server that records every request."""

    def __init__(
        self,
        status: int = 200,
        usage: bool = True,
        error_body: bool = False,
        no_choices: bool = False,
        reject_param: str | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                stub.paths.append(self.path)
                stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                rejected = reject_param is not None and reject_param in body
                if rejected:
                    payload: dict[str, Any] = {
                        "error": {
                            "message": f"Unsupported parameter: '{reject_param}' is not "
                            "supported with this model.",
                            "type": "invalid_request_error",
                            "param": reject_param,
                            "code": "unsupported_parameter",
                        }
                    }
                elif error_body:
                    payload = {"error": {"message": "upstream route failed", "code": 502}}
                elif no_choices:
                    payload = {
                        "id": "c",
                        "object": "chat.completion",
                        "model": "m",
                        "choices": None,
                    }
                elif status != 200:
                    payload = {"error": {"message": "boom", "type": "server_error"}}
                else:
                    payload = {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "served-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": " The answer [1]. "},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    if usage:
                        payload["usage"] = {
                            "prompt_tokens": 12,
                            "completion_tokens": 4,
                            "total_tokens": 16,
                        }
                raw = json.dumps(payload).encode()
                self.send_response(400 if rejected else status)
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
def stub() -> Iterator[StubChat]:
    server = StubChat()
    yield server
    server.server.shutdown()


def model(url: str, **options: float | str) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        base_url=url, api_key="secret", model="my-model", timeout_seconds=5, options=options
    )


def test_sends_chat_completions_and_returns_text_model_and_usage(stub: StubChat) -> None:
    completion = model(stub.url).complete(MESSAGES)

    assert completion.text == "The answer [1]."
    assert completion.model == "served-model"
    assert (completion.prompt_tokens, completion.completion_tokens) == (12, 4)
    [request] = stub.requests
    assert stub.paths == ["/v1/chat/completions"]
    assert request["model"] == "my-model"
    assert request["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    assert stub.headers[0]["authorization"] == "Bearer secret"


def test_sends_only_the_generation_options_that_are_set(stub: StubChat) -> None:
    model(stub.url).complete(MESSAGES)
    model(stub.url, temperature=0.0).complete(MESSAGES)
    model(stub.url, reasoning_effort="low").complete(MESSAGES)

    plain, with_temperature, reasoning = stub.requests
    assert "temperature" not in plain and "reasoning_effort" not in plain
    assert with_temperature["temperature"] == 0.0
    assert reasoning["reasoning_effort"] == "low"
    assert "temperature" not in reasoning


def rejecting(param: str) -> Iterator[StubChat]:
    server = StubChat(reject_param=param)
    yield server
    server.server.shutdown()


@pytest.fixture
def rejects_effort() -> Iterator[StubChat]:
    yield from rejecting("reasoning_effort")


@pytest.fixture
def rejects_temperature() -> Iterator[StubChat]:
    yield from rejecting("temperature")


def fallback_model(url: str, fallback: dict[str, float] | None) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        base_url=url,
        api_key="secret",
        model="my-model",
        timeout_seconds=5,
        options={"reasoning_effort": "low"},
        fallback_options=fallback,
    )


def test_rejected_reasoning_effort_is_retried_with_the_temperature(
    rejects_effort: StubChat,
) -> None:
    chat = fallback_model(rejects_effort.url, {"temperature": 0.0})

    first = chat.complete(MESSAGES)
    chat.complete(MESSAGES)

    assert first.text == "The answer [1]."
    rejected, retried, later = rejects_effort.requests
    assert rejected["reasoning_effort"] == "low" and "temperature" not in rejected
    assert retried["temperature"] == 0.0 and "reasoning_effort" not in retried
    # The switch sticks: later calls do not pay for another rejected request.
    assert later["temperature"] == 0.0 and "reasoning_effort" not in later


def test_rejected_reasoning_effort_without_fallback_is_a_provider_error(
    rejects_effort: StubChat,
) -> None:
    with pytest.raises(GenerationProviderError):
        fallback_model(rejects_effort.url, None).complete(MESSAGES)

    assert len(rejects_effort.requests) == 1


def test_other_rejected_parameters_do_not_trigger_the_fallback(
    rejects_temperature: StubChat,
) -> None:
    chat = OpenAICompatibleChatModel(
        base_url=rejects_temperature.url,
        api_key="secret",
        model="my-model",
        timeout_seconds=5,
        options={"temperature": 0.0},
        fallback_options={"temperature": 0.0},
    )

    with pytest.raises(GenerationProviderError):
        chat.complete(MESSAGES)

    assert len(rejects_temperature.requests) == 1


def test_missing_usage_is_reported_as_unknown() -> None:
    server = StubChat(usage=False)
    try:
        completion = model(server.url).complete(MESSAGES)
    finally:
        server.server.shutdown()

    assert (completion.prompt_tokens, completion.completion_tokens) == (None, None)


def test_http_errors_and_unreachable_hosts_are_provider_errors() -> None:
    failing = StubChat(status=500)
    try:
        with pytest.raises(GenerationProviderError, match="my-model"):
            model(failing.url).complete(MESSAGES)
    finally:
        failing.server.shutdown()
    with pytest.raises(GenerationProviderError):
        OpenAICompatibleChatModel("http://127.0.0.1:9/v1", "", "m", 1).complete(MESSAGES)


@pytest.mark.parametrize("kind", ["error_body", "no_choices"])
def test_a_200_response_without_a_completion_is_a_provider_error(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    server = StubChat(**{kind: True})  # type: ignore[arg-type]
    try:
        with pytest.raises(GenerationProviderError, match="my-model"):
            model(server.url).complete(MESSAGES)
    finally:
        server.server.shutdown()

    # The original exception is logged so a local bug is not hidden behind a 502.
    assert any(record.exc_info for record in caplog.records)
