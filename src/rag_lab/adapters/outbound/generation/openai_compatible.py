"""Chat model for any OpenAI-compatible ``/chat/completions`` API (OpenAI, OpenRouter,
Ollama, llama.cpp, vLLM, ...), built on LangChain's ``ChatOpenAI``.
"""

import logging
from collections.abc import Mapping, Sequence

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_lab.application.errors import GenerationProviderError
from rag_lab.application.ports.generation import ChatMessage, Completion

# The OpenAI SDK refuses an empty key; local servers (Ollama, llama.cpp) ignore it.
_PLACEHOLDER_KEY = "not-needed"
_log = logging.getLogger(__name__)


class OpenAICompatibleChatModel:
    """Sends system + user messages and reports every failure as ``GenerationProviderError``.

    ``options`` holds only the generation parameters configured in ``.env``
    (``temperature`` and/or ``reasoning_effort``, see ``Settings.llm_generation_options``):
    a parameter that is not set is never sent, because reasoning models reject
    ``temperature``. When the provider rejects ``reasoning_effort`` (a classic model) and
    ``fallback_options`` is given, the request is retried once with those options, and
    every later request uses them. The Responses API is disabled: only Chat Completions
    is implemented by every compatible provider.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        options: Mapping[str, float | str] | None = None,
        fallback_options: Mapping[str, float | str] | None = None,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = SecretStr(api_key or _PLACEHOLDER_KEY)
        self._timeout_seconds = timeout_seconds
        self._client = self._build_client(options or {})
        self._fallback_options = fallback_options

    def _build_client(self, options: Mapping[str, float | str]) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout_seconds,
            # One retry: a generation that timed out may still be billed upstream.
            max_retries=1,
            use_responses_api=False,
            **dict(options),
        )

    @property
    def name(self) -> str:
        return self._model_name

    def _invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        try:
            return self._client.invoke(messages)
        except openai.BadRequestError as error:
            if self._fallback_options is None or "reasoning_effort" not in str(error):
                raise
            _log.warning(
                "chat model %r rejected reasoning_effort; retrying with %s from now on",
                self._model_name,
                dict(self._fallback_options) or "provider defaults",
            )
            self._client = self._build_client(self._fallback_options)
            self._fallback_options = None
            return self._client.invoke(messages)

    def complete(self, messages: Sequence[ChatMessage]) -> Completion:
        converted: list[BaseMessage] = [
            SystemMessage(m.content) if m.role == "system" else HumanMessage(m.content)
            for m in messages
        ]
        try:
            reply = self._invoke(converted)
        except (openai.OpenAIError, ValueError, TypeError, KeyError) as error:
            # ValueError/TypeError/KeyError: HTTP 200 with an error object or no choices
            # (some routers, e.g. OpenRouter, report upstream failures this way). The full
            # error goes to the log, so a local bug is not mistaken for a provider failure.
            _log.warning("chat completion failed", exc_info=error)
            raise GenerationProviderError(
                f"chat model {self._model_name!r} at {self._base_url} failed: "
                f"{type(error).__name__}"
            ) from error
        if not isinstance(reply, AIMessage) or not isinstance(reply.content, str):
            raise GenerationProviderError(f"chat model {self._model_name!r} returned no text")
        usage = reply.usage_metadata
        return Completion(
            text=reply.content.strip(),
            model=str(reply.response_metadata.get("model_name") or self._model_name),
            prompt_tokens=usage["input_tokens"] if usage else None,
            completion_tokens=usage["output_tokens"] if usage else None,
        )
