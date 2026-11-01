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
    ``temperature``. The Responses API is disabled: only Chat Completions is implemented
    by every compatible provider.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        options: Mapping[str, float | str] | None = None,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._client = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key or _PLACEHOLDER_KEY),
            timeout=timeout_seconds,
            # One retry: a generation that timed out may still be billed upstream.
            max_retries=1,
            use_responses_api=False,
            **dict(options or {}),
        )

    @property
    def name(self) -> str:
        return self._model_name

    def complete(self, messages: Sequence[ChatMessage]) -> Completion:
        converted: list[BaseMessage] = [
            SystemMessage(m.content) if m.role == "system" else HumanMessage(m.content)
            for m in messages
        ]
        try:
            reply = self._client.invoke(converted)
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
