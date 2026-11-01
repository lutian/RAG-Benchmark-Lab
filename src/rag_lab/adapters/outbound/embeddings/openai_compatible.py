"""Embedder for any OpenAI-compatible ``/embeddings`` API (OpenAI, OpenRouter, Ollama,
llama.cpp, vLLM, ...), built on LangChain's ``OpenAIEmbeddings``.
"""

from collections.abc import Sequence

import openai
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from rag_lab.application.errors import EmbeddingProviderError
from rag_lab.domain.embeddings import EmbeddingSpec

# The OpenAI SDK refuses an empty key; local servers (Ollama, llama.cpp) ignore it.
_PLACEHOLDER_KEY = "not-needed"


class OpenAICompatibleEmbedder:
    """Sends raw text in batches and reports every failure as ``EmbeddingProviderError``.

    Two LangChain defaults are changed on purpose:

    * ``check_embedding_ctx_length=False``: by default LangChain tokenizes with
      tiktoken and sends token ids, which only OpenAI understands. Raw text works
      with every compatible provider.
    * ``dimensions`` is never sent: most providers do not support shortening vectors.
      ``EMBEDDING_DIMENSIONS`` is used to validate what comes back instead.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._spec = EmbeddingSpec(model, dimensions)
        self._base_url = base_url
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_base=base_url,
            openai_api_key=SecretStr(api_key or _PLACEHOLDER_KEY),
            chunk_size=batch_size,
            check_embedding_ctx_length=False,
            request_timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(list(texts))
        except openai.OpenAIError as error:
            raise self._provider_error(error) from error

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(text)
        except openai.OpenAIError as error:
            raise self._provider_error(error) from error

    def _provider_error(self, error: Exception) -> EmbeddingProviderError:
        return EmbeddingProviderError(
            f"embedding request to {self._base_url} with model {self._spec.model!r} "
            f"failed: {type(error).__name__}: {error}"
        )
