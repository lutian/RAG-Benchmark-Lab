"""Runtime configuration loaded from environment variables and the ``.env`` file.

Every model name, endpoint and API key lives in ``.env`` so that switching the LLM
or embedding provider (OpenAI, OpenRouter, Ollama, llama.cpp, ...) never requires a
code change. Real environment variables take precedence over the file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_lab.domain.collections import FULLTEXT_LANGUAGES


class Settings(BaseSettings):
    """Typed application settings. Field names map to upper-case env variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Persistence
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_lab"

    # Chat LLM (any OpenAI-compatible endpoint)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    # Both optional: a parameter that is not set is not sent to the provider.
    # Reasoning models (e.g. OpenAI GPT-6) reject temperature/top_p unless the
    # reasoning effort is "none"; GPT-6.1 Sol and GPT-6 Astra never accept it.
    # When both are set, the effort is sent first and the temperature is the fallback
    # for models that reject reasoning_effort (see llm_fallback_options).
    llm_temperature: float | None = None
    llm_reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    llm_timeout_seconds: float = 60.0

    # Embeddings (any OpenAI-compatible endpoint)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64

    # Reranker: local cross-encoder or an HTTP /rerank API
    reranker_provider: Literal["local", "api"] = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_base_url: str = ""
    reranker_api_key: SecretStr = SecretStr("")

    # Ingestion
    max_upload_mb: int = 25
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120
    # PostgreSQL text search configuration for new collections (stemming, stop words).
    default_fulltext_language: str = "english"

    # Retrieval
    retrieval_candidates: int = Field(default=50, gt=0)  # per method, before hybrid fusion
    hybrid_rrf_k: int = Field(default=60, gt=0)  # Reciprocal Rank Fusion constant
    rerank_candidates: int = Field(default=30, gt=0)  # fused chunks handed to the reranker

    # UI -> API
    api_url: str = "http://localhost:8000"

    @field_validator("default_fulltext_language")
    @classmethod
    def _known_fulltext_language(cls, value: str) -> str:
        if value not in FULLTEXT_LANGUAGES:
            raise ValueError(
                f"unknown full-text language {value!r}; use one of "
                f"{', '.join(sorted(FULLTEXT_LANGUAGES))}"
            )
        return value

    def llm_generation_options(self) -> dict[str, float | str]:
        """Parameters for the first chat request (only the ones set).

        The reasoning effort wins: the temperature goes along only when the effort is
        unset or ``none``, the only cases where reasoning models accept it.
        """
        options: dict[str, float | str] = {}
        if self.llm_reasoning_effort is not None:
            options["reasoning_effort"] = self.llm_reasoning_effort
        if self.llm_temperature is not None and self.llm_reasoning_effort in (None, "none"):
            options["temperature"] = self.llm_temperature
        return options

    def llm_fallback_options(self) -> dict[str, float | str] | None:
        """Parameters to retry with when the provider rejects ``reasoning_effort``.

        ``None`` when no reasoning effort is sent, so there is nothing to fall back from.
        """
        if self.llm_reasoning_effort is None:
            return None
        if self.llm_temperature is None:
            return {}
        return {"temperature": self.llm_temperature}


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (read once, then cached)."""
    return Settings()
