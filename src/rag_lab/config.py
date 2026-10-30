"""Runtime configuration loaded from environment variables and the ``.env`` file.

Every model name, endpoint and API key lives in ``.env`` so that switching the LLM
or embedding provider (OpenAI, OpenRouter, Ollama, llama.cpp, ...) never requires a
code change. Real environment variables take precedence over the file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_temperature: float = 0.0
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

    # UI -> API
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (read once, then cached)."""
    return Settings()
