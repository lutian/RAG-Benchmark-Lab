"""Settings are loaded from environment / .env and never leak secrets."""

from pathlib import Path

import pytest

from rag_lab.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DATABASE_URL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "RERANKER_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_reads_provider_configuration_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DATABASE_URL=postgresql+psycopg://u:p@db:5432/rag",
                "LLM_BASE_URL=https://openrouter.ai/api/v1",
                "LLM_API_KEY=sk-llm-secret",
                "LLM_MODEL=meta-llama/llama-3.1-8b-instruct",
                "EMBEDDING_BASE_URL=http://ollama:11434/v1",
                "EMBEDDING_API_KEY=ollama",
                "EMBEDDING_MODEL=nomic-embed-text",
                "EMBEDDING_DIMENSIONS=768",
                "RERANKER_PROVIDER=api",
            ]
        )
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/rag"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_api_key.get_secret_value() == "sk-llm-secret"
    assert settings.llm_model == "meta-llama/llama-3.1-8b-instruct"
    assert settings.embedding_base_url == "http://ollama:11434/v1"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.embedding_dimensions == 768
    assert settings.reranker_provider == "api"


def test_environment_variables_override_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.llm_model == "from-env"


def test_api_keys_are_not_exposed_in_repr(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=sk-llm-secret\nEMBEDDING_API_KEY=sk-emb-secret\n")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    rendered = repr(settings) + str(settings.model_dump())
    assert "sk-llm-secret" not in rendered
    assert "sk-emb-secret" not in rendered


def test_rejects_unknown_reranker_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_PROVIDER", "magic")

    with pytest.raises(ValueError, match="reranker_provider"):
        Settings(_env_file=None)  # type: ignore[call-arg]
