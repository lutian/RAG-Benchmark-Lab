# Configuration — `.env`

All models, endpoints, API keys and tunables live in `.env`. Nothing is hard-coded, so switching from OpenAI to OpenRouter, Ollama or llama.cpp needs no code change.

> Other languages: [Português](../pt-BR/configuration.md)

## Quick path

```bash
cp .env.example .env
# edit LLM_* and EMBEDDING_* for your provider, then:
docker compose up --build
```

## How it is loaded

`rag_lab/config.py` defines a typed `Settings` class (pydantic-settings):

- each field maps to an upper-case variable (`llm_model` → `LLM_MODEL`);
- real environment variables **override** `.env`;
- API keys are `SecretStr`: they never appear in `repr()`, logs or `model_dump()` (a test checks this);
- invalid values fail fast at startup (for example `RERANKER_PROVIDER=magic`).

**Why pydantic-settings and not `os.getenv`?** You get typed, validated and documented configuration in one place, and secrets are masked by default.

## Variables

| Variable | Default | Used by | Notes |
|----------|---------|---------|-------|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `rag` / — / `rag_lab` | Compose `db` | Password is **required**; Compose refuses to start without it |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag_lab` | API, Alembic | Host `db` inside Compose, `localhost` from your machine |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Generation (phase 5) | Any OpenAI-compatible `/chat/completions` endpoint |
| `LLM_API_KEY` | empty | Generation | Use any non-empty value for Ollama |
| `LLM_MODEL` | `gpt-4o-mini` | Generation | Provider model id |
| `LLM_REASONING_EFFORT` | unset | Generation | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; supported values vary by model. Sent as `reasoning_effort` only when set |
| `LLM_TEMPERATURE` | unset | Generation | Sent only when set. Startup fails if combined with a reasoning effort other than `none` (see below) |
| `LLM_TIMEOUT_SECONDS` | `60` | Generation | Request timeout |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | OpenAI | Indexing (phase 2) | Any OpenAI-compatible `/embeddings` endpoint |
| `EMBEDDING_DIMENSIONS` | `1536` | Indexing | Must match the model (e.g. `768` for `nomic-embed-text`). Not sent to the provider: used to validate returned vectors and bind the collection |
| `EMBEDDING_BATCH_SIZE` | `64` | Indexing | Texts per embeddings request |
| `RERANKER_PROVIDER` | `local` | Reranking | `local` (cross-encoder in-process, needs the `local-rerank` extra) or `api` (HTTP `/rerank`) |
| `RERANKER_MODEL` / `RERANKER_BASE_URL` / `RERANKER_API_KEY` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking | Hugging Face model id for `local`, model name sent to the API for `api`; URL (e.g. `http://host:8080/v1`) and key only for `api` |
| `RERANK_CANDIDATES` | `30` | Reranking | Hybrid hits handed to the reranker; more = better chance to rescue a buried chunk, slower |
| `INSTALL_LOCAL_RERANKER` | `true` | Docker build | Compose build argument; `false` builds an image without torch (use with `RERANKER_PROVIDER=api`) |
| `MAX_UPLOAD_MB` | `25` | Ingestion | Larger files get HTTP 413 |
| `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` | `800` / `120` | Ingestion | Characters; defaults for new collections |
| `DEFAULT_FULLTEXT_LANGUAGE` | `english` | Collections | PostgreSQL text search configuration for new collections (`portuguese`, `spanish`, `simple`…). Validated at startup |
| `RETRIEVAL_CANDIDATES` | `50` | Retrieval | Chunks each method contributes before hybrid fusion |
| `HYBRID_RRF_K` | `60` | Retrieval | Reciprocal Rank Fusion constant; higher flattens the advantage of top positions |
| `API_URL` | `http://localhost:8000` | UI | `http://api:8000` inside Compose |

## Provider recipes

| Provider | `*_BASE_URL` | `*_API_KEY` | Example models |
|----------|--------------|-------------|----------------|
| OpenAI | `https://api.openai.com/v1` | your key | `gpt-4o-mini`, `text-embedding-3-small` (1536) |
| OpenRouter | `https://openrouter.ai/api/v1` | your key | `meta-llama/llama-3.1-8b-instruct` |
| Ollama (Compose profile) | `http://ollama:11434/v1` | `ollama` | `llama3.2`, `nomic-embed-text` (768) |
| Ollama on the host | `http://host.docker.internal:11434/v1` | `ollama` | same |
| llama.cpp server | `http://<host>:8080/v1` | any | the GGUF model you loaded |

### Temperature vs reasoning effort

OpenAI's reasoning models reject sampling parameters: *"When reasoning effort is not `none`, remove `temperature`, `top_p`, and `top_logprobs`"* ([Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model)). GPT-6.1 Sol and GPT-6 Astra do not support `none`, so they **never** accept `temperature`.

| Model type | Configure | Leave unset |
|------------|-----------|-------------|
| Reasoning model (e.g. `gpt-6.1-sol`) | `LLM_REASONING_EFFORT=low` (or higher) | `LLM_TEMPERATURE` |
| GPT-6 Sol / Luna with reasoning off | `LLM_REASONING_EFFORT=none`, optionally `LLM_TEMPERATURE` | — |
| Classic chat model (e.g. `gpt-4o-mini`, most Ollama/llama.cpp models) | `LLM_TEMPERATURE=0.0` for reproducible runs | `LLM_REASONING_EFFORT` |

**Why no hard-coded model list?** The rule depends on the reasoning effort, not just the model name, and new models ship often. The app sends only what you configure. It rejects at startup the one combination the API documents as invalid (temperature with active reasoning), instead of failing on the first request.

LLM and embeddings are configured **independently**: for example, generate with OpenRouter and embed locally with Ollama.

> **Embedding binding:** a collection's vectors are only comparable with vectors from the same model and dimension. Changing `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` requires re-indexing, or a new collection (enforced from phase 2).
