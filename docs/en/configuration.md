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
| `LLM_TEMPERATURE` | unset | Generation | Sent only when set. With an active reasoning effort it is the fallback for models that reject `reasoning_effort` (see below) |
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
| Classic chat model (e.g. `gpt-4o-mini`, most Ollama/llama.cpp models) | `LLM_TEMPERATURE=0.0` for reproducible runs | `LLM_REASONING_EFFORT` (or keep it: see the fallback below) |

**Setting both (effort first, temperature as fallback).** With `LLM_REASONING_EFFORT` and `LLM_TEMPERATURE` both set, the first request sends only `reasoning_effort` (plus `temperature` when the effort is `none`). If the provider answers HTTP 400 naming `reasoning_effort`, the model is treated as a classic one: the request is retried once with only `temperature`, and every later request in that process uses it. The switch is logged as a warning, and each benchmark run records the rule in its environment (for example `gpt-6-sol (reasoning_effort=low; if reasoning_effort is rejected: temperature=0.0)`). Providers that silently ignore unknown parameters (some local servers) never trigger the fallback, so they run without `temperature`.

**Why no hard-coded model list?** The rule depends on the reasoning effort, not just the model name, and new models ship often. The app sends only what you configure and lets the provider's own rejection pick the fallback.

LLM and embeddings are configured **independently**: for example, generate with OpenRouter and embed locally with Ollama.

> **Embedding binding:** a collection's vectors are only comparable with vectors from the same model and dimension. Changing `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` requires re-indexing, or a new collection (enforced from phase 2).

## Reranker setup

The fourth strategy, `hybrid_rerank`, reorders the best `RERANK_CANDIDATES` hybrid hits with a cross-encoder: a model that reads the question and each chunk **together** and scores how well the chunk answers it. That is more accurate than comparing two separate embeddings, but too slow to run over the whole collection, which is why it only reranks a short list.

### Option A: local cross-encoder (default)

```dotenv
RERANKER_PROVIDER=local
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_BASE_URL=
RERANKER_API_KEY=
```

`RERANKER_BASE_URL` and `RERANKER_API_KEY` are ignored with `local`; leave them empty.

1. **Install the runtime.** Docker: nothing to do, the image installs it (`INSTALL_LOCAL_RERANKER=true` is the default). Local development: `uv sync --extra local-rerank` (sentence-transformers + CPU torch, about 1 GB).
2. **Get the model.** Nothing to do either: the model is downloaded from Hugging Face on the **first** reranked search and cached. In Docker the cache is the `models` volume (`HF_HOME=/app/data/hf`), so it survives restarts; locally it is `~/.cache/huggingface` (or `$HF_HOME`). To download it ahead of time (for example before a demo or on a machine that will go offline):

   ```bash
   docker compose exec api python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   # locally: uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

3. **Check it.** Run a search with `"method": "hybrid_rerank"` (or the Retrieval page). The first call takes a few seconds (load); the next ones are fast. Each hit shows its hybrid rank before reranking.

Choosing `RERANKER_MODEL` (any Sentence Transformers cross-encoder id from Hugging Face works):

| Model | Download | Languages | When to use it |
|-------|----------|-----------|----------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (default) | ~90 MB | English | English corpora; fast on CPU. Used for the sample report |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130 MB | English | A bit more accurate, about twice as slow |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~470 MB | Multilingual (incl. Portuguese) | Non-English corpora; still CPU-friendly |
| `BAAI/bge-reranker-base` | ~1.1 GB | English, Chinese | Higher quality, slower |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | Multilingual | Best quality of this list; slow on CPU, plan for a GPU or the API option |

Match the reranker to the corpus language: an English-only model still ranks Portuguese text, but with much weaker separation between relevant and irrelevant chunks. After changing the model, recreate the API so it reads the new `.env` (`docker compose up -d api`; `docker compose restart` does **not** reload `.env`). The benchmark report records the reranker used (`local: <model>`), so runs with different models stay comparable.

### Option B: a `/rerank` API

```dotenv
RERANKER_PROVIDER=api
RERANKER_MODEL=<model name sent to the API>
RERANKER_BASE_URL=<base URL; the app calls {RERANKER_BASE_URL}/rerank>
RERANKER_API_KEY=<key, sent as "Authorization: Bearer ..."; empty for none>
```

The app sends `{model, query, documents, top_n}` and reads `{"results": [{"index", "relevance_score"}]}`, the schema shared by these providers:

| Provider | `RERANKER_BASE_URL` | `RERANKER_MODEL` example | `RERANKER_API_KEY` |
|----------|---------------------|--------------------------|--------------------|
| Jina AI | `https://api.jina.ai/v1` | `jina-reranker-v2-base-multilingual` | your key |
| Cohere | `https://api.cohere.com/v2` | `rerank-v3.5` | your key |
| vLLM (`vllm serve BAAI/bge-reranker-v2-m3`) | `http://<host>:8000/v1` | `BAAI/bge-reranker-v2-m3` | empty, or the server's key |
| llama.cpp (`llama-server --reranking -m <reranker>.gguf`) | `http://<host>:8080/v1` | any (the loaded model answers) | empty |

Hugging Face Text Embeddings Inference (TEI) is **not** compatible: its `/rerank` expects `texts` instead of `documents`.

With the API option you do not need torch in the image: build with `INSTALL_LOCAL_RERANKER=false docker compose up --build` for a much smaller image. From a container, reach a server on your machine with `http://host.docker.internal:<port>/v1` (Docker Desktop; on Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` to the `api` service).

### When it fails

A reranked search that cannot rerank returns **HTTP 502** with the reason (the other three methods keep working):

| Message | Fix |
|---------|-----|
| `cannot load cross-encoder ... Install the 'local-rerank' extra or set RERANKER_PROVIDER=api` | Run `uv sync --extra local-rerank`, or rebuild the image with `INSTALL_LOCAL_RERANKER=true`; also check the model id and that Hugging Face is reachable on first use |
| `RERANKER_BASE_URL is required when RERANKER_PROVIDER=api` | Set the base URL |
| `reranker at ... returned HTTP 401` (or 4xx/5xx) | Check the key, the model name and the URL (it must not already end in `/rerank`) |
