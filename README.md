# RAG Benchmark Lab

**A lab that shows *why* one retrieval strategy beats another, with reproducible experiments on your own documents.**

This is not a chatbot that reads PDFs. You upload any corpus. The lab chunks it in several ways and retrieves with lexical, vector, hybrid and reranked search. It then measures quality, latency and cost against a golden dataset, so you can diagnose whether a bad answer comes from **retrieval** or from **generation**.

> 🇧🇷 [Leia em português](README.pt-BR.md)

## What it does

1. Upload documents into collections; they are split into chunks and indexed with embeddings and PostgreSQL full-text search.
2. Retrieve with vector search, full-text search or both combined (Reciprocal Rank Fusion), with optional cross-encoder reranking.
3. **Ask** answers with citations using any LLM behind an OpenAI-compatible API (OpenAI, OpenRouter, Ollama, llama.cpp).
4. Benchmarks run golden-dataset questions, store the metrics and record the exact environment (models and parameters), so configurations can be compared reproducibly.

## Status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1. Ingestion | Upload, parsing (PDF/MD/TXT), metadata, configurable chunking, chunk-independent reference layer | ✅ Done |
| 2. Indexing | Embeddings from any OpenAI-compatible API, persisted in pgvector; resumable, model-bound collections | ✅ Done |
| 3. Retrieval | Exact vector search, PostgreSQL full-text (per-collection language) and hybrid RRF; filters, per-stage latency, rank per method | ✅ Done |
| 4. Reranking | Hybrid + rerank as the fourth strategy: local cross-encoder (Sentence Transformers) or any `/rerank` API; rank before/after reranking | ✅ Done |
| 5. Generation | Grounded answers from any OpenAI-compatible LLM: numbered citations checked by parsing, two abstention paths (no evidence / model) | ✅ Done |
| 6. Evaluation | Golden dataset (LLM drafts + human review, export/import), chunking × strategy benchmark: Recall@k, MRR, latency, CPU, citation validity, LLM judge; Markdown/JSON report and `rag-lab` CLI | ✅ Done |
| 7. Presentation | Benchmark dashboard, sample corpus with a 52-question golden dataset, `rag-lab demo`, benchmark guide | ✅ Done |

## Quick start (Docker)

Requirements: Docker with Compose v2.

```bash
git clone https://github.com/lutian/RAG-Benchmark-Lab.git
cd RAG-Benchmark-Lab
cp .env.example .env            # set POSTGRES_PASSWORD and your provider keys
docker compose up --build -d
```

To check the Docker setup itself without any provider key, run `make compose-smoke` before starting the stack (or after `docker compose down`). It builds a throwaway copy of `db`, `api` and `ui`, ingests a sample document, runs a full-text search, and removes the copy.

| Service | URL |
|---------|-----|
| UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 · interactive docs at http://localhost:8000/docs |
| PostgreSQL + pgvector | `localhost:5432` (loopback only) |

If port 8000 is already in use on your machine, set `API_PORT` in `.env` (for example `API_PORT=8001`) and use that port instead. Containers still reach the API at `http://api:8000`.

Then open the UI, create a collection in the sidebar (rename or delete it there too, each behind a confirmation dialog), upload PDF, Markdown or TXT files, and inspect the chunk boundaries of each chunk set. Press **Index chunk set** to embed it with the provider configured in `.env`. On the **Retrieval** page, ask a question and compare vector, full-text, hybrid and reranked results side by side. The first reranked search downloads the cross-encoder (~90 MB) into the `models` volume. On the **Ask** page, get an answer with citations next to the sources it was given. On **Golden dataset**, generate questions with the LLM, add your own, and approve the ones the benchmark should use; then run the benchmark from the **Benchmark** page (or `docker compose exec api rag-lab benchmark --collection <name>`) and compare strategies on the dashboard.

Optional local models with Ollama:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
# in .env: LLM_BASE_URL=http://ollama:11434/v1, EMBEDDING_BASE_URL=http://ollama:11434/v1,
#          EMBEDDING_MODEL=nomic-embed-text, EMBEDDING_DIMENSIONS=768
```

## See it work in 5 minutes: the sample benchmark

The repository ships a small fictional handbook (`examples/corpus/`, 7 documents) with a reviewed golden dataset of 52 questions (`examples/golden.json`). One command ingests it, indexes every chunk set, imports the questions and runs the full benchmark:

```bash
make demo-docker          # inside Compose (report copied to ./reports)
# or locally: make demo   (uv, PostgreSQL from `docker compose up -d db`)
```

Then open **Benchmark** in the UI. Reference results ([`examples/sample-report.md`](examples/sample-report.md), MiniLM embeddings, local cross-encoder, one CPU):

| Strategy (`fixed-800-120`) | Recall@1 | Recall@5 | MRR@10 | p50 latency |
|----------------------------|----------|----------|--------|-------------|
| Vector | 71.2% | 98.1% | 0.816 | 24 ms |
| Full-text | 67.3% | 96.2% | 0.803 | 2 ms |
| Hybrid (RRF) | 71.2% | 100.0% | 0.838 | 27 ms |
| Hybrid + rerank | 96.2% | 100.0% | 0.978 | 1,855 ms |

How to read these numbers (and run your own corpus): [benchmark guide](docs/en/benchmark-guide.md).

## Try it with curl

```bash
# 1. Create a collection (one chunk set per strategy by default)
curl -s -X POST localhost:8000/collections -H 'content-type: application/json' \
     -d '{"name": "my-docs"}'
# Rename it, or delete it with everything in it (irreversible)
curl -s -X PATCH localhost:8000/collections/<collection_id> -H 'content-type: application/json' \
     -d '{"name": "team-docs"}'
curl -s -X DELETE localhost:8000/collections/<collection_id>

# 2. Upload a document
curl -s -X POST localhost:8000/collections/<collection_id>/documents -F "file=@report.pdf"

# 3. Compare how two strategies cut the same document
curl -s "localhost:8000/collections/<collection_id>/chunk-sets/fixed-800-120/chunks?document_id=<id>"
curl -s "localhost:8000/collections/<collection_id>/chunk-sets/structure-800-120/chunks?document_id=<id>"

# 4. Embed a chunk set (uses EMBEDDING_* from .env) and check progress
curl -s -X POST localhost:8000/collections/<collection_id>/chunk-sets/structure-800-120/index
curl -s localhost:8000/collections/<collection_id>/index

# 5. Search (method: vector | fulltext | hybrid | hybrid_rerank)
curl -s -X POST localhost:8000/collections/<collection_id>/search -H 'content-type: application/json' \
     -d '{"query": "What is the refund policy?", "chunk_set": "structure-800-120", "method": "hybrid_rerank", "top_k": 5}'

# 6. Answer with citations (uses LLM_* from .env)
curl -s -X POST localhost:8000/collections/<collection_id>/answer -H 'content-type: application/json' \
     -d '{"question": "What is the refund policy?", "chunk_set": "structure-800-120", "method": "hybrid"}'
```

## Local development (without Docker for the app)

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.12 | https://www.python.org/downloads/ |
| uv | ≥ 0.13 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (or `pipx install uv`) |
| PostgreSQL + pgvector | 17 | `docker compose up -d db` (easiest) |

```bash
uv sync                          # creates .venv with app + dev dependencies (from uv.lock)
cp .env.example .env             # set DATABASE_URL host to localhost
docker compose up -d db          # database only
make migrate                     # alembic upgrade head
make api                         # http://localhost:8000 (auto-reload)
make ui                          # http://localhost:8501 (set API_URL=http://localhost:8000)
```

### Libraries and why they are here

| Library | Role |
|---------|------|
| FastAPI, Uvicorn, python-multipart | HTTP API, ASGI server, file uploads |
| Pydantic, pydantic-settings | Request validation; typed `.env` configuration |
| SQLAlchemy 2 (Core), psycopg 3, Alembic | PostgreSQL access and versioned migrations |
| pypdf | PDF text extraction (BSD license) |
| langchain-openai (+ openai SDK) | Client for any OpenAI-compatible embeddings API |
| pgvector (Python) + `vector` extension | Vector column type for SQLAlchemy and PostgreSQL |
| Streamlit, httpx | UI and its HTTP client |
| pytest, Hypothesis, fpdf2 | Tests, property-based tests, PDF fixtures |
| ruff, mypy | Lint/format and strict type checking |

Optional extra `local-rerank` (`uv sync --extra local-rerank`, installed in the Docker image): **sentence-transformers** + **torch** (CPU build on Linux) for the local cross-encoder reranker. Skip it when `RERANKER_PROVIDER=api`. Each phase documents its new dependencies here.

### Common commands

| Command | What it does |
|---------|--------------|
| `make test` | Full test suite (PostgreSQL contract tests skip without `TEST_DATABASE_URL`) |
| `make test-integration` | Contract tests against a disposable PostgreSQL |
| `make lint` / `make format` | Ruff check / fix |
| `make typecheck` | mypy `--strict` |
| `make up` / `make down` / `make logs` | Docker Compose lifecycle |

## Configuration

Everything (models, endpoints, API keys, limits) is set in `.env`. Any **OpenAI-compatible** provider works for the LLM and for embeddings, configured independently: OpenAI, OpenRouter, Ollama, llama.cpp, vLLM… See [docs/en/configuration.md](docs/en/configuration.md).

### Reranker

The reranker works out of the box with these defaults (the Docker image already includes it; the model, ~90 MB, downloads on the first reranked search):

```dotenv
RERANKER_PROVIDER=local
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_BASE_URL=
RERANKER_API_KEY=
```

`RERANKER_BASE_URL` and `RERANKER_API_KEY` are only used with `RERANKER_PROVIDER=api` (Jina, Cohere, vLLM, llama.cpp). For a non-English corpus, use a multilingual model such as `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Models, API providers, pre-downloading and troubleshooting: [Reranker setup](docs/en/configuration.md#reranker-setup).

## Architecture in one picture

```text
 Streamlit UI ──HTTP──▶ FastAPI ──▶ Application (use cases, ports) ──▶ Domain (reference layer, chunking)
                                          │
                                          ├─▶ Parsers (PDF / Markdown / TXT)
                                          └─▶ PostgreSQL (SQLAlchemy Core)  |  In-memory store (tests)
```

Hexagonal architecture: the domain has no framework imports, and every external concern sits behind a port. The key idea is the **reference layer**. Each document is parsed once into a canonical text, and chunks and golden labels are character spans over it. Chunking can change freely without invalidating the evaluation dataset.

## Documentation

| Topic | English | Português |
|-------|---------|-----------|
| Architecture overview | [architecture](docs/en/architecture.md) | [arquitetura](docs/pt-BR/architecture.md) |
| Domain layer | [domain](docs/en/domain.md) | [domínio](docs/pt-BR/domain.md) |
| Application layer | [application](docs/en/application.md) | [aplicação](docs/pt-BR/application.md) |
| Adapters (parsers, DB, API, UI) | [adapters](docs/en/adapters.md) | [adaptadores](docs/pt-BR/adapters.md) |
| Configuration | [configuration](docs/en/configuration.md) | [configuração](docs/pt-BR/configuration.md) |
| Testing | [testing](docs/en/testing.md) | [testes](docs/pt-BR/testing.md) |
| Decisions (ADRs) | [decisions](docs/en/decisions.md) | [decisões](docs/pt-BR/decisions.md) |
| Running and reading the benchmark | [benchmark guide](docs/en/benchmark-guide.md) | [guia do benchmark](docs/pt-BR/benchmark-guide.md) |

## Validation targets (proposed, not results)

| Indicator | Target |
|-----------|--------|
| Evaluation questions | 50 |
| Strategies compared | 4 (vector, full-text, hybrid, hybrid + rerank) |
| Recall@5 | ≥ 85% |
| MRR@10 | ≥ 0.75 |
| Answers with valid citations | ≥ 95% |
| Critical isolation failures | 0 |

These targets may be adjusted to the corpus difficulty and the quality of the reference labels.

## Project layout

```text
src/rag_lab/
  domain/            spans, documents, chunks, collections, chunking, retrieval, answers, evaluation
  application/       services/ (ingestion, indexing, retrieval, answering, golden, benchmark),
                     ports/, prompts.py, reporting.py, run_codec.py, errors.py
  adapters/
    inbound/api/     FastAPI app, routes, schemas
    outbound/        parsers/, persistence/ (memory, postgres), embeddings/, reranking/, generation/
  ui/                Streamlit pages, theme, API client
  cli.py             rag-lab command line
  config.py          settings from .env
  bootstrap.py       composition root
migrations/          Alembic revisions 0001-0004
tests/               unit/, contracts/ (memory + PostgreSQL), integration/
docs/en, docs/pt-BR  layer docs, configuration, testing, ADRs
odd/tasks/           feature documents (specs, tasks, log) per phase
```
