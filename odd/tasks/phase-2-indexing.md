# Phase 2 — Indexing (embeddings in pgvector)

Objective: embed every chunk set of a collection through any OpenAI-compatible embeddings API and persist the vectors in PostgreSQL/pgvector, with the collection bound to the embedding model and dimension.
Branch: `feat/foundation-ingestion` (continues the pushed branch) · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "2. Indexação — Embeddings persistidos no pgvector". MVP items "4. Geração de embeddings locais." and "5. Indexação no PostgreSQL com pgvector."
- **S2 — Provider.** "Os LLMs e modelos de embeddings devem ser via API, pode usar ollama ou llama.cpp mas a ideia é chamar openai, openrouter, ou outro openai compatible". Model, URL and key from `.env` ("as definições de modelos, api keys e configurações devem ficar no .env").
- **S3 — Binding.** Agreed in the Phase 1 interview: a collection is bound to the embedding model and dimension used to index it; a different model requires re-indexing or a new collection.
- **S4 — Isolation.** "Falhas críticas de isolamento = 0": embeddings are scoped by collection like documents and chunks.
- **S5 — Quality bar.** TDD, layered architecture, docs in EN and PT explaining purpose, behavior and why this solution ("deve explicar com clareza para que serve o que faz e por que foi escolhida essa solução e não outra").

Assumptions (not user-stated): indexing is an explicit action (not on upload); progress is committed per batch so a failed run resumes; ANN indexes (HNSW) are created in Phase 3 together with retrieval.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S3 | inline | done | eb3f939 |
| T2 | S1 S3 S4 | inline | done | eb3f939 |
| T3 | S1 S4 | inline | done | eb3f939 |
| T4 | S2 | inline | done | 9a5faa5 |
| T5 | S1 S2 | inline | done | 9a5faa5 |
| T6 | S5 | inline | done | (this commit) |

- T1 Domain: `EmbeddingSpec`, collection binding rules.
- T2 Application: `Embedder` + `EmbeddingRepository` ports, `IndexingService` (batched, resumable, rebuild), in-memory adapter.
- T3 PostgreSQL: migration 0002 (pgvector extension, binding columns, `chunk_embeddings`), repository, contract tests.
- T4 OpenAI-compatible embedder via LangChain `OpenAIEmbeddings`, tested against a local stub HTTP server.
- T5 API endpoints (index, status), composition root, Streamlit indexing controls.
- T6 Docs EN/PT, README dependencies, ADR.

Route evidence: no Writer trigger reason (sequential, dependent units); parent writes inline.

## Log

- L1 (2026-10-30, user): "3" — answering "Deixar como está e seguir para a Fase 2" after the whole-branch review exceeded the context budget.
- L2 Environment: pgvector unavailable in the system Postgres; local verification uses `pgserver` (PostgreSQL + pgvector 0.6.2) in a scratch venv. Compose uses `pgvector/pgvector:pg17`.
- L3 Discovery: `langchain-openai` 1.7.3 `OpenAIEmbeddings.check_embedding_ctx_length` defaults to True (tiktoken token ids sent instead of text); must be False for non-OpenAI providers.
- L4 Verification: 136 passed without DB; contracts also passed against PostgreSQL + pgvector 0.6.2 (pgserver), 13 new tests incl. embedding isolation/binding/rebuild. `alembic check` no drift; 0002 downgrade/upgrade OK. ruff + mypy --strict clean. E2E: uvicorn + pgvector + local stub /v1/embeddings: upload -> index structure-800-120 (2 chunks embedded, 3d) -> status -> rows in chunk_embeddings. Not verified: a real provider (OpenAI/Ollama) and Docker Compose.
- L5 Review: whole-branch candidate exceeded review budget earlier; user chose to leave it. Phase 2 commits pending assessment.
- L6 `.env.example` cannot be edited from this environment (permission deny on `.env*`); EMBEDDING_* variables already present there.
