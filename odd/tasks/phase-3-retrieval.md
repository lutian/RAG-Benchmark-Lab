# Phase 3 — Retrieval (vector, full-text, hybrid)

Objective: search a chunk set of a collection by vector similarity, PostgreSQL full-text and a hybrid fusion of both, with filters, per-stage latency and per-method ranks for diagnosis.
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "3. Recuperação    Vector search, full-text search e híbrida". MVP item "6. Recuperação por vetor, texto completo e híbrida."
- **S2 — Flow.** "Pergunta → Busca selecionada → Filtros → Reranking → Contexto → LLM → Resposta": the caller selects the search method and may filter (by document).
- **S3 — Experiment.** "comparar recuperação vetorial, lexical, híbrida e híbrida com reranking no mesmo conjunto de perguntas. Registrar qualidade, latência e consumo de recursos." Retrieval returns latency per stage.
- **S4 — Diagnosis.** "por que o chunking afeta a recuperação, quando BM25 supera embeddings": results expose the score and the rank each method gave every hit; the UI compares methods side by side.
- **S5 — Isolation.** "Falhas críticas de isolamento = 0": every search is scoped by collection and chunk set.
- **S6 — Quality bar.** TDD, layered architecture, docs in EN and PT with purpose, behavior and why ("deve explicar com clareza para que serve o que faz e por que foi escolhida essa solução e não outra").

Assumptions (not user-stated): full-text language is chosen per collection at creation (default from `.env`); hybrid uses Reciprocal Rank Fusion; vector search is exact (no ANN index) so the benchmark measures strategies, not ANN error; vector/hybrid search refuses a chunk set that is not fully indexed.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S1 S4 | inline | done | 7e8c51b |
| T2 | S1 S2 S3 S5 | inline | done | 7e8c51b |
| T3 | S1 S5 | inline | done | 7e8c51b |
| T4 | S2 S3 S4 | inline | done | baee8b2 |
| T5 | S6 | inline | done | 8a7d4fd |

- T1 Domain: `SearchMethod`, `ScoredChunk`, reciprocal rank fusion; full-text language on `Collection`.
- T2 Application: `SearchRepository` port, `RetrievalService` (method dispatch, filters, timings, binding/coverage checks), in-memory adapter.
- T3 PostgreSQL: migration 0003 (collection language, `chunks.search_vector` + GIN), exact cosine search, full-text search, contract tests.
- T4 API `POST /collections/{id}/search`, collection language in create; Streamlit "Retrieval" page comparing methods side by side.
- T5 Docs EN/PT (layers, configuration, ADR-013/014, ADR-012 update), README.

Route evidence: no Writer trigger reason (sequential, dependent units); parent writes inline. Risk: item 1 (migration 0003, additive with backfill).

## Log

- L1 (2026-10-31, user): "sim, faz o push. E continua até entregar todos os passos. Só para se precisar de minha definição"
- L2 Decision: exact KNN instead of the HNSW indexes planned in ADR-012 — ANN recall loss would contaminate the strategy comparison, benchmark corpora are small, and HNSW with filters loses recall without pgvector 0.8 iterative scans. Recorded as ADR-014; ADR-012 trade-off updated.
- L3 Decision: lexical ranking is PostgreSQL `ts_rank_cd` over an OR query (not true BM25; no BM25 extension ships in the pgvector image). Recorded as ADR-013.
- L4 Verification: 185 passed with TEST_DATABASE_URL (PostgreSQL + pgvector 0.6.2 via pgserver), contracts for memory and postgres incl. 4 search contracts and 3 PostgreSQL-only full-text tests. Migration 0003: downgrade to 0002 with data → upgrade backfilled `'bank':1 'rais':2 'rate':3`; `alembic check` no drift; round trip OK. ruff + mypy --strict clean. E2E uvicorn + pgvector + stub /v1/embeddings: vector before indexing → 409 "not indexed"; after indexing, vector/fulltext/hybrid return ranked hits with per-method ranks and timings. Not verified: Docker Compose, real provider.
- L5 Behaviour change: a blank collection name or duplicate chunking configs used to raise an unhandled ValueError (HTTP 500); now InvalidRequestError → 422.
- L6 `.env.example` cannot be edited here: DEFAULT_FULLTEXT_LANGUAGE, RETRIEVAL_CANDIDATES, HYBRID_RRF_K documented in docs/configuration.md instead.
