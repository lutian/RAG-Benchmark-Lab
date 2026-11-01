# Phase 4 — Reranking (local cross-encoder and /rerank API)

Objective: reorder hybrid candidates with a reranker (local cross-encoder or an HTTP `/rerank` API) as the fourth compared strategy, with its own latency and the rank shift it caused.
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "4. Reranking    Reordenação de candidatos e comparação". MVP item "7. Reranking com cross-encoder local."
- **S2 — Experiment.** "comparar recuperação vetorial, lexical, híbrida e híbrida com reranking no mesmo conjunto de perguntas. Registrar qualidade, latência e consumo de recursos." Target "Estratégias comparadas 4".
- **S3 — Diagnosis.** "quando o reranking ajuda": each reranked hit shows its rank before reranking.
- **S4 — Provider.** Agreed in the Phase 1 interview: local cross-encoder (Sentence Transformers) by default plus an adapter for HTTP `/rerank` APIs; configured in `.env` (`RERANKER_*`).
- **S5 — Quality bar.** TDD, layered architecture, docs in EN and PT with purpose, behavior and why.

Assumptions (not user-stated): reranking applies to hybrid candidates only (the fourth strategy); the API adapter speaks the Jina/Cohere `/rerank` schema (also served by llama.cpp, vLLM, TEI-compatible responses accepted); sentence-transformers is an optional extra installed in the Docker image with CPU-only torch.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S1 S2 S3 | inline | done | f038c8e |
| T2 | S4 | inline | done | ea3192e, 244cce9 |
| T3 | S2 S3 | inline | done | ea3192e |
| T4 | S5 | inline | done | ff5d9fb |

- T1 Domain + application: `SearchMethod.HYBRID_RERANK`, `Reranker` port, `RerankerError`, reranking in `RetrievalService` (candidates, `rerank_ms`, `hybrid` rank kept).
- T2 Adapters: `CrossEncoderReranker` (lazy model load), `HttpReranker` (Jina/Cohere/TEI), composition root, optional dependency + Dockerfile.
- T3 API timings, Streamlit fourth column.
- T4 Docs EN/PT, ADR-009 accepted, README.

Route evidence: sequential dependent units, parent writes inline. Risk: item 5 (new optional dependency sentence-transformers/torch, Dockerfile).

## Log

- L1 (2026-10-31, user): "sim, faz o push. E continua até entregar todos os passos. Só para se precisar de minha definição"
- L2 (2026-11-01, user): "sempre faça a revisão com gentle-ai, não pergunte mais para esse projeto"
- L3 Verification: 206 passed with TEST_DATABASE_URL (pgvector via pgserver), incl. migration 0003 backfill test (phase 3 advisory). ruff + mypy --strict clean. Real cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, torch 2.15.0+cpu) scored the relevant chunk 6.6 vs -11. E2E uvicorn + pgvector + stub embeddings + local reranker: hybrid_rerank 200 with `ranks.hybrid`; first call 7.9 s (model load), then 35 ms. Not verified: Docker build with INSTALL_LOCAL_RERANKER, real /rerank API.
- L4 Native review (consent granted per L2): code slice 75fcbd5..ea3192e (lineage review-c2a1ec52d461660e, lenses risk/resilience/reliability) **approved**, acknowledged; dependency commit 244cce9 alone (review-14eecb0e426f22ff, risk/resilience) **approved**, acknowledged. Advisories fixed afterwards: malformed /rerank bodies → RerankerError (502), upstream error body no longer echoed, missing RERANKER_BASE_URL fails only reranked searches (not API startup), UI toggle to skip reranking. Accepted: model download on first reranked search (documented; benchmarks warm up), failed model load retried on next request.
