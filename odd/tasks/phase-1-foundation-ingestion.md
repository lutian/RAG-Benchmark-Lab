# Phase 0+1 — Foundation and Ingestion

Objective: project skeleton (hexagonal layout, config, Docker Compose, docs) plus the ingestion pipeline (loader, parser, chunk-independent reference layer, configurable chunking, metadata).
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase with user review after the phase · Test runner: `pytest` (via `uv run pytest`).

## Specs

- **S1 — Project identity and docs.** "É um projeto para demonstrar habilidades, assim que tem que ter um readme bem completo explicando como instalar todos os módulos e librarias". README in EN + pt-BR. "Tem que ter documentação de cada módulo, camada, serviço" and "documentação em camadas em PT também" — layer/module docs in EN and PT. "as documentações de cada camada, serviço, etc. deve explicar com clareza para que serve o que faz e por que foi escolhida essa solução e não outra (se achar que dá para evoluir a documentação fica a vontade para adicionar as informações que você considere relevantes)" — each doc covers purpose, behavior, and rationale vs alternatives (ADR-style).
- **S2 — Architecture.** "Tem que ter bem definida a arquitetura e separar por camadas quando for possível". Hexagonal: domain, application (use cases + ports), adapters (inbound/outbound). LangChain only inside adapters.
- **S3 — TDD.** "Tem que ter TDD". RED before GREEN for each rule.
- **S4 — Docker.** "Tem que poder rodar em contêiner Docker tudo documentado". Compose with Postgres+pgvector, API, UI.
- **S5 — Configuration.** "as definições de modelos, api keys e configurações devem ficar no .env". Provider choice: "Tem que poder escolher o provedor de LLM e de embeddings openAi friendly" (OpenAI-compatible base URL + key + model in `.env`).
- **S6 — Ingestion.** MVP items: "1. Upload e ingestão de arquivos PDF, Markdown e TXT." and "2. Extração de texto com preservação de metadados."
- **S7 — Chunking.** "3. Chunking configurável: tamanho fixo, recursivo e baseado em estrutura." and "o chunking deve ser variável: A solução é criar uma camada de referência independente do chunking". Canonical document text with stable char offsets; chunks are spans `[start, end)` over it; several chunking configs per collection.
- **S8 — Isolation.** "Falhas críticas de isolamento = 0" scoped to collections: every read is scoped by collection and tests prove no cross-collection leakage.
- **S9 — Design.** Streamlit + custom CSS following the reference screen (light background, blue accent, Inter-like sans font, three-panel layout, highlighted evidence).

Assumption: chunk size is measured in characters (tokenizer-agnostic).

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S2 S4 S5 S9 | inline | done | 6568d79, 024f0c5, 3b3171d |
| T2 | S3 S7 | inline | done | c1b1bc3 |
| T3 | S3 S6 | inline | done | a5ee18e |
| T4 | S3 S7 | inline | done | c4ae990 |
| T5 | S3 S6 S7 S8 | inline | done | 6d4fd0f |
| T6 | S6 S7 S9 | inline | done | 0094f31 |
| T7 | S1 | inline | done | (this commit) |

- T1 Skeleton: pyproject (uv), settings from `.env`, FastAPI `/health`, Streamlit shell + theme, Dockerfiles, Compose, Makefile, CI.
- T2 Domain reference layer: Span, Document (canonical text), structural elements (pages, sections), Chunk, ChunkingConfig.
- T3 Parsers: TXT, Markdown (heading sections), PDF (page spans) with metadata.
- T4 Chunkers: fixed, recursive, structure-based; invariant `chunk.text == document.text[start:end]`.
- T5 Application use cases + ports, in-memory and Postgres (SQLAlchemy + Alembic) adapters, isolation tests.
- T6 API endpoints (collections, upload, chunk sets) + Streamlit ingestion page.
- T7 README EN/pt-BR + layer/module docs EN/PT.

Route evidence: no Writer trigger reason (units are sequential and depend on the domain), so the parent writes inline.

## Log

- L1 (2026-10-30, user, original request — idea document excerpt): "quero criar um projeto de nome RAG Benchmark Lab ... Não será simplesmente um chatbot que lê PDFs. Será uma ferramenta que mostra por que uma estratégia de recuperação funciona melhor do que outra, com experimentos reproduzíveis." Requirements: complete README, layered architecture, per-module docs, TDD, Docker, selectable OpenAI-friendly LLM/embedding provider.
- L2 (2026-10-30, user decisions): Streamlit + custom CSS; any corpus ("a ideia é testar qualquer documento, não importa do que fale nem de quem seja, o benchmark é o relevante"); golden set synthetic + manual review; isolation by collection; README bilingual; LLM/embeddings "via API, pode usar ollama ou llama.cpp mas a ideia é chamar openai, openrouter, ou outro openai compatible"; reranker local cross-encoder + `/rerank` API adapter; benchmark = chunking × retrieval matrix; phase-by-phase delivery.
- L3 (2026-10-30, user): "documentação em camadas em PT também"; "o chunking deve ser variável: A solução é criar uma camada de referência independente do chunking"; "as definições de modelos, api keys e configurações devem ficar no .env"; repo `https://github.com/lutian/RAG-Benchmark-Lab.git`.
- L5 (2026-10-30, user): "as documentações de cada camada, serviço, etc. deve explicar com clareza para que serve o que faz e por que foi escolhida essa solução e não outra (se achar que dá para evoluir a documentação fica a vontade para adicionar as informações que você considere relevantes)". S1 updated; docs will include ADRs.
- L4 Environment: no Docker or uv on the dev machine; uv used from a scratch venv; Postgres 17 binaries available (no pgvector) for a throwaway integration-test cluster. Docker Compose cannot be executed locally.
- L6 Verification: 93 passed, 7 skipped without DB; 100 passed with TEST_DATABASE_URL against local PostgreSQL 17 (contracts run on memory + Postgres). `alembic check`: no drift. ruff + mypy --strict clean. E2E smoke: uvicorn + local Postgres, PDF upload -> structure chunks carry page numbers [1], [2]; Streamlit health ok. Docker Compose NOT executed (no Docker on dev machine).
- L7 Review: T1 scaffold reviewed (risk lens) after splitting uv.lock out (lens_context_budget_exceeded on the combined commit); approved with 2 advisories (ports on all interfaces) fixed in 3b3171d. Lock-file commit 6568d79 not reviewable (generated, over budget). T2-T7 slice: assessed under_budget (pending).
- L8 RED evidence: each task's tests were run failing (import/collection errors or assertion failures) before implementation; exception: Chunk/Collection entity tests (T2) were written together with the code.
- L9 Next: Phase 2 (embeddings via OpenAI-compatible API + pgvector), after user review of Phase 1.
