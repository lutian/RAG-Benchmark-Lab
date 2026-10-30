# Architecture decision records

Each record states the decision first, then the alternatives that were rejected and why. Status values: **Accepted** (implemented), **Planned** (agreed, implemented in a later phase).

> Other languages: [Português](../pt-BR/decisions.md)

| ADR | Decision | Status |
|-----|----------|--------|
| [001](#adr-001-hexagonal-architecture) | Hexagonal architecture | Accepted |
| [002](#adr-002-a-chunking-independent-reference-layer) | Chunking-independent reference layer (character spans) | Accepted |
| [003](#adr-003-chunk-size-in-characters) | Chunk size measured in characters | Accepted |
| [004](#adr-004-own-chunkers-langchain-at-the-provider-edge) | Own chunkers; LangChain only at the provider edge | Accepted |
| [005](#adr-005-postgresql-for-everything) | PostgreSQL + pgvector for documents, vectors and full-text | Accepted (vectors planned) |
| [006](#adr-006-openai-compatible-providers-configured-in-env) | OpenAI-compatible providers configured in `.env` | Accepted (clients planned) |
| [007](#adr-007-streamlit-with-custom-css) | Streamlit with custom CSS | Accepted |
| [008](#adr-008-collections-as-the-isolation-boundary) | Collections as the isolation boundary | Accepted |
| [009](#adr-009-local-cross-encoder-plus-rerank-api-adapter) | Local cross-encoder plus `/rerank` API adapter | Planned |
| [010](#adr-010-synthetic-golden-questions-with-human-review) | Synthetic golden questions with human review | Planned |

---

## ADR-001: Hexagonal architecture

**Decision:** domain → application (use cases + ports) → adapters, wired in one composition root.

**Why:** the project exists to *compare* strategies and providers. Ports turn each of them into a swappable implementation, while the core logic being measured stays fixed. It also lets most tests run without infrastructure.

| Alternative | Why not |
|-------------|---------|
| Layered MVC around FastAPI | Business rules end up in routes and ORM models; swapping storage or providers touches everything |
| LangChain-centric app (chains everywhere) | Fast to start, but the framework becomes the architecture; hard to test and to explain each step |

## ADR-002: A chunking-independent reference layer

**Decision:** parse each document once into a canonical text. Store pages, sections, paragraphs, chunks and (later) golden labels as `[start, end)` spans over it, and measure relevance by span overlap.

**Why:** golden labels tied to chunk ids break whenever the chunking changes, which makes a chunking × retrieval matrix impossible. Spans make labels permanent and chunk sets disposable.

| Alternative | Why not |
|-------------|---------|
| Label chunk ids | Labels must be redone for every chunking config |
| Label documents only | Too coarse: a 50-page PDF is "relevant" for every question about it |
| Match by text similarity | Fuzzy and non-reproducible; thresholds become a hidden variable |

## ADR-003: Chunk size in characters

**Decision:** `chunk_size` and `chunk_overlap` count characters.

**Why:** token counts depend on each embedding model's tokenizer. Characters keep chunk sets identical across providers, so comparing embedding providers compares only the embeddings. **Trade-off:** chunk sizes in tokens vary a little between models. That is acceptable, because limits are far above typical sizes.

## ADR-004: Own chunkers, LangChain at the provider edge

**Decision:** chunking algorithms are written in the domain and offset-exact by construction. LangChain is used in adapters for LLM and embedding clients (phases 2 and 5).

**Why:** exact offsets are the foundation of ADR-002. Library splitters return strings and recover offsets afterwards. Writing about 150 lines of tested code is cheaper than adapting around that.

## ADR-005: PostgreSQL for everything

**Decision:** one PostgreSQL holds the reference layer, pgvector embeddings (phase 2) and full-text indexes (phase 3). Access goes through SQLAlchemy Core, and migrations through Alembic.

**Why:** vector, lexical and hybrid retrieval run in the same engine, so latency comparisons are fair, and hybrid queries can be one SQL statement. A single service also keeps `docker compose up` simple.

| Alternative | Why not |
|-------------|---------|
| Dedicated vector DB (Qdrant, Weaviate) + Elasticsearch | Two extra services; cross-engine latency skews the comparison |
| SQLAlchemy ORM | Domain entities are frozen dataclasses; the ORM needs mutable mapped classes or duplicate models |
| `Base.metadata.create_all()` | No schema history; Alembic gives versioned, reviewable migrations |

## ADR-006: OpenAI-compatible providers configured in `.env`

**Decision:** LLM and embeddings are called through the OpenAI-compatible API with `*_BASE_URL`, `*_API_KEY` and `*_MODEL` from `.env`, configured independently.

**Why:** one protocol covers OpenAI, OpenRouter, Ollama, llama.cpp, vLLM and others. Switching providers is a configuration change, which is exactly what a benchmark lab needs. **Consequence:** collections are bound to the embedding model and dimension used to index them.

## ADR-007: Streamlit with custom CSS

**Decision:** Streamlit for the UI, styled after the reference design (white canvas, `#2563EB` accent, Inter, cards, badges, highlighted evidence). The UI is an HTTP client of the API.

| Alternative | Why not |
|-------------|---------|
| React / Next.js | Closer to the design pixel by pixel, but a second language and toolchain for a project whose core is retrieval |
| Streamlit importing services directly | Couples UI and core; the API would not be the single tested entry point |

## ADR-008: Collections as the isolation boundary

**Decision:** every read takes a `collection_id` at the port level. Cross-collection access returns 404. Contract tests prove it on every adapter.

**Why:** "zero critical isolation failures" is a project goal. Putting the scope in the *signature* makes leaking impossible by construction, rather than dependent on remembering a filter. Multi-tenant authentication was deliberately left out of the MVP.

## ADR-009: Local cross-encoder plus rerank API adapter

**Decision (planned, phase 4):** the default reranker is a Sentence Transformers cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) running in-process. A second adapter calls HTTP `/rerank` endpoints (Jina/Cohere style, TEI, llama.cpp, vLLM). The choice comes from `RERANKER_PROVIDER`.

**Why:** the OpenAI API has no rerank endpoint. A small local model works offline on CPU, and the API adapter avoids shipping torch where a rerank service already exists.

## ADR-010: Synthetic golden questions with human review

**Decision (planned, phase 6):** the LLM proposes questions from sampled passages; the source passage becomes the labelled span; a Streamlit screen lets a human edit, approve or reject them and add their own.

**Why:** the corpus is arbitrary, so no labels exist up front. Fully synthetic questions copy the passage wording and inflate lexical (BM25) scores. Human review keeps the benchmark honest.
