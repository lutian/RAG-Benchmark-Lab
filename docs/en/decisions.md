# Architecture decision records

Each record states the decision first, then the alternatives that were rejected and why. Status values: **Accepted** (implemented), **Planned** (agreed, implemented in a later phase).

> Other languages: [Português](../pt-BR/decisions.md)

| ADR | Decision | Status |
|-----|----------|--------|
| [001](#adr-001-hexagonal-architecture) | Hexagonal architecture | Accepted |
| [002](#adr-002-a-chunking-independent-reference-layer) | Chunking-independent reference layer (character spans) | Accepted |
| [003](#adr-003-chunk-size-in-characters) | Chunk size measured in characters | Accepted |
| [004](#adr-004-own-chunkers-langchain-at-the-provider-edge) | Own chunkers; LangChain only at the provider edge | Accepted |
| [005](#adr-005-postgresql-for-everything) | PostgreSQL + pgvector for documents, vectors and full-text | Accepted |
| [006](#adr-006-openai-compatible-providers-configured-in-env) | OpenAI-compatible providers configured in `.env` | Accepted (embeddings done, chat planned) |
| [007](#adr-007-streamlit-with-custom-css) | Streamlit with custom CSS | Accepted |
| [008](#adr-008-collections-as-the-isolation-boundary) | Collections as the isolation boundary | Accepted |
| [009](#adr-009-local-cross-encoder-plus-rerank-api-adapter) | Local cross-encoder plus `/rerank` API adapter | Accepted |
| [010](#adr-010-synthetic-golden-questions-with-human-review) | Synthetic golden questions with human review | Accepted |
| [011](#adr-011-explicit-resumable-indexing) | Explicit, resumable indexing with model-bound collections | Accepted |
| [012](#adr-012-one-vector-table-without-a-fixed-dimension) | One vector table without a fixed dimension | Accepted |
| [013](#adr-013-postgresql-full-text-as-the-lexical-baseline) | PostgreSQL full-text (OR query, `ts_rank_cd`) as the lexical method | Accepted |
| [014](#adr-014-exact-vector-search-for-the-benchmark) | Exact vector search, no ANN index | Accepted |
| [015](#adr-015-numbered-citations-checked-by-parsing-and-a-sentinel-for-abstention) | Numbered citations checked by parsing, and a sentinel for abstention | Accepted |
| [016](#adr-016-benchmark-runs-as-versioned-json-documents-plus-a-cli) | Benchmark runs as versioned JSON documents, plus a CLI | Accepted |

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

**Decision:** the default reranker is a Sentence Transformers cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) running in-process and loaded on first use. A second adapter calls HTTP `/rerank` endpoints (Jina/Cohere schema: Jina, Cohere v2, llama.cpp, vLLM). The choice comes from `RERANKER_PROVIDER`. Reranking is the fourth strategy, `hybrid_rerank`: it reorders the best `RERANK_CANDIDATES` hybrid hits. sentence-transformers is an optional extra with CPU-only torch on Linux.

**Why:** the OpenAI API has no rerank endpoint. A small local model works offline on CPU (≈35 ms for 3 candidates after loading), and the API adapter avoids shipping torch where a rerank service already exists. Reranking only hybrid output keeps the comparison to the four strategies the experiment defines and measures exactly one change: hybrid vs hybrid + rerank.

| Alternative | Why not |
|-------------|---------|
| LLM as reranker (prompted scoring) | Slow and costly per candidate; mixes generation quality into a retrieval metric |
| Rerank every method | Four extra strategies to compare without a question asking for them |
| torch as a required dependency | ~1 GB for users who rerank through an API; hence the optional extra and `INSTALL_LOCAL_RERANKER` |

## ADR-010: Synthetic golden questions with human review

**Decision:** the LLM proposes questions from sampled passages; the source passage becomes the labelled span; a Streamlit screen lets a human edit, approve or reject them and add their own.

**Why:** the corpus is arbitrary, so no labels exist up front. Fully synthetic questions copy the passage wording and inflate lexical (BM25) scores. Human review keeps the benchmark honest. The generation prompt asks the model to paraphrase instead of copying the passage, which reduces (but does not remove) that lexical bias; the review is where the rest is caught.

## ADR-011: Explicit, resumable indexing

**Decision:** embedding is an explicit action per chunk set (`POST …/index`), not part of upload. Pending work is "chunks without a vector", embedded in batches with one commit per batch. The first run binds the collection to the embedding model and dimension; a different model returns 409 unless `rebuild=true`.

**Why:** uploads stay fast and work offline. Indexing cost and time can be measured per chunk set for the benchmark. A provider failure halfway through (rate limit, timeout) loses at most one batch, and the next call resumes without a job table.

| Alternative | Why not |
|-------------|---------|
| Embed during upload | Upload latency depends on a remote API; a failure would reject a perfectly valid document |
| Background job queue (Celery, RQ) | An extra service and a broker for an MVP; the resumable design gives most of the benefit |
| One transaction for the whole chunk set | A failure at 95% throws away all paid-for embeddings |
| No binding, store the model per vector only | Mixed-model collections would make similarity scores silently meaningless |

## ADR-012: One vector table without a fixed dimension

**Decision:** a single `chunk_embeddings` table with an untyped `vector` column, plus a `dimensions` column guarded by `CHECK (vector_dims(embedding) = dimensions)`.

**Why:** collections are bound to different models (768, 1024, 1536… dimensions), and a typed column `vector(n)` would force one model for the whole database. **Trade-off:** HNSW indexes require a fixed dimension. Retrieval does not need one (see ADR-014); if a large deployment ever does, it can be built per collection as a partial expression index (`(embedding::vector(n)) … WHERE collection_id = …`).

| Alternative | Why not |
|-------------|---------|
| `vector(1536)` column | Locks every collection to one model |
| One table per collection or per dimension | Dynamic DDL in the application, harder migrations and isolation reasoning |
| Vector column on `chunks` | A rebuild would rewrite the reference layer table; vectors are derived data and are kept apart |

## ADR-013: PostgreSQL full-text as the lexical baseline

**Decision:** the lexical method is PostgreSQL full-text search. Each chunk stores `to_tsvector(<collection language>, text)` (GIN index). A question becomes `plainto_tsquery(language, question)` with `&` rewritten to `|`, and matches are ranked with `ts_rank_cd(..., 1)` (cover density divided by `1 + log(length)`). The language is a per-collection setting (`fulltext_language`).

**Why:** it runs in the same database as the vectors (fair latency, one service), handles stemming and stop words for ~30 languages, and is a well-understood baseline. The OR rewrite matters: with AND, a natural-language question only matches chunks containing *every* content word, so lexical recall would collapse for reasons unrelated to the method.

**Trade-off:** `ts_rank_cd` is not BM25. It has no inverse document frequency, so a common term weighs as much as a rare one. The lab reports it as "full-text", not "BM25".

| Alternative | Why not |
|-------------|---------|
| BM25 extension (ParadeDB `pg_search`, VectorChord-bm25) | Not in the official pgvector image; a custom image and a native extension for an MVP |
| BM25 in Python (`rank_bm25`) | Loads every chunk into memory per query; latency would not be comparable with the vector method running in SQL |
| Elasticsearch / OpenSearch | A second engine; cross-engine latency distorts the comparison (ADR-005) |
| One global language | Mixed corpora are common; a Portuguese corpus with English stemming silently loses recall |

## ADR-014: Exact vector search for the benchmark

**Decision:** vector search is an exact scan (`ORDER BY embedding <=> :query`) over the vectors of one chunk set. No HNSW or IVFFlat index is created.

**Why:** the lab measures *retrieval strategies*. An approximate index adds a second, uncontrolled source of error: a low Recall@5 could come from the embeddings or from the ANN index skipping neighbours. Benchmark corpora are small (thousands of chunks), where an exact scan takes milliseconds. HNSW with filters (chunk set, documents) also loses results unless iterative scans are enabled (pgvector ≥ 0.8).

**Trade-off:** latency grows linearly with the chunk set. For a corpus of millions of chunks, add a per-collection HNSW partial expression index (ADR-012) and report its recall separately.

| Alternative | Why not |
|-------------|---------|
| HNSW per collection (as originally planned) | Dynamic DDL per collection and dimension; approximate results contaminate the comparison; filtered queries can return fewer than `top_k` |
| IVFFlat | Needs training data before creation and has the same approximation problem |

## ADR-015: Numbered citations checked by parsing, and a sentinel for abstention

**Decision:** retrieved chunks are given to the LLM as numbered sources. The answer must cite them as `[n]`; citations are extracted with a regular expression and are valid when at least one exists and all point to a provided source. When the sources are insufficient the model answers `NO_ANSWER …`. With no retrieved chunk the LLM is not called.

**Why:** it works with every OpenAI-compatible provider (no structured-output support needed), the citation metric is free and reproducible, and the two abstention paths separate retrieval failures from generation decisions, which is the diagnosis the lab exists for.

| Alternative | Why not |
|-------------|---------|
| JSON / structured output with source ids | Not supported by every compatible server; a parse failure would lose the whole answer |
| Quote matching (answer must quote the source) | Penalises correct paraphrases; brittle with PDF whitespace |
| LLM judge for every answer | Costs a second call per question; kept as an optional evaluation step, not the baseline metric |
| Score threshold for abstention | Scores are not calibrated across models and corpora |

## ADR-016: Benchmark runs as versioned JSON documents, plus a CLI

**Decision:** a run (config, environment, per-question results, summaries) is stored as one JSONB document with a `format_version`, encoded by a single `run_codec` shared by the database, the API and the CLI. The `rag-lab` CLI runs benchmarks and writes the Markdown report and JSON next to each other.

**Why:** runs are written once and always read whole; a document keeps them immutable and self-describing (models and settings in effect travel with the numbers). The CLI makes "a reproducible comparative report" a command, not a sequence of clicks.

| Alternative | Why not |
|-------------|---------|
| One table row per question × strategy | Joins and migrations for data that is never queried row by row |
| MLflow / Weights & Biases | Another service to run for what is a table and two files per run |
| Reports only in the UI | Not scriptable, not diffable, not committable |
