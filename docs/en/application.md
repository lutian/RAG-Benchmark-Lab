# Application layer — `rag_lab.application`

The application layer holds the **use cases** (what the system can do) and the **ports** (what it needs from the outside world). It orchestrates domain objects and never knows whether the parser is pypdf or the database is PostgreSQL.

> Other languages: [Português](../pt-BR/application.md)

## Modules

| Module | Purpose |
|--------|---------|
| `services/ingestion.py` | `IngestionService`: collections, document ingestion, chunk sets |
| `services/indexing.py` | `IndexingService`: embed chunk sets, binding, progress |
| `services/answering.py` | `AnswerService`: retrieval → numbered context → LLM → checked citations or abstention |
| `prompts.py` | The grounded-answer prompt (rules, numbered sources) |
| `services/golden.py` | `GoldenDatasetService`: synthetic and manual questions, review, export/import |
| `services/benchmark.py` | `BenchmarkService`: the chunking × strategy matrix on approved questions |
| `reporting.py` / `run_codec.py` | Markdown report of a run; versioned JSON encoding of runs |
| `services/retrieval.py` | `RetrievalService`: vector, full-text, hybrid and reranked hybrid search with filters and timings |
| `ports/parsing.py` | `DocumentParser` port and its output `ParsedDocument` |
| `ports/embeddings.py` | `Embedder` (the model) and `EmbeddingRepository` (the vector store) ports |
| `ports/generation.py` | `ChatModel` port, `ChatMessage`, `Completion` |
| `ports/reranking.py` | `Reranker`: scores (question, text) pairs |
| `ports/search.py` | `SearchRepository`: ranked vector and full-text lookups over one chunk set |
| `ports/repositories.py` | Repository ports and the `UnitOfWork` transaction boundary |
| `errors.py` | Expected failures, translated to HTTP codes by the API |

## `IngestionService`

| Use case | Behaviour |
|----------|-----------|
| `create_collection(name, configs)` | Trims and validates the name; a duplicate name raises `ConflictError` |
| `add_chunking_config(id, config)` | Registers a new chunk set **and chunks every existing document** with it, in one transaction |
| `ingest_document(id, filename, bytes)` | Size guard → SHA-256 → duplicate check → parse → build `Document` → chunk with every config → persist atomically |
| `list_documents`, `get_document`, `list_chunks` | Reads, always scoped by `collection_id` |

**Idempotent ingestion:** uploading the same bytes twice into a collection returns the existing document with `duplicate=True`. Re-uploads never inflate the corpus, which would skew benchmark results.

**Atomicity:** the document, its structure and all its chunk sets are written in a single unit of work. A failure (for example a corrupt PDF) leaves nothing behind. This is verified for both adapters by `tests/contracts/test_transactions.py`.

**Why a service class and not one class per use case?** The ingestion use cases share their dependencies (unit of work, parser, limits) and are small. One cohesive service keeps the composition root simple. When a use case grows (retrieval, evaluation), it gets its own service.

## `IndexingService`

| Use case | Behaviour |
|----------|-----------|
| `index_chunk_set(id, key, rebuild=False)` | Binds the collection to the embedder's spec, then embeds pending chunks batch by batch (`EMBEDDING_BATCH_SIZE`) and commits each batch. Returns an `IndexingReport` (total, already indexed, embedded now, seconds) |
| `index_chunk_set(..., rebuild=True)` | Deletes every vector of the collection, unbinds it, then indexes from scratch with the current model |
| `status(id)` | `IndexStatus`: the binding plus `embedded / total` for every chunk set |

Design points:

- **The provider is called outside any transaction.** A slow embeddings API never holds a database connection or locks.
- **Resumable by construction.** "Pending" means "chunk without a vector", so re-running after a failure continues where it stopped. No job table is needed.
- **Vectors are validated before saving.** The count must match the batch and every vector must have the bound dimension. Otherwise `EmbeddingProviderError` explains which setting to check (`EMBEDDING_DIMENSIONS`).
- **Explicit action, not on upload.** Uploading stays fast and works without a provider. Indexing can be repeated per chunk set and timed for the benchmark. See [ADR-011](decisions.md#adr-011-explicit-resumable-indexing).
- **Bounded loop.** The batch loop runs at most `ceil(pending / batch) + 1` times, so a faulty store can never spin forever.

## `RetrievalService`

| Use case | Behaviour |
|----------|-----------|
| `search(id, key, query, method, top_k=10, document_ids=None)` | Ranks the chunks of one chunk set and returns a `SearchResult`: hits (chunk, score, rank, rank per method) and timings (`embed_ms`, `search_ms`, `rerank_ms`, `total_ms`) |

Flow: validate (non-blank query, `1 ≤ top_k ≤ 100`) → check the chunk set exists → for `vector`/`hybrid`, check the embedding binding and coverage → embed the query → search → fuse (hybrid) → rerank (`hybrid_rerank`).

Design points:

- **Vector scores are only meaningful against a complete index of the same model.** `vector` and `hybrid` return 409 when the collection is not indexed, is bound to another model than the configured embedder, or has chunks without vectors (`3/4 chunks`). A silently partial index would bias every benchmark number. `fulltext` works right after upload.
- **Hybrid digs deeper than `top_k`.** Each method returns `RETRIEVAL_CANDIDATES` (default 50) chunks before fusion, because a chunk ranked 12th by both methods can deserve a top-5 fused position.
- **Timings per stage.** Embedding the query (a network call) and searching (the database) are measured separately, so the benchmark can show where latency comes from. Full-text never calls the embedder (`embed_ms = 0`).
- **Filters.** `document_ids` restricts every method to those documents (the "Filters" step of the question flow).
- **Reranking (`hybrid_rerank`).** The best `RERANK_CANDIDATES` (default 30) hybrid hits go to the `Reranker`, which scores each (question, chunk) pair; hits are reordered by that score (ties keep the hybrid order) and cut to `top_k`. Each hit keeps `ranks["hybrid"]`, its position before reranking, so the UI shows moves like "hybrid #7 → #1". Reranking time is reported as `rerank_ms`. A reranker that returns a different number of scores is a `RerankerError`.
- **Query vectors are validated** against the embedder's dimension; a wrong length is an `EmbeddingProviderError` (502), never a database error.
- **Explainable hits.** Every hit carries `ranks`, e.g. `{"vector": 9, "fulltext": 1}` for a chunk that only the lexical method found near the top.

## `AnswerService`

| Use case | Behaviour |
|----------|-----------|
| `answer(id, key, question, method=hybrid, top_k=5, document_ids=None)` | Retrieves with `RetrievalService`, numbers the hits as sources, asks the `ChatModel`, then parses and checks the citations. Returns an `Answer`: text, `abstained` + `abstention_reason`, sources, valid citations (number → hit), `CitationCheck`, model, token usage and timings (retrieval stages + `generation_ms`) |

Design points:

- **Two abstention reasons, two culprits.** `no_evidence_retrieved`: retrieval returned nothing, so the LLM is **not called** (no cost, no chance to hallucinate). `model_abstained`: sources existed but the model said `NO_ANSWER`. When diagnosing a bad answer, the first points at retrieval, the second at retrieval *quality* or generation.
- **Every source is returned**, cited or not, so you can see whether the right chunk was retrieved but ignored (generation problem) or never retrieved (retrieval problem).
- **The prompt lives in `application/prompts.py`**: one versioned place, unit-tested for its rules (numbered sources, sentinel, answer in the question's language). Sources are labelled with page and section so the model can tell them apart.
- **No score threshold.** Vector and RRF scores are not calibrated across corpora and models, so a fixed cut-off would abstain arbitrarily. The model decides from the content; the benchmark measures how often it is right to.

## `GoldenDatasetService`

| Use case | Behaviour |
|----------|-----------|
| `generate(id, count, seed)` | Cuts every document into ~400-character passages with the recursive chunker (independent of the collection's chunk sets), drops passages under 100 characters or already used as evidence, samples them with `random.Random(seed)` round-robin across documents, and asks the LLM for one question per passage (`SKIP` = no factual content). Each question is a **draft**, committed as soon as it exists |
| `add_manual(id, text, document_id, excerpt)` | Locates the pasted excerpt in the canonical text, ignoring whitespace differences (PDF line breaks); 422 when it is not there |
| `questions` / `views` / `update` / `delete` | List (optionally by status), list with excerpts and filenames for review, edit text or status, delete |
| `export(id)` / `import_(id, payload)` | Portable JSON whose evidence carries the document **content hash**, span and excerpt; import maps hashes to the target collection's documents and skips questions whose files are missing or whose excerpt is found neither at the exported offsets nor elsewhere in the document (stale offsets are never kept); a malformed entry rejects the payload with 422 |

**Why seeded passages from the reference layer?** The same seed and documents give the same passages, so a dataset can be regenerated; using the reference text (not a chunk set) keeps the labels independent of the chunking under test.

## `BenchmarkService`

Runs `BenchmarkConfig` (chunk sets × methods, `ks`, MRR cut-off, optional generation and judge) on the **approved** questions:

1. Validate: chunk sets exist (404) and approved questions exist (422).
2. **Warm up** each chunk set × method once, unmeasured. The first reranked query loads the cross-encoder; without warm-up that cost would land on one random question. Warm-up also fails fast (e.g. 409 for an unindexed chunk set) before anything is measured.
3. For every chunk set × method × question: retrieve `max(ks, mrr_cutoff)` chunks, timing wall clock and CPU, then score with `score_ranking`.
4. With `generation`: answer with `answer_top_k` sources (abstention, citation validity, tokens); with `judge`: one more LLM call returns `SUPPORTED` / `UNSUPPORTED` (`parse_verdict`; anything else counts as unknown).
5. `summarize` per strategy and store the run.

## Reports — `reporting.py` and `run_codec.py`

`markdown_report(run, collection_name)` renders: setup (models and settings in effect), the retrieval matrix with ✅/❌ against the project targets, head-to-head duels (lexical vs vector, hybrid vs vector, rerank vs hybrid), generation quality and the retrieval-vs-generation diagnosis. `best_strategy` (highest Recall@5, then MRR, then lowest p50) is shared by the report, API and CLI. `run_codec` is the single, versioned JSON format of a run (stored in PostgreSQL, served by the API, written by the CLI).

## Ports

### `ChatModel`

```python
name: str
def complete(self, messages: Sequence[ChatMessage]) -> Completion  # text, model, prompt/completion tokens
```

Implementations raise `GenerationProviderError`. Token counts are `None` when the provider does not report usage.

### `Reranker`

```python
name: str
def score(self, query: str, texts: Sequence[str]) -> list[float]  # one score per text, input order
```

A cross-encoder reads the question and the chunk **together**, so it judges relevance better than comparing two independently computed embeddings, at the cost of one model pass per candidate. That is why it only reorders a short candidate list. Implementations raise `RerankerError`.


### `DocumentParser`

```python
def parse(self, filename: str, data: bytes) -> ParsedDocument
```

It returns canonical text, structural elements and metadata, or raises `UnsupportedDocumentError` / `DocumentParsingError`. The service assigns identity (id, hash, collection) afterwards, so parsers stay pure transformations.

### Repositories and `UnitOfWork`

| Port | Responsibility |
|------|----------------|
| `CollectionRepository` | Add, update, get by id or name, list |
| `DocumentRepository` | Add, get **(collection_id, document_id)**, list by collection, find by hash |
| `ChunkRepository` | Replace one document's chunks in one chunk set; list a chunk set (optionally one document) |
| `EmbeddingRepository` | Pending chunks of a chunk set, save vectors, count, delete all vectors of a collection |
| `SearchRepository` | `vector(...)`: exact nearest neighbours by cosine similarity; `fulltext(...)`: chunks sharing at least one normalised term with the query, ranked by coverage. Both scoped by collection and chunk set, with an optional document filter |
| `GoldenQuestionRepository` | Add (evidence must belong to the collection), update text/status only, get, list by status, delete |
| `BenchmarkRunRepository` | Add, get, list newest first |
| `UnitOfWork` | Context manager exposing the repositories; `commit()` makes changes visible, leaving the block without committing rolls back |

### `Embedder`

```python
spec: EmbeddingSpec
def embed_documents(self, texts: Sequence[str]) -> list[list[float]]
def embed_query(self, text: str) -> list[float]
```

Implementations wrap provider failures in `EmbeddingProviderError`. `embed_query` is used by retrieval for the question.

**Isolation is in the signatures.** No read method returns documents or chunks without a `collection_id`. A caller cannot "forget the filter", because there is no unfiltered method to call.

**Why Unit of Work + Repository instead of passing a database session around?**

| Option | Problem it has here |
|--------|---------------------|
| SQLAlchemy `Session` in services | Couples use cases to SQLAlchemy; the in-memory adapter becomes impossible |
| Repositories that commit on every call | A document could be saved without its chunks, breaking atomicity |
| **Unit of Work (chosen)** | One explicit transaction boundary that works the same for PostgreSQL and memory |

## Errors

| Error | Meaning | HTTP |
|-------|---------|------|
| `NotFoundError` | Entity does not exist **or belongs to another collection** | 404 |
| `ConflictError` | Duplicate collection name or chunk set; collection bound to another embedding model; vector/hybrid search on a chunk set that is not fully indexed | 409 |
| `InvalidRequestError` | Values that cannot be used: `hybrid_rerank` without a reranker, blank query, `top_k` out of range, unknown full-text language, blank collection name | 422 |
| `EmbeddingProviderError` | Embeddings API failed, was unreachable, or returned vectors of the wrong size | 502 |
| `GenerationProviderError` | Chat LLM failed, timed out or was unreachable (the upstream body is not echoed) | 502 |
| `RerankerError` | Reranker could not load (missing `local-rerank` extra), its API failed, or it returned malformed scores | 502 |
| `UploadTooLargeError` | File exceeds `MAX_UPLOAD_MB` | 413 |
| `UnsupportedDocumentError` | Extension is not `.pdf`, `.md`, `.markdown`, `.txt` | 415 |
| `DocumentParsingError` | Supported type but unreadable (corrupt, empty, scanned PDF without text) | 422 |

A document from another collection returns **404 and not 403**, so the API does not even reveal that the id exists elsewhere.
