# Domain layer — `rag_lab.domain`

The domain is the vocabulary of the lab: **spans, documents, chunks, collections and chunking strategies**. It is pure Python with no I/O and no frameworks. Every rule here can be tested in milliseconds, and it behaves the same whether it runs behind the API, a CLI or a benchmark runner.

> Other languages: [Português](../pt-BR/domain.md)

## Modules

| Module | Purpose | Key types |
|--------|---------|-----------|
| `spans.py` | Character interval over the canonical text | `Span` |
| `documents.py` | A parsed document and its structure | `Document`, `StructuralElement`, `ElementKind`, `MediaType` |
| `chunks.py` | A retrievable unit derived from a document | `Chunk` |
| `collections.py` | Isolation boundary, chunk sets to build, embedding binding | `Collection`, `EmbeddingMismatchError` |
| `embeddings.py` | Which model produced a vector and its length | `EmbeddingSpec` |
| `retrieval.py` | Search methods, scored hits and rank fusion | `SearchMethod`, `ScoredChunk`, `reciprocal_rank_fusion()` |
| `chunking/config.py` | How to chunk | `ChunkingConfig`, `ChunkingStrategy` |
| `chunking/strategies.py` | The chunking algorithms | `chunk_document()` |

---

## `Span`

**What it is:** a half-open interval `[start, end)` of character offsets. `start < end` is enforced, so empty spans cannot exist.

**What it does:** `length`, `overlap(other)` (number of shared characters, `0` when spans only touch) and `contains(offset)`.

**Why it exists:** chunks, citations and golden labels are all spans of the same text, so all of them can be compared with one operation, `overlap`. Phase 6 metrics (Recall@k, MRR) count a retrieved chunk as relevant when it overlaps a labelled span.

**Why half-open?** Adjacent spans `[0,10)` and `[10,20)` share no character and their lengths add up exactly. Python slicing (`text[start:end]`) uses the same convention, so off-by-one errors disappear.

## `Document` and `StructuralElement`

**What it is:** the **reference layer**. A document holds:

- `text`: the canonical text, normalised once (unified line endings, no NUL characters);
- `elements`: pages (PDF), sections (Markdown headings) or paragraphs (TXT), each a `Span` plus a label, level and heading `path`;
- `metadata`: title, author, page count, encoding… as strings;
- `content_hash`: SHA-256 of the original bytes, used for deduplication.

**Invariants:** the text is never empty, and every element lies inside the text.

**Helpers used by chunks and citations:**

| Method | Returns |
|--------|---------|
| `excerpt(span)` | The text slice |
| `section_path_for(span)` | Heading breadcrumb of the deepest section containing the span start, e.g. `("Install", "Docker")` |
| `page_numbers_for(span)` | Every 1-based page the span overlaps, e.g. `(3, 4)` for a chunk that crosses a page break |

**Sections cover only their own content.** A section runs up to the next heading of *any* level, and its `path` carries the parent chain. Elements therefore never overlap, while chunks still know their full heading context.

**Why not store chunks only?** Chunks change with every configuration. The canonical text does not. Storing structure as spans means a document is parsed once, and any number of chunk sets can be rebuilt from the database without the original file. See [ADR-002](decisions.md#adr-002-a-chunking-independent-reference-layer).

## `Chunk`

A span of a document plus denormalised location metadata (`page_numbers`, `section_path`) for citations. Build chunks with `Chunk.from_document(document, span, index)`, so that `chunk.text == document.text[span.start:span.end]` always holds.

## `Collection`

A named set of documents, plus the list of `ChunkingConfig`s every document is chunked with (one **chunk set** per config). It is the **isolation boundary**: queries are always scoped to one collection. Config keys must be unique, and `with_chunking_config()` returns a new collection (entities are immutable).

It also records its **full-text language** (`fulltext_language`, default `english`): the PostgreSQL text search configuration used to stem words and drop stop words. It must be one of `FULLTEXT_LANGUAGES` (the configurations PostgreSQL ships, plus `simple` for no stemming). It is chosen at creation and never changes, because every stored chunk was normalised with it.

## `EmbeddingSpec` and the embedding binding

**What it is:** `EmbeddingSpec(model, dimensions)`, a validated value object (non-empty model, positive dimension).

**What it does:** a collection starts **unbound** (`embedding is None`). The first indexing run calls `bind_embedding(spec)`:

| Collection state | `bind_embedding(spec)` |
|------------------|------------------------|
| Unbound | Binds to `spec` |
| Bound to the same spec | No-op |
| Bound to a different spec | `EmbeddingMismatchError` ("rebuild the index or create a new collection") |

`unbind_embedding()` is used by a rebuild.

**Why:** vectors from different models, or with different dimensions, live in different spaces. Comparing a `nomic-embed-text` query vector with `text-embedding-3-small` chunk vectors returns numbers, but meaningless ones, and nothing would fail loudly. The binding turns that silent corruption into an explicit error.

## `ChunkingConfig`

`strategy` + `chunk_size` + `chunk_overlap`, both **in characters**, validated (`size > 0`, `0 ≤ overlap < size`). Its stable `key` (for example `recursive-800-120`) names a chunk set in storage, in the API and in benchmark reports.

**Why characters and not tokens?** Token counts depend on the embedding model's tokenizer. Counting characters keeps a configuration identical across providers, so a benchmark comparing OpenAI and Ollama embeddings uses exactly the same chunks. See [ADR-003](decisions.md#adr-003-chunk-size-in-characters).

## Chunking strategies

`chunk_document(document, config)` dispatches to one of three strategies. All of them share these guarantees, checked by property-based tests (Hypothesis, 300 random cases per run):

1. a chunk is an **exact slice** of the canonical text;
2. no chunk is longer than `chunk_size`, and none is whitespace-only;
3. every non-whitespace character belongs to at least one chunk (**no content is lost**);
4. chunks come in reading order.

| Strategy | How it cuts | Strength | Weakness | Role in the benchmark |
|----------|-------------|----------|----------|-----------------------|
| `fixed` | Sliding window of `size` chars, step `size - overlap` | Simple, predictable | Cuts words and sentences | Baseline |
| `recursive` | Splits on `\n\n` → `\n` → `. ` → ` ` → characters, then greedily packs pieces up to `size`, carrying ≤ `overlap` chars of trailing pieces | Respects natural boundaries | Ignores document structure | General purpose |
| `structure` | Sections and pages are **hard** boundaries (never crossed); oversized units are split recursively; paragraphs are **soft** (small ones are merged) | Chunks align with headings and pages, so citations are precise | Uneven chunk sizes | Structure-aware |

**Why implement chunkers instead of using LangChain's splitters?** LangChain's text splitters return strings. Their optional `start_index` is reconstructed afterwards by searching for each chunk's text in the document, so offsets are *recovered*, not *tracked*. Exact offsets are the foundation of the reference layer, so the algorithms here are written to be offset-exact by construction (about 150 lines, fully tested). LangChain is used where it adds value: the LLM and embedding clients in later phases. See [ADR-004](decisions.md#adr-004-own-chunkers-langchain-at-the-provider-edge).

## Retrieval — `SearchMethod` and Reciprocal Rank Fusion

**What it is:** the three methods the lab compares (`vector`, `fulltext`, `hybrid`), the `ScoredChunk` a method returns (higher score is better) and `reciprocal_rank_fusion()`.

**What fusion does:** hybrid search gets one ranking per method and scores every chunk with

```
score(chunk) = Σ over rankings  1 / (k + rank of the chunk in that ranking)
```

A chunk ranked high by **both** methods beats a chunk ranked first by only one. Each fused hit keeps the rank every method gave it (`{"vector": 7, "fulltext": 1}`), which is how the UI explains *why* a chunk was retrieved. Ties keep the order of the first ranking, so results are deterministic.

**Why RRF and not a weighted sum of scores?**

| Option | Problem |
|--------|---------|
| `α·cosine + (1-α)·ts_rank` | The scores live on unrelated scales (cosine ∈ [-1, 1], `ts_rank_cd` unbounded and corpus-dependent); `α` would need tuning per corpus and per query |
| Min-max normalisation, then sum | Sensitive to outliers; one very long chunk shifts everyone's normalised score |
| **RRF (chosen)** | Uses ranks only, so no calibration; one parameter (`k`, default 60 as in the original paper, `HYBRID_RRF_K`) |

**Why in the domain?** Fusion is pure arithmetic over rankings. Keeping it out of SQL makes it identical for every storage adapter and testable in milliseconds.
