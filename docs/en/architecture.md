# Architecture

RAG Benchmark Lab uses a **hexagonal architecture** (ports and adapters). Business rules live in a pure core. Frameworks, databases and model providers plug in from the outside. Retrieval strategies, LLM providers and storage can therefore be swapped and compared without touching the rules being measured.

> Other languages: [Português](../pt-BR/architecture.md)

## The layers at a glance

```mermaid
flowchart LR
    subgraph Inbound["Inbound adapters"]
        API["FastAPI<br/>adapters/inbound/api"]
        UI["Streamlit UI<br/>ui/ (HTTP client)"]
    end
    subgraph Core["Core"]
        APP["Application<br/>use cases + ports"]
        DOM["Domain<br/>reference layer, chunking"]
    end
    subgraph Outbound["Outbound adapters"]
        PAR["Parsers<br/>PDF / Markdown / TXT"]
        PG["PostgreSQL + pgvector<br/>SQLAlchemy Core + Alembic"]
        MEM["In-memory store"]
        EMB["Embeddings provider<br/>OpenAI-compatible API (LangChain)"]
    end
    UI -- HTTP --> API
    API --> APP
    APP --> DOM
    APP -. port .-> PAR
    APP -. port .-> PG
    APP -. port .-> MEM
    APP -. port .-> EMB
```

| Layer | Package | Purpose | May import |
|-------|---------|---------|------------|
| Domain | `rag_lab.domain` | Business concepts and pure algorithms (spans, documents, chunking) | Standard library only |
| Application | `rag_lab.application` | Use cases, and the ports (interfaces) they need | Domain |
| Adapters | `rag_lab.adapters` | Implement ports (outbound) or call use cases (inbound) | Application, domain, frameworks |
| Composition root | `rag_lab.bootstrap` | Decides which adapter implements which port | Everything |
| UI | `rag_lab.ui` | Streamlit pages; talks to the system **only through the HTTP API** | `config`, its own modules |

**The dependency rule:** arrows only point inwards. The domain never imports FastAPI, SQLAlchemy, pypdf or LangChain. The type checker (`mypy --strict`) and the test suite run on every commit, so a violation shows up immediately.

## Data flows

### Ingestion (implemented)

```mermaid
sequenceDiagram
    participant C as Client (UI / curl)
    participant A as API route
    participant S as IngestionService
    participant P as ParserRegistry
    participant D as Domain (chunking)
    participant U as UnitOfWork
    C->>A: POST /collections/{id}/documents (file)
    A->>S: ingest_document(collection_id, filename, bytes)
    S->>S: size guard, sha256 hash
    S->>U: duplicate? (same hash in collection)
    S->>P: parse(filename, bytes)
    P-->>S: canonical text + pages/sections/paragraphs + metadata
    S->>D: chunk_document(document, config) for every chunk set
    S->>U: add document + chunks, commit (one transaction)
    A-->>C: 201 document summary + chunk counts per set
```

### Indexing (implemented)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as IndexingService
    participant U as UnitOfWork
    participant E as Embedder (OpenAI-compatible)
    C->>S: index_chunk_set(collection, chunk set, rebuild?)
    S->>U: bind collection to model + dimension (or 409 if bound to another), commit
    loop until no pending chunks
        S->>U: next batch of chunks without a vector
        S->>E: embed_documents(texts)  (no transaction open)
        S->>S: check count and dimension of vectors
        S->>U: save vectors, commit
    end
    S-->>C: report (total, already indexed, embedded now, seconds)
```

Each batch is committed on its own, so a provider failure halfway loses nothing: the next run embeds only what is still pending.

### Retrieval (implemented)

```mermaid
sequenceDiagram
    participant C as Client
    participant R as RetrievalService
    participant E as Embedder
    participant DB as SearchRepository
    participant RR as Reranker
    C->>R: search(collection, chunk set, query, method)
    R->>DB: check chunk set (+ binding and coverage for vector/hybrid)
    alt vector or hybrid
        R->>E: embed_query(query)
        R->>DB: vector(top candidates)
    end
    alt fulltext or hybrid
        R->>DB: fulltext(top candidates)
    end
    R->>R: reciprocal_rank_fusion (hybrid)
    opt hybrid_rerank
        R->>RR: score(query, top RERANK_CANDIDATES texts)
        R->>R: reorder by reranker score
    end
    R-->>C: hits + rank per method + timings
```

### Question answering (implemented)

```mermaid
flowchart LR
    Q[Question] --> R[RetrievalService<br/>method + filters + rerank]
    R -->|no hits| A1[Abstain: no_evidence_retrieved]
    R -->|hits| P[Numbered sources<br/>prompts.py]
    P --> L[ChatModel]
    L -->|NO_ANSWER| A2[Abstain: model_abstained]
    L -->|text with n citations| C[check_citations]
    C --> ANS[Answer + sources + citation validity + timings]
```

### Evaluation (implemented)

```mermaid
flowchart LR
    D[Documents] -->|seeded passages| G[LLM drafts]
    M[Manual question + excerpt] --> R
    G --> R[Human review<br/>approve / reject / edit]
    R -->|approved| B[BenchmarkService]
    B -->|chunk sets x methods| S[Recall@k, MRR, latency, CPU]
    B -->|optional| A[Answers: citations, abstention, judge]
    S --> P[Run stored + Markdown/JSON report]
    A --> P
```

## The reference layer: why chunking can vary freely

This is the most important design decision in the project. See [ADR-002](decisions.md#adr-002-a-chunking-independent-reference-layer).

- Each document is parsed **once** into a **canonical text**.
- Pages, sections and paragraphs are stored as **spans** (`[start, end)` character offsets) over that text.
- A **chunk** is just a span plus derived metadata (page numbers, heading path).
- Golden-dataset labels (phase 6) will also be spans over the same text.

```text
canonical text:  |# Install\nRun it.\n\n## Docker\ndocker compose up\n|
sections:        [0 ─────────── 19)[19 ───────────────────────── 47)
fixed-20-0:      [0 ──── 20)[20 ──── 40)[40 ─ 47)
structure-800-0: [0 ─────── 17)  [19 ────────────────────── 46)
golden label:                          [29 ──────────── 46)
```

Relevance is measured by **overlap between spans**, not by matching chunk ids. A label written once stays valid for every chunking configuration, and that is what makes a chunking × retrieval benchmark matrix possible.

## Isolation

The **collection** is the isolation boundary. Every repository method that returns documents or chunks takes a `collection_id`, so the port's signature itself enforces the scope. The contract tests in `tests/contracts/test_isolation.py` run against both the in-memory and the PostgreSQL adapters. They prove that identical documents in two collections stay separate, and that a document id from another collection returns *not found*.

## Where to go next

| I want to understand… | Read |
|-----------------------|------|
| Spans, documents, chunking algorithms | [domain.md](domain.md) |
| Use cases, ports, transactions | [application.md](application.md) |
| Parsers, PostgreSQL, API, UI | [adapters.md](adapters.md) |
| `.env` settings and providers | [configuration.md](configuration.md) |
| Test strategy and how to run tests | [testing.md](testing.md) |
| Why each technology was chosen | [decisions.md](decisions.md) |
