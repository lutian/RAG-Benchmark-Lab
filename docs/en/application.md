# Application layer — `rag_lab.application`

The application layer holds the **use cases** (what the system can do) and the **ports** (what it needs from the outside world). It orchestrates domain objects and never knows whether the parser is pypdf or the database is PostgreSQL.

> Other languages: [Português](../pt-BR/application.md)

## Modules

| Module | Purpose |
|--------|---------|
| `services/ingestion.py` | `IngestionService`: collections, document ingestion, chunk sets |
| `ports/parsing.py` | `DocumentParser` port and its output `ParsedDocument` |
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

## Ports

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
| `UnitOfWork` | Context manager exposing the three repositories; `commit()` makes changes visible, leaving the block without committing rolls back |

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
| `ConflictError` | Duplicate collection name or chunk set | 409 |
| `UploadTooLargeError` | File exceeds `MAX_UPLOAD_MB` | 413 |
| `UnsupportedDocumentError` | Extension is not `.pdf`, `.md`, `.markdown`, `.txt` | 415 |
| `DocumentParsingError` | Supported type but unreadable (corrupt, empty, scanned PDF without text) | 422 |

A document from another collection returns **404 and not 403**, so the API does not even reveal that the id exists elsewhere.
