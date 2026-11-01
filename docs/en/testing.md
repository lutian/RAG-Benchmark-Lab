# Testing strategy

The project is built **test-first**: every rule gets a failing test (RED), then the code that passes it (GREEN), then a refactor while the tests stay green.

> Other languages: [Português](../pt-BR/testing.md)

## Quick path

```bash
uv sync                      # install dev dependencies
uv run pytest                # unit + contract tests (PostgreSQL ones skip without a DB)
make lint typecheck          # ruff + mypy --strict
```

Run the PostgreSQL contract tests with a disposable database:

```bash
docker compose up -d db
TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_lab uv run pytest -m integration
```

> The tests **truncate every table** in that database. Never point `TEST_DATABASE_URL` at data you care about.

## Test pyramid

| Level | Location | What it proves | Speed |
|-------|----------|----------------|-------|
| Domain unit | `tests/unit/domain/` | Span maths, document invariants, chunking algorithms | ms |
| Property-based | `tests/unit/domain/test_chunkers.py` | Chunking invariants for **any** text and config (Hypothesis) | < 1 s |
| Adapter unit | `tests/unit/adapters/` | Parsers on real PDF bytes (generated with fpdf2), API contract via `TestClient` | ms |
| Application | `tests/unit/application/` | Use-case rules (dedup, size limit, chunk set backfill, atomicity) | ms |
| Contract | `tests/contracts/` | **Same tests against in-memory and PostgreSQL**: isolation, transactions, round-trip of the reference layer | s |
| UI | `tests/unit/ui/` | Pages render with a faked API (`streamlit.testing.AppTest`); highlight rendering | ms |

## Testing the embeddings adapter

`tests/unit/adapters/embeddings/` starts a real HTTP server on `127.0.0.1` that speaks the `/v1/embeddings` protocol and records every request. The tests assert what actually goes over the wire: raw text (not token ids), batch boundaries, the model name, the `Authorization` header, no `dimensions` field. They also cover how 500 responses and unreachable hosts are reported. Mocking the SDK would not catch a LangChain default that changes the payload.

Everything above the adapter uses `tests/fakes.py::FakeEmbedder`: deterministic vectors, with switches to fail on the n-th call or return the wrong dimension. Retrieval tests use `ConceptEmbedder`, which maps synonyms to the same dimension, so a test can show vector search finding a paraphrase ("car" → "automobile") that full-text search misses.

## Testing the rerankers

`tests/unit/adapters/reranking/` runs `HttpReranker` against a real local `/rerank` stub (request body, auth header, scores returned out of order, TEI-style lists, HTTP errors, missing indexes, unreachable host) and `CrossEncoderReranker` with an injected fake model (lazy loading, one load, missing library). Tests never download a model; `tests/fakes.py::FakeReranker` stands in above the adapter.

## Testing the chat model

`tests/unit/adapters/generation/` runs `OpenAICompatibleChatModel` against a local `/v1/chat/completions` stub: messages and model on the wire, `temperature` / `reasoning_effort` sent only when configured, usage present or absent, HTTP errors and unreachable hosts. Above the adapter, `tests/fakes.py::FakeChatModel` returns scripted replies and records prompts.

## Testing the evaluation

Metric functions are tested on hand-computed rankings (`tests/unit/domain/test_evaluation.py`). `BenchmarkService` runs end to end in memory with `ConceptEmbedder`, `FakeReranker` and a `FakeChatModel` that answers and judges, asserting the matrix shape, the warm-up (one extra reranker call per method), and generation/judge rates. The CLI is tested with an in-memory container and a temporary output directory. Golden storage has contract tests on both adapters, and `tests/integration/test_golden_constraints.py` proves on PostgreSQL that a labelled document cannot be deleted alone while its collection can.

## Testing the presentation layer

`tests/unit/ui/test_dashboard_data.py` checks the frames behind every dashboard chart; `test_benchmark_page.py` renders the page with a fake API (tiles against targets, starting a run). The demo command runs the real `examples/` corpus and golden file through an in-memory container, twice, to prove it is idempotent. `examples/build_golden.py` fails if any evidence excerpt is not found in the parsed text, so the shipped dataset always matches the corpus.

## PostgreSQL-only behaviour

`tests/integration/test_postgres_fulltext.py` covers what the in-memory store deliberately does not imitate: stemming and stop words per collection language (`rates` matches `rate` in `english` but not in `simple`; `banco` matches `bancos` in `portuguese`). `tests/integration/test_migrations.py` downgrades to 0002, inserts a chunk, upgrades again and proves migration 0003 made it searchable.

## Why contract tests?

The in-memory store makes most tests fast. Fast tests are only trustworthy if the fake behaves like the real thing, so the contract suite is parametrised over both adapters. If PostgreSQL and memory ever disagree (ordering, rollback, isolation), the suite fails.

## Why property-based tests for chunking?

Example-based tests check the cases we thought of. Chunking bugs hide in the cases nobody thought of: whitespace runs, a chunk size of 1, overlap close to the size, accented characters. Hypothesis generates hundreds of such inputs and checks the four invariants (exact slices, size bound, no lost content, reading order) on every run.

## CI

`.github/workflows/ci.yml` runs on every push and PR: `uv sync --frozen` → ruff (lint + format check) → mypy strict → pytest with coverage, against a `pgvector/pgvector:pg17` service container. Contract tests therefore always run against real PostgreSQL in CI.

`.github/workflows/docker.yml` checks the delivery path the tests cannot see: it builds the image with the default settings (local cross-encoder included), starts `db`, `api` and `ui` with `docker compose up --wait`, checks that the reranker is installed and that the UI container reaches the API over the Compose network, then runs `scripts/compose_smoke.py`. The smoke test creates a collection (proving the migrations ran against `db`), uploads a sample document, expects full-text search to rank the "Final Sale" chunk first, and checks the Streamlit health endpoint. It uses full-text search on purpose: it needs no provider key, so the job runs on every PR without secrets. Run the same check locally with `make compose-smoke` (needs a `.env`): it uses a separate Compose project and removes it with its volumes afterwards, so the smoke collection never lands in your real database. It publishes the same ports, so stop the main stack first. The workflow keeps its own copy of the `up` command because it runs extra checks between start-up and teardown and prints the container logs on failure.
