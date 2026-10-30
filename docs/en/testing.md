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

Everything above the adapter uses `tests/fakes.py::FakeEmbedder`: deterministic vectors, with switches to fail on the n-th call or return the wrong dimension.

## Why contract tests?

The in-memory store makes most tests fast. Fast tests are only trustworthy if the fake behaves like the real thing, so the contract suite is parametrised over both adapters. If PostgreSQL and memory ever disagree (ordering, rollback, isolation), the suite fails.

## Why property-based tests for chunking?

Example-based tests check the cases we thought of. Chunking bugs hide in the cases nobody thought of: whitespace runs, a chunk size of 1, overlap close to the size, accented characters. Hypothesis generates hundreds of such inputs and checks the four invariants (exact slices, size bound, no lost content, reading order) on every run.

## CI

`.github/workflows/ci.yml` runs on every push and PR: `uv sync --frozen` → ruff (lint + format check) → mypy strict → pytest with coverage, against a `pgvector/pgvector:pg17` service container. Contract tests therefore always run against real PostgreSQL in CI.
