# Benchmark report — sample-handbook

Run `b7a9ed11-0010-4537-8a2d-59fad836ae7e` · 2026-11-01 17:46 UTC · 52 approved questions (target 50)

**Best strategy:** `fixed-800-120` + `hybrid_rerank` — Recall@5 100.0%, MRR@10 0.978, p50 1854.9 ms.

> Reference run of `rag-lab demo` on the sample corpus (retrieval only). Embeddings:
> `sentence-transformers/all-MiniLM-L6-v2` served through a local OpenAI-compatible
> endpoint; reranker: the local cross-encoder on a single-vCPU VPS without GPU. Numbers change
> with the embedding model and hardware: run `make demo` to produce your own.

## Setup

| Setting | Value |
|---------|-------|
| embedding | `all-MiniLM-L6-v2 (384d)` |
| hybrid | `RRF k=60, 50 candidates` |
| llm | `gpt-4o-mini` |
| rerank_candidates | `30` |
| reranker | `local: cross-encoder/ms-marco-MiniLM-L-6-v2` |
| chunk sets | `fixed-800-120`, `recursive-800-120`, `structure-800-120` |
| methods | `vector`, `fulltext`, `hybrid`, `hybrid_rerank` |
| generation / judge | False / False |

## Retrieval

Targets: Recall@5 ≥ 85%, MRR@10 ≥ 0.75. A chunk is relevant when it overlaps a labelled evidence span.

| Chunk set | Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 ms | p95 ms | CPU ms |
|---|---|---|---|---|---|---|---|---|
| `fixed-800-120` | `vector` | 71.2% | 98.1% ✅ | 100.0% | 0.816 ✅ | 24.0 | 31.5 | 6.6 |
| `fixed-800-120` | `fulltext` | 67.3% | 96.2% ✅ | 98.1% | 0.803 ✅ | 2.4 | 3.6 | 1.6 |
| `fixed-800-120` | `hybrid` | 71.2% | 100.0% ✅ | 100.0% | 0.838 ✅ | 26.7 | 36.0 | 8.0 |
| `fixed-800-120` | `hybrid_rerank` | 96.2% | 100.0% ✅ | 100.0% | 0.978 ✅ | 1854.9 | 2045.6 | 1665.1 |
| `recursive-800-120` | `vector` | 71.2% | 100.0% ✅ | 100.0% | 0.837 ✅ | 23.1 | 31.9 | 6.4 |
| `recursive-800-120` | `fulltext` | 73.1% | 94.2% ✅ | 96.2% | 0.833 ✅ | 2.6 | 5.1 | 1.8 |
| `recursive-800-120` | `hybrid` | 75.0% | 98.1% ✅ | 100.0% | 0.861 ✅ | 25.9 | 36.6 | 7.7 |
| `recursive-800-120` | `hybrid_rerank` | 94.2% | 100.0% ✅ | 100.0% | 0.968 ✅ | 1743.2 | 1872.3 | 1561.3 |
| `structure-800-120` | `vector` | 75.0% | 98.1% ✅ | 98.1% | 0.854 ✅ | 25.0 | 36.6 | 6.8 |
| `structure-800-120` | `fulltext` | 65.4% | 92.3% ✅ | 94.2% | 0.782 ✅ | 2.6 | 3.9 | 1.7 |
| `structure-800-120` | `hybrid` | 76.9% | 96.2% ✅ | 98.1% | 0.865 ✅ | 27.7 | 37.8 | 8.5 |
| `structure-800-120` | `hybrid_rerank` | 96.2% | 100.0% ✅ | 100.0% | 0.973 ✅ | 1860.8 | 2200.9 | 1605.0 |

## Head to head

Per question, which method ranked the evidence higher (by reciprocal rank).

| Chunk set | Question | Wins | Losses | Ties |
|---|---|---|---|---|
| `fixed-800-120` | Does lexical beat vectors? (`fulltext` vs `vector`) | 12 | 12 | 28 |
| `fixed-800-120` | Does fusion help vectors? (`hybrid` vs `vector`) | 8 | 3 | 41 |
| `fixed-800-120` | Does reranking help hybrid? (`hybrid_rerank` vs `hybrid`) | 14 | 0 | 38 |
| `recursive-800-120` | Does lexical beat vectors? (`fulltext` vs `vector`) | 8 | 8 | 36 |
| `recursive-800-120` | Does fusion help vectors? (`hybrid` vs `vector`) | 6 | 3 | 43 |
| `recursive-800-120` | Does reranking help hybrid? (`hybrid_rerank` vs `hybrid`) | 11 | 2 | 39 |
| `structure-800-120` | Does lexical beat vectors? (`fulltext` vs `vector`) | 6 | 11 | 35 |
| `structure-800-120` | Does fusion help vectors? (`hybrid` vs `vector`) | 5 | 3 | 44 |
| `structure-800-120` | Does reranking help hybrid? (`hybrid_rerank` vs `hybrid`) | 11 | 2 | 39 |
