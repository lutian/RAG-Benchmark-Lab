# Phase 7 — Presentation (dashboard, sample corpus, demo, technical guide)

Objective: a comparative metrics dashboard, a sample corpus with a reviewed 50-question golden dataset, one command that runs the whole experiment end to end, and the technical guide that explains the results.
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "7. Apresentação    Dashboard e documentação técnica". MVP item "10. Dashboard comparativo de métricas."
- **S2 — Completion criterion.** "o repositório executa localmente, indexa um corpus de exemplo, responde a perguntas com fontes e gera um relatório comparativo reproduzível."
- **S3 — What to demonstrate.** "por que o chunking afeta a recuperação, quando BM25 supera embeddings, quando o reranking ajuda e como diagnosticar se o problema está na recuperação ou na geração."
- **S4 — Targets.** "Perguntas de avaliação 50", "Estratégias comparadas 4", "Recall@5 ≥ 85%", "MRR@10 ≥ 0,75", "Respostas com citações válidas ≥ 95%" — shown against results.
- **S5 — Design.** Streamlit + custom CSS following the reference design (agreed in Phase 1).
- **S6 — Quality bar.** TDD, docs in EN and PT with purpose, behavior and why.

Assumptions (not user-stated): the sample corpus is original fictional content written for this repo; its golden dataset ships as an export file and is imported as approved; `rag-lab demo` creates the collection, ingests, indexes, imports and benchmarks.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S1 S3 S4 S5 | inline | done | 371a947, 434ba8c |
| T2 | S2 S4 | inline | done | a15bae4 |
| T3 | S2 | inline | done | a15bae4, c5e066c, 434ba8c |
| T4 | S3 S6 | inline | done | c672be4 |

- T1 Streamlit "Benchmark" dashboard: run a benchmark, KPI tiles vs targets, Recall@5 matrix, latency chart, head-to-head, per-question drill-down, report download.
- T2 Sample corpus (`examples/corpus/`) and 50-question golden dataset (`examples/golden.json`).
- T3 `rag-lab demo` (+ `make demo`, image includes `examples/`).
- T4 Technical guide EN/PT (running and reading the experiment), README completion.

Route evidence: sequential, parent writes inline. Risk: item 5 for the Dockerfile `COPY examples` (build change).

## Log

- L1 (2026-10-31, user): "sim, faz o push. E continua até entregar todos os passos. Só para se precisar de minha definição"
- L2 (2026-11-01, user): "sempre faça a revisão com gentle-ai, não pergunte mais para esse projeto"
- L3 Verification: 256 unit tests + contracts/integration green with TEST_DATABASE_URL; ruff + mypy --strict clean. Real E2E: `rag-lab demo` on PostgreSQL + pgvector (pgserver) with real embeddings (sentence-transformers all-MiniLM-L6-v2 behind a local OpenAI-compatible endpoint) and the real local cross-encoder: 7 documents, 84 chunks over 3 chunk sets, 52 golden questions, 624 measured queries in 5 min 19 s; every strategy met Recall@5 ≥ 85% and MRR@10 ≥ 0.75; best fixed-800-120 + hybrid_rerank (Recall@1 96.2%, MRR 0.978, p50 1.85 s on one vCPU). Saved as examples/sample-report.md. Not verified: Docker Compose build/run, a hosted LLM provider (generation metrics only with stubs), the Streamlit dashboard in a real browser.
- L4 Guide claim corrected with evidence: the first draft said full-text wins on product codes; the per-question ranks show it wins on distinctive shared terms (Final Sale, gift card, points, expire) and vectors win on paraphrases (plane/air, crushed/damaged, membership level/tier).
- L5 Native review: a0ad641..a15bae4 (review-7752c69bb59b214b, 3 lenses) approved; c5e066c Dockerfile alone (review-8b3eaafa95a000fa, item 5) approved; advisories fixed in 434ba8c.
- L6 Final review c5e066c..c672be4 (review-80c12c42f61adb17): one CRITICAL finding (removed try/except made one rejected LLM reply abort a generation batch), bounded correction in its own commit with a regression test, targeted validation **approved**, acknowledged. A whole-branch review requested by the stop hook stopped at lens_context_budget_exceeded (~8,900 lines); every slice was reviewed separately. Final suite: 302 passed with PostgreSQL.
