# Phase 6 — Evaluation (golden dataset and per-strategy report)

Objective: build a reviewed golden dataset of questions labelled with evidence spans, run the chunking × strategy benchmark matrix, and produce a reproducible comparative report (retrieval quality, latency, resources, and optionally generation quality).
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "6. Avaliação    Golden dataset e relatório por estratégia". MVP item "9. Dataset de perguntas com documentos relevantes esperados."
- **S2 — Golden dataset.** Agreed in the Phase 1 interview: LLM-synthetic questions plus manual review in Streamlit (edit, approve, reject, add your own). Relevance is span overlap on the reference layer ("relevância medida por sobreposição de spans").
- **S3 — Metrics and targets.** "Perguntas de avaliação 50", "Estratégias comparadas 4", "Recall@5 ≥ 85%", "MRR@10 ≥ 0,75", "Respostas com citações válidas ≥ 95%", "Falhas críticas de isolamento 0". "São critérios propostos para o experimento, não resultados alcançados."
- **S4 — Experiment.** "comparar recuperação vetorial, lexical, híbrida e híbrida com reranking no mesmo conjunto de perguntas. Registrar qualidade, latência e consumo de recursos." Benchmark = chunking × {vector, full-text, hybrid, hybrid+rerank} matrix.
- **S5 — Generation evaluation.** Agreed: deterministic citation check plus an optional LLM judge.
- **S6 — Reproducibility.** "Critério de conclusão: o repositório executa localmente, indexa um corpus de exemplo, responde a perguntas com fontes e gera um relatório comparativo reproduzível."
- **S7 — Quality bar.** TDD, layered architecture, docs in EN and PT with purpose, behavior and why.

Assumptions (not user-stated): a chunk is relevant when it overlaps a labelled span of the same document; Recall@k = share of a question's labelled spans touched by the top-k chunks; MRR@10 uses the first relevant chunk; only approved questions are evaluated; runs are stored in PostgreSQL and exported as JSON + Markdown; a CLI runs the same benchmark for reproducibility; golden datasets export/import by document content hash.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S2 S3 | inline | done | bca9bd1 |
| T2 | S2 | inline | done | bca9bd1, e390ebd |
| T3 | S2 | inline | done | 2743b87 |
| T4 | S3 S4 S5 | inline | done | 0791d81 |
| T5 | S6 | inline | done | 0791d81 |
| T6 | S7 | inline | done | (docs commit) |

- T1 Domain: golden question + evidence spans, relevance by overlap, Recall@k / MRR@k / hit rate, aggregation.
- T2 Persistence: migration 0004 (golden_questions, golden_evidence, evaluation_runs), repositories, contracts.
- T3 Golden dataset service: synthetic generation (seeded passage sampling + LLM), manual add by evidence excerpt, edit/approve/reject/delete, export/import; API + Streamlit review page.
- T4 Benchmark service: matrix run with warm-up, per-query latency and CPU time, optional generation metrics (citation validity, abstention) and LLM judge; persisted runs; API.
- T5 CLI `rag-lab` (benchmark, export report as JSON + Markdown).
- T6 Docs EN/PT, ADRs, README.

Route evidence: sequential dependent units, parent writes inline. Risk: item 1 (migration 0004, additive new tables).

## Log

- L1 (2026-10-31, user): "sim, faz o push. E continua até entregar todos os passos. Só para se precisar de minha definição"
- L2 (2026-11-01, user): "sempre faça a revisão com gentle-ai, não pergunte mais para esse projeto"
- L3 Native review: bca9bd1 alone (item 1, migration 0004) approved and acknowledged (review-1a2a44fb400e1fcb); advisories fixed in e390ebd (deferred no-cascade evidence FK, evidence/collection guard, update never inserts, run tie order) — 0004 was edited in place because it was never published (only the local test DB had it; downgraded and re-applied). e390ebd reviewed alone (review-930b6c945661e4d4) approved.
- L4 Verification: 287 passed with TEST_DATABASE_URL (pgvector via pgserver); `alembic check` no drift; 0004 downgrade/upgrade OK; ruff + mypy --strict clean. E2E: uvicorn + pgvector + stub embeddings/chat: golden questions added and approved via API, `rag-lab benchmark --method vector --method fulltext --method hybrid --generation --judge` wrote the Markdown + JSON report; the run is listed by `rag-lab runs` and `GET /benchmarks`. Not verified: real LLM provider, Docker Compose.
