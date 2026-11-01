# Phase 5 — Generation (grounded answers, citations, abstention)

Objective: answer a question from the chunks one retrieval strategy returns, with numbered citations that are checked deterministically, and abstain when the evidence is insufficient.
Branch: `feat/foundation-ingestion` · Delivery: `ask-on-risk`, phase-by-phase · Test runner: `uv run pytest`.

## Specs

- **S1 — Phase deliverable.** "5. Geração    Respostas fundamentadas e citações". MVP item "8. Respostas com citações e abstenção quando não houver evidência."
- **S2 — Flow.** "Pergunta → Busca selecionada → Filtros → Reranking → Contexto → LLM → Resposta".
- **S3 — Target.** "Respostas com citações válidas ≥ 95%": citation validity must be measurable per answer.
- **S4 — Diagnosis.** "como diagnosticar se o problema está na recuperação ou na geração": the answer exposes the retrieved sources, which ones it cited, and separate retrieval/generation latency.
- **S5 — Provider.** "Os LLMs e modelos de embeddings devem ser via API ... chamar openai, openrouter, ou outro openai compatible"; settings from `.env` (`LLM_*`, including reasoning effort / temperature rules from Phase 2).
- **S6 — Quality bar.** TDD, layered architecture, docs in EN and PT with purpose, behavior and why.

Assumptions (not user-stated): citations are bracketed source numbers (`[1]`, `[2, 3]`); the model abstains by answering with a fixed sentinel; with no retrieved chunk the service abstains without calling the LLM; a citation is valid when it points to a provided source; the LLM judge is Phase 6.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T1 | S1 S3 | inline | done | f06d447 |
| T2 | S1 S2 S4 | inline | done | f06d447 |
| T3 | S5 | inline | done | 9aa21b3 |
| T4 | S4 | inline | done | 9aa21b3 |
| T5 | S6 | inline | done | ae4f932 |

- T1 Domain: citation parsing, abstention sentinel, citation validity.
- T2 Application: `ChatModel` port, prompt builder, `AnswerService` (retrieval → context → LLM → validated answer, timings, token usage).
- T3 Adapter: `OpenAICompatibleChatModel` over LangChain `ChatOpenAI` (Chat Completions only), tested against a local stub.
- T4 API `POST /collections/{id}/answer`; Streamlit "Ask" page (answer with citation links, evidence panel).
- T5 Docs EN/PT, ADR-015, README.

Route evidence: sequential dependent units, parent writes inline. Risk: none expected (new endpoint and modules; no schema or dependency change).

## Log

- L1 (2026-10-31, user): "sim, faz o push. E continua até entregar todos os passos. Só para se precisar de minha definição"
- L2 Verification: 232 passed with TEST_DATABASE_URL; ruff + mypy --strict clean. E2E uvicorn + pgvector + stub /v1/embeddings + stub /v1/chat/completions with LLM_REASONING_EFFORT=low: answer "The bank raised interest rates [1]." with citations_valid, 2 sources, usage 50+8; the request carried `reasoning_effort: low` and no `temperature`. Not verified: a real LLM provider.
- L3 Native review (consent per standing instruction): range 244cce9..ae4f932 (rerank fix + phase 5), lineage review-5e4729144c163928, four lenses, **approved**, acknowledged. Advisories fixed afterwards: HTTP 200 error bodies → GenerationProviderError (502); chat max_retries 1; one citation pattern shared by domain and UI; UI flags citations from the API's `invalid_citations`; answer timings nested as `retrieval` + `generation_ms` + `total_ms`; explicit no-evidence branch; separate UI generation timeout; markdown-wrapped NO_ANSWER detected; boolean rerank scores rejected; answer API test with strategy + document filter.
