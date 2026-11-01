# Benchmark guide — running and reading the experiment

This guide shows how to run the comparison and how to turn its numbers into answers: why chunking matters, when lexical search beats embeddings, when reranking helps, and whether a bad answer comes from retrieval or generation.

> Other languages: [Português](../pt-BR/benchmark-guide.md)

## Quick path

```bash
make demo              # local: ingest + index + golden import + benchmark of examples/
make demo-docker       # same inside the Compose stack; report copied to ./reports
```

Then open **Benchmark** in the UI (http://localhost:8501) to explore the run. [`examples/sample-report.md`](../../examples/sample-report.md) is a reference run.

## What one run measures

| Axis | Values |
|------|--------|
| Chunking | every chunk set of the collection (default: `fixed`, `recursive`, `structure` at 800/120) |
| Retrieval | `vector`, `fulltext`, `hybrid` (RRF), `hybrid_rerank` |
| Questions | every **approved** golden question (target: 50) |
| Retrieval quality | Recall@1/5/10, MRR@10, by evidence-span overlap |
| Cost | p50/p95 latency and CPU time per query (after a warm-up) |
| Answers (optional) | abstention rate, valid citations, LLM-judge support, tokens |

## Reading the results — the four questions

The numbers below come from the reference run (52 questions, MiniLM embeddings, local cross-encoder, CPU only).

### 1. Why does chunking affect retrieval?

The same questions and the same method score differently per chunk set. In the reference run, full-text Recall@5 goes from 96.2% (`fixed`) to 92.3% (`structure`), while vector search is best on `recursive` (100%). Smaller, structure-aligned chunks (42 instead of 21) give lexical search fewer words per chunk, so a question that matches a sentence with *paraphrased* words no longer shares terms with its chunk. Look at the **Rank of the first relevant chunk** table in the dashboard: a question whose rank changes a lot between chunk sets is a chunking effect, not a model effect.

### 2. When does lexical search (full-text) beat embeddings?

Check **Head to head → Does lexical beat vectors?** In the reference run it is 12 wins, 12 losses, 28 ties on `fixed`, and the per-question ranks show a clear pattern:

- **Full-text wins** when the question repeats a distinctive term of the passage and the embedding confuses neighbouring topics: "Can I return something with a red **Final Sale** tag?", "Do I earn **points** when I buy a **gift card**?", "When do loyalty points **expire**?", "How long does a shop **keep** an order…?" (vector rank 7, full-text rank 1).
- **Vectors win** on paraphrases, where the question and the passage use different words for the same idea: "sent by **plane**" vs "by **air**", "a box that arrived **crushed**" vs "**damaged** in transit", "**membership level**" vs "**tier**" (full-text found nothing), "**split my payment**" vs "**instalments**".

Full-text is also roughly 10 times faster (2–3 ms vs 23–25 ms p50), because it needs no embedding call. Hybrid search takes the best of both: it wins 8 and loses 3 against vectors alone.

### 3. When does reranking help?

**Does reranking help hybrid?** is 14 wins, 0 losses on `fixed`. Recall@5 barely moves (it was already 98–100%), but **Recall@1 jumps from ~71% to ~96%** and MRR from 0.84 to 0.98. Reranking helps when the evidence is already among the candidates but not at the top. It cannot rescue a chunk the first stage never retrieved. The cost is clear in the latency and CPU columns: ~1.8 s per query for 30 candidates on one CPU, which is why it reorders a short list only.

### 4. Is the problem in retrieval or in generation?

Run with **Evaluate answers** (and optionally the judge). The report's *Retrieval or generation?* section splits bad answers:

- **Retrieval miss**: the evidence was not among the `answer_top_k` sources the LLM saw. Fix chunking, the method or `top_k`; a better LLM will not help.
- **Generation error**: the evidence was there and the answer still abstained, cited wrongly or was unsupported. Fix the prompt, the model or its settings.

The **Ask** page shows the same diagnosis for a single question: every source given to the LLM, with the cited ones marked.

## Using your own corpus

1. Create a collection on **Ingestion** (pick the full-text language of your documents) and upload files.
2. Press **Index chunk set** for each chunk set.
3. On **Golden dataset**, generate drafts with the LLM and **review them**: edit questions that copy the passage word for word (they inflate full-text scores), reject vague ones, add your own. Approve at least 50.
4. Run the benchmark from **Benchmark** or `rag-lab benchmark --collection <name>`.
5. Export the golden dataset (JSON) to rerun the same benchmark later or in another database.

## Pitfalls

| Pitfall | Why it matters | What the lab does |
|---------|----------------|-------------------|
| Synthetic questions copy the passage | Inflates lexical recall | Paraphrase instruction in the prompt + human review |
| Cold model on the first query | One question absorbs seconds of loading | Unmeasured warm-up per strategy |
| Approximate vector index | Recall loss blamed on the embeddings | Exact search (ADR-014) |
| Comparing models with different vectors | Similarities from different spaces | Collections bound to one embedding model |
| Small golden set | One question = 2% of Recall@5 with 50 questions | Report shows the question count; target 50 |
| Mixed-language corpus | Wrong stemming hurts full-text | Full-text language per collection; `simple` for mixed corpora |
