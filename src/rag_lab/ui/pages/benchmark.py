"""Benchmark dashboard: run the experiment and compare strategies against the targets.

Reading order follows the questions the lab answers: the headline (best strategy vs
targets), the chunking x method matrix, the latency cost, generation quality, and the
per-question drill-down that shows *where* a strategy lost.
"""

from typing import Any

import altair as alt
import httpx
import streamlit as st

from rag_lab.application.reporting import (
    TARGET_MRR_AT_10,
    TARGET_QUESTIONS,
    TARGET_RECALL_AT_5,
    TARGET_VALID_CITATIONS,
)
from rag_lab.ui import api_client
from rag_lab.ui.dashboard import (
    METHOD_LABELS,
    drilldown_frame,
    latency_frame,
    matrix_frame,
    metric_options,
    strategy_label,
    summary_table,
)
from rag_lab.ui.theme import apply_theme

ACCENT = "#2563EB"  # design accent; the only hue used by these charts

apply_theme()
client = api_client.get_client()


def _fail(error: Exception) -> None:
    st.error(f"API error: {error}")
    st.stop()


def _delta(value: float | None, target: float) -> str | None:
    if value is None:
        return None
    return f"{value - target:+.1%} vs target {target:.0%}"


try:
    collections = client.list_collections()
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

st.markdown("## Benchmark")
if not collections:
    st.info("Create a collection, index it and approve golden questions first.")
    st.stop()

with st.sidebar:
    by_name = {c["name"]: c for c in collections}
    collection = by_name[st.selectbox("Collection", list(by_name))]
collection_id = collection["id"]
chunk_sets = [c["key"] for c in collection["chunking_configs"]]

with st.expander("Run a new benchmark"), st.form("run"):
    chosen_sets = st.multiselect("Chunk sets", chunk_sets, default=chunk_sets)
    chosen_methods = st.multiselect(
        "Methods", list(METHOD_LABELS), default=list(METHOD_LABELS),
        format_func=METHOD_LABELS.__getitem__,
    )  # fmt: skip
    generation = st.checkbox("Evaluate answers (one LLM call per question x strategy)")
    judge = st.checkbox("LLM judge of answer support (needs answers)")
    if st.form_submit_button("Run benchmark", type="primary"):
        body: dict[str, Any] = {
            "chunk_sets": chosen_sets,
            "methods": chosen_methods,
            "generation": generation,
            "judge": judge and generation,
        }
        with st.spinner("Running every question with every strategy…"):
            try:
                client.run_benchmark(collection_id, body)
                st.rerun()
            except (api_client.ApiError, httpx.HTTPError) as error:
                st.error(str(error))

try:
    runs = client.benchmark_runs(collection_id)
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)
if not runs:
    st.info("No benchmark runs yet. Approve golden questions, then run one above.")
    st.stop()

with st.sidebar:
    labels = {
        r["id"]: f"{r['created_at'][:16].replace('T', ' ')} · {r['question_count']} q" for r in runs
    }
    run_id = st.selectbox("Run", list(labels), format_func=labels.__getitem__)
run = next(r for r in runs if r["id"] == run_id)
summaries = run["summaries"]
options = metric_options(summaries)
headline = "Recall@5" if "Recall@5" in options else next(iter(options))
best = max(
    summaries, key=lambda s: (s["recall"].get(options[headline].split(":")[-1], 0), s["mrr"])
)

# --- headline: best strategy against the targets --------------------------------------
st.markdown(
    f'<div class="rl-card-meta">Best strategy: <b>{strategy_label(best)}</b></div>',
    unsafe_allow_html=True,
)
tiles = st.columns(4)
tiles[0].metric("Questions", f"{run['question_count']} / {TARGET_QUESTIONS}")
recall5 = best["recall"].get("5")
tiles[1].metric(
    "Recall@5", "—" if recall5 is None else f"{recall5:.1%}", _delta(recall5, TARGET_RECALL_AT_5)
)
tiles[2].metric(
    "MRR", f"{best['mrr']:.3f}",
    _delta(best["mrr"], TARGET_MRR_AT_10) if run["config"]["mrr_cutoff"] == 10 else None,
)  # fmt: skip
tiles[3].metric(
    "Valid citations",
    "—" if best["citation_valid_rate"] is None else f"{best['citation_valid_rate']:.1%}",
    _delta(best["citation_valid_rate"], TARGET_VALID_CITATIONS),
)

# --- the matrix ----------------------------------------------------------------------
matrix_column, latency_column = st.columns(2, gap="large")
with matrix_column:
    metric = st.selectbox("Metric", list(options), index=list(options).index(headline))
    frame = matrix_frame(summaries, options[metric])
    base = alt.Chart(frame).encode(
        x=alt.X("method:N", title=None, sort=list(METHOD_LABELS.values())),
        y=alt.Y("chunk_set:N", title=None),
        tooltip=["chunk_set", "method", alt.Tooltip("value:Q", format=".3f", title=metric)],
    )
    heat = base.mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "value:Q", title=metric,
            scale=alt.Scale(domain=[0, 1], range=["#EFF4FF", ACCENT]),
        )
    )  # fmt: skip
    labels_layer = base.mark_text(fontSize=12).encode(
        text=alt.Text("value:Q", format=".2f"),
        color=alt.condition("datum.value > 0.6", alt.value("white"), alt.value("#1F2937")),
    )
    st.altair_chart(
        (heat + labels_layer).properties(height=60 * len(set(frame.chunk_set)) + 40),
        use_container_width=True,
    )

with latency_column:
    st.markdown("Median latency per query (p95 in the tooltip)")
    latency = latency_frame(summaries)
    bars = (
        alt.Chart(latency)
        .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("p50_ms:Q", title="p50 ms"),
            y=alt.Y("strategy:N", title=None, sort="-x"),
            tooltip=[
                "strategy",
                alt.Tooltip("p50_ms:Q", format=".1f"),
                alt.Tooltip("p95_ms:Q", format=".1f"),
                alt.Tooltip("cpu_ms:Q", format=".1f", title="CPU ms"),
            ],
        )
    )
    st.altair_chart(bars.properties(height=26 * len(latency) + 40), use_container_width=True)

st.markdown("#### All strategies")
st.dataframe(summary_table(summaries), hide_index=True, use_container_width=True)

# --- drill-down and report ---------------------------------------------------------------
try:
    full = client.benchmark_run(collection_id, run_id)
    questions = {q["id"]: q["text"] for q in client.golden_questions(collection_id)}
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)
st.markdown("#### Rank of the first relevant chunk, per question")
st.caption("Empty = the evidence was not retrieved. Rows empty everywhere point at the label.")
st.dataframe(drilldown_frame(full["results"], questions), hide_index=True, use_container_width=True)

if st.button("Prepare report (Markdown)"):
    try:
        report = client.benchmark_report(collection_id, run_id)
    except (api_client.ApiError, httpx.HTTPError) as error:
        st.error(str(error))
    else:
        st.download_button("Download report", report, file_name=f"benchmark-{run_id[:8]}.md")
        st.markdown(report)
