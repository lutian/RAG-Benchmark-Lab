"""Landing page: what the lab does and how the pipeline is organised."""

import streamlit as st

from rag_lab.ui.theme import apply_theme

apply_theme()

st.markdown("# RAG Benchmark Lab")
st.markdown(
    "Compare **lexical**, **vector**, **hybrid** and **reranked** retrieval on your own "
    "documents, with reproducible experiments and grounded answers."
)

col_ingest, col_query = st.columns(2)
with col_ingest:
    st.markdown(
        '<div class="rl-card"><div class="rl-card-title">Ingestion pipeline</div>'
        '<div class="rl-card-meta">Document → Parsing → Reference text → Chunking → '
        "Embedding → Indexing</div></div>",
        unsafe_allow_html=True,
    )
with col_query:
    st.markdown(
        '<div class="rl-card"><div class="rl-card-title">Question pipeline</div>'
        '<div class="rl-card-meta">Question → Retrieval strategy → Filters → Reranking → '
        "Context → LLM → Answer with citations</div></div>",
        unsafe_allow_html=True,
    )
