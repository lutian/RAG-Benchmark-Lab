"""Retrieval page: run one query with every search method and compare them side by side.

Each column is one method. Every hit shows its rank, score and the rank the *other*
methods gave the same chunk, which is what explains why a method won or lost.
"""

import html
from typing import Any

import httpx
import streamlit as st

from rag_lab.ui import api_client
from rag_lab.ui.highlight import highlight_terms
from rag_lab.ui.theme import apply_theme

METHODS = {"vector": "Vector", "fulltext": "Full-text", "hybrid": "Hybrid (RRF)"}

apply_theme()
client = api_client.get_client()


def _fail(error: Exception) -> None:
    st.error(f"API error: {error}")
    st.stop()


def _hit_card(hit: dict[str, Any], query: str, filenames: dict[str, str]) -> str:
    location = " > ".join(hit["section_path"])
    if hit["page_numbers"]:
        pages = ", ".join(str(p) for p in hit["page_numbers"])
        location = f"p. {pages}" + (f" · {location}" if location else "")
    ranks = "".join(
        f'<span class="rl-badge neutral">{html.escape(METHODS.get(name, name))} #{rank}</span>'
        for name, rank in hit["ranks"].items()
    )
    filename = html.escape(filenames.get(hit["document_id"], hit["document_id"]))
    return (
        f'<div class="rl-card"><div class="rl-card-title">#{hit["rank"]} {filename}'
        f'<span class="rl-badge">{hit["score"]:.4f}</span>{ranks}</div>'
        f'<div class="rl-card-meta">{html.escape(location)} · chars {hit["start"]}-{hit["end"]}'
        f'</div><div class="rl-snippet">{highlight_terms(hit["text"], query)}</div></div>'
    )


try:
    collections = client.list_collections()
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

st.markdown("## Retrieval")
if not collections:
    st.info("Create a collection and ingest documents on the Ingestion page first.")
    st.stop()

# --- sidebar: scope and filters -------------------------------------------------------
with st.sidebar:
    by_name = {c["name"]: c for c in collections}
    collection = by_name[st.selectbox("Collection", list(by_name))]
    chunk_set = st.selectbox("Chunk set", [c["key"] for c in collection["chunking_configs"]])
    top_k = st.slider("Top k", 1, 30, 5)
    try:
        documents = client.list_documents(collection["id"])
    except api_client.ApiError as error:
        _fail(error)
    filenames = {d["id"]: d["filename"] for d in documents}
    selected = st.multiselect(
        "Only these documents",
        list(filenames),
        format_func=lambda i: filenames[i],
        help="Empty = search every document of the collection.",
    )

with st.form("search"):
    query = st.text_input("Question", placeholder="Ask something about your documents")
    submitted = st.form_submit_button("Compare methods", type="primary")

if not submitted or not query.strip():
    st.caption(
        "Runs vector, full-text and hybrid search on the same chunk set. "
        f"Full-text language: {collection['fulltext_language']}."
    )
    st.stop()

columns = st.columns(len(METHODS), gap="medium")
for column, (method, label) in zip(columns, METHODS.items(), strict=True):
    with column:
        st.markdown(f"#### {label}")
        try:
            result = client.search(
                collection["id"], query, chunk_set, method, top_k, selected or None
            )
        except api_client.ApiError as error:
            st.warning(str(error))
            continue
        timings = result["timings"]
        st.markdown(
            f'<div class="rl-card-meta">{len(result["hits"])} hits · '
            f"{timings['total_ms']:.1f} ms (embed {timings['embed_ms']:.1f} · "
            f"search {timings['search_ms']:.1f})</div>",
            unsafe_allow_html=True,
        )
        if not result["hits"]:
            st.info("No chunk matched.")
        for hit in result["hits"]:
            st.markdown(_hit_card(hit, query, filenames), unsafe_allow_html=True)
