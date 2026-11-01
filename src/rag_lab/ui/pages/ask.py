"""Ask page: one question, one strategy, a grounded answer and its evidence.

Three panels, as in the reference design: scope and strategy in the sidebar, the
answer in the middle (citations as badges, invalid ones flagged), and on the right
every source given to the LLM, with the cited ones marked. Seeing the sources next to
the answer is how you tell a retrieval miss from a generation mistake.
"""

import html
from typing import Any

import httpx
import streamlit as st

from rag_lab.ui import api_client
from rag_lab.ui.highlight import highlight_terms, render_answer
from rag_lab.ui.theme import apply_theme

METHODS = {
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
    "vector": "Vector",
    "fulltext": "Full-text",
}
ABSTENTION_REASONS = {
    "no_evidence_retrieved": "Retrieval found no evidence: the LLM was not called.",
    "model_abstained": "The LLM judged the retrieved sources insufficient.",
}

apply_theme()
client = api_client.get_client()


def _fail(error: Exception) -> None:
    st.error(f"API error: {error}")
    st.stop()


def _source_card(number: int, hit: dict[str, Any], cited: bool, query: str, name: str) -> str:
    location = " > ".join(hit["section_path"])
    if hit["page_numbers"]:
        pages = ", ".join(str(p) for p in hit["page_numbers"])
        location = f"p. {pages}" + (f" · {location}" if location else "")
    badge = '<span class="rl-badge">cited</span>' if cited else ""
    css = "rl-card selected" if cited else "rl-card"
    return (
        f'<div class="{css}"><div class="rl-card-title">[{number}] {html.escape(name)}{badge}'
        f'</div><div class="rl-card-meta">{html.escape(location)} · score {hit["score"]:.4f}'
        f'</div><div class="rl-snippet">{highlight_terms(hit["text"], query)}</div></div>'
    )


try:
    collections = client.list_collections()
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

st.markdown("## Ask")
if not collections:
    st.info("Create a collection and ingest documents on the Ingestion page first.")
    st.stop()

with st.sidebar:
    by_name = {c["name"]: c for c in collections}
    collection = by_name[st.selectbox("Collection", list(by_name))]
    chunk_set = st.selectbox("Chunk set", [c["key"] for c in collection["chunking_configs"]])
    method = st.selectbox("Retrieval strategy", list(METHODS), format_func=METHODS.__getitem__)
    top_k = st.slider("Sources for the LLM", 1, 15, 5)
    try:
        documents = client.list_documents(collection["id"])
    except (api_client.ApiError, httpx.HTTPError) as error:
        _fail(error)
    filenames = {d["id"]: d["filename"] for d in documents}
    selected = st.multiselect(
        "Only these documents", list(filenames), format_func=lambda i: filenames[i]
    )

with st.form("ask"):
    question = st.text_input("Question", placeholder="Ask something about your documents")
    submitted = st.form_submit_button("Answer", type="primary")

if not submitted or not question.strip():
    st.caption("Answers cite their sources as [n]; the model abstains when they are not enough.")
    st.stop()

with st.spinner("Retrieving and generating…"):
    try:
        result = client.answer(
            collection["id"], question, chunk_set, method, top_k, selected or None
        )
    except (api_client.ApiError, httpx.HTTPError) as error:
        _fail(error)

answer_column, evidence_column = st.columns([3, 2], gap="large")
cited_numbers = {c["number"] for c in result["citations"]}

with answer_column:
    if result["abstained"]:
        st.warning(ABSTENTION_REASONS.get(result["abstention_reason"], "Abstained."))
    if result["answer"]:
        # An abstention is not scored for citations: show its text as-is.
        body = (
            html.escape(result["answer"])
            if result["abstained"]
            else render_answer(result["answer"], set(result["invalid_citations"]))
        )
        st.markdown(
            f'<div class="rl-card"><div class="rl-answer">{body}</div></div>',
            unsafe_allow_html=True,
        )
    if not result["abstained"]:
        status = (
            '<span class="rl-badge">citations valid</span>'
            if result["citations_valid"]
            else '<span class="rl-badge warn">citations invalid or missing</span>'
        )
        st.markdown(status, unsafe_allow_html=True)
    timings = result["timings"]
    tokens = (
        f" · {result['prompt_tokens']} + {result['completion_tokens']} tokens"
        if result["prompt_tokens"] is not None
        else ""
    )
    st.markdown(
        f'<div class="rl-card-meta">{html.escape(result["model"] or "no LLM call")}{tokens} · '
        f"retrieval {timings['retrieval']['total_ms']:.0f} ms · "
        f"generation {timings['generation_ms']:.0f} ms"
        "</div>",
        unsafe_allow_html=True,
    )

with evidence_column:
    st.markdown(f"#### Evidence ({len(result['sources'])} sources)")
    for number, hit in enumerate(result["sources"], start=1):
        name = filenames.get(hit["document_id"], hit["document_id"])
        st.markdown(
            _source_card(number, hit, number in cited_numbers, question, name),
            unsafe_allow_html=True,
        )
