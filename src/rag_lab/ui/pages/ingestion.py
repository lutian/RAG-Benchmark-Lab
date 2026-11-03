"""Ingestion page: collections, uploads, and a chunk-boundary inspector.

Layout follows the reference design: filters in the sidebar, the document list in
the middle, and a preview with highlighted evidence (here: chunk boundaries) on the right.
"""

import html
from typing import Any

import httpx
import streamlit as st

from rag_lab.ui import api_client
from rag_lab.ui.highlight import as_preview_block, render_chunks
from rag_lab.ui.theme import apply_theme

STRATEGIES = ["fixed", "recursive", "structure"]
# Common PostgreSQL text search configurations; "simple" = no stemming (mixed languages).
LANGUAGES = ["english", "portuguese", "spanish", "french", "german", "italian", "simple"]

apply_theme()
client = api_client.get_client()


def _fail(error: Exception) -> None:
    st.error(f"API error: {error}")
    st.stop()


# Widget state cannot change after the widget is drawn, so a rename stores the new name
# here and the selectbox picks it up on the next run.
PENDING_SELECTION = "ingestion-pending-collection"
SELECTED_COLLECTION = "ingestion-collection"
# Which confirmation dialog is open ("rename" or "delete"). Kept in session state so the
# dialog survives full reruns; closing it with X or Escape clears it.
OPEN_DIALOG = "ingestion-collection-dialog"


def _close_dialog() -> None:
    st.session_state.pop(OPEN_DIALOG, None)


def _open_dialog(name: str) -> None:
    st.session_state[OPEN_DIALOG] = name


@st.dialog("Rename collection", on_dismiss=_close_dialog)
def _rename_dialog(collection: dict[str, Any]) -> None:
    new_name = st.text_input("New name", value=collection["name"])
    st.caption(
        f"Rename **{collection['name']}** to **{new_name.strip() or '…'}**? "
        "Documents, indexes, golden questions and runs are kept."
    )
    confirm, cancel = st.columns(2)
    confirmed = confirm.button("Rename", type="primary", width="stretch")
    if cancel.button("Cancel", width="stretch"):
        _close_dialog()
        st.rerun()
    if confirmed:
        try:
            renamed = client.rename_collection(collection["id"], new_name)
        except (api_client.ApiError, httpx.HTTPError) as error:
            st.error(str(error))
            return
        st.session_state[PENDING_SELECTION] = renamed["name"]
        _close_dialog()
        st.rerun()


@st.dialog("Delete collection", on_dismiss=_close_dialog)
def _delete_dialog(collection: dict[str, Any]) -> None:
    st.warning(
        f"Delete **{collection['name']}**? This permanently removes its documents, "
        "chunk sets, vectors, golden questions and benchmark runs. It cannot be undone."
    )
    confirm, cancel = st.columns(2)
    confirmed = confirm.button("Delete", type="primary", width="stretch")
    if cancel.button("Cancel", width="stretch"):
        _close_dialog()
        st.rerun()
    if confirmed:
        try:
            client.delete_collection(collection["id"])
        except (api_client.ApiError, httpx.HTTPError) as error:
            st.error(str(error))
            return
        st.session_state.pop(SELECTED_COLLECTION, None)
        _close_dialog()
        st.rerun()


try:
    collections = client.list_collections()
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

# --- sidebar: collection selection and creation ---------------------------------------
with st.sidebar:
    st.markdown("**Collection**")
    names = [c["name"] for c in collections]
    pending = st.session_state.pop(PENDING_SELECTION, None)
    if pending in names:
        st.session_state[SELECTED_COLLECTION] = pending
    selected_name = (
        st.selectbox("Collection", names, key=SELECTED_COLLECTION, label_visibility="collapsed")
        if names
        else None
    )
    if selected_name is not None:
        current = next(c for c in collections if c["name"] == selected_name)
        rename_column, delete_column = st.columns(2)
        rename_column.button("Rename", width="stretch", on_click=_open_dialog, args=("rename",))
        delete_column.button("Delete", width="stretch", on_click=_open_dialog, args=("delete",))
        dialog = st.session_state.get(OPEN_DIALOG)
        if dialog == "rename":
            _rename_dialog(current)
        elif dialog == "delete":
            _delete_dialog(current)

    with st.expander("New collection"), st.form("new-collection", clear_on_submit=True):
        name = st.text_input("Name")
        strategies = st.multiselect("Chunking strategies", STRATEGIES, default=STRATEGIES)
        size = st.number_input("Chunk size (characters)", 50, 20_000, 800, step=50)
        overlap = st.number_input("Overlap (characters)", 0, 5_000, 120, step=10)
        language = st.selectbox(
            "Full-text language",
            LANGUAGES,
            help="Stemming and stop words for full-text search. Fixed after creation.",
        )
        if st.form_submit_button("Create", type="primary"):
            configs = [
                {"strategy": s, "chunk_size": int(size), "chunk_overlap": int(overlap)}
                for s in strategies
            ]
            try:
                client.create_collection(name, configs, language)
                st.rerun()
            except api_client.ApiError as error:
                st.error(str(error))

st.markdown("## Ingestion")
if selected_name is None:
    st.info("Create a collection in the sidebar to start ingesting documents.")
    st.stop()

collection: dict[str, Any] = next(c for c in collections if c["name"] == selected_name)
chunk_set_keys = [c["key"] for c in collection["chunking_configs"]]

# --- upload --------------------------------------------------------------------------
uploads = st.file_uploader(
    "Upload PDF, Markdown or TXT",
    type=["pdf", "md", "markdown", "txt"],
    accept_multiple_files=True,
)
if uploads and st.button("Ingest files", type="primary"):
    for upload in uploads:
        try:
            result = client.upload(collection["id"], upload.name, upload.getvalue())
            counts = ", ".join(f"{k}: {v}" for k, v in result["chunk_counts"].items())
            note = "already ingested" if result["duplicate"] else "ingested"
            st.success(f"{upload.name} {note} — chunks per set: {counts}")
        except api_client.ApiError as error:
            st.error(f"{upload.name}: {error}")

try:
    documents = client.list_documents(collection["id"])
except api_client.ApiError as error:
    _fail(error)

st.markdown(
    f'<div class="rl-card-meta">{len(documents)} documents · '
    f"{len(chunk_set_keys)} chunk sets</div>",
    unsafe_allow_html=True,
)
if not documents:
    st.info("No documents yet. Upload files above.")
    st.stop()

list_column, preview_column = st.columns([2, 3], gap="large")

with list_column:
    for document in documents:
        title = html.escape(document["metadata"].get("title", document["filename"]))
        st.markdown(
            f'<div class="rl-card"><div class="rl-card-title">{html.escape(document["filename"])}'
            f'<span class="rl-badge neutral">{html.escape(document["media_type"])}</span></div>'
            f'<div class="rl-card-meta">{title} · {document["characters"]:,} characters</div>'
            "</div>",
            unsafe_allow_html=True,
        )

with preview_column:
    by_id = {d["id"]: d for d in documents}
    document_id = st.selectbox(
        "Preview document", list(by_id), format_func=lambda i: by_id[i]["filename"]
    )
    config_key = st.selectbox("Chunk set", chunk_set_keys)

    # --- indexing: progress, binding and the action that embeds the chunk set ---------
    try:
        index = client.index_status(collection["id"])
    except api_client.ApiError as error:
        _fail(error)
    progress = next(s for s in index["chunk_sets"] if s["config_key"] == config_key)
    binding = index["embedding"]
    model = (
        f"{html.escape(binding['model'])} ({binding['dimensions']}d)" if binding else "not indexed"
    )
    st.markdown(
        f'<div class="rl-card-meta">{progress["embedded"]} / {progress["total"]} chunks embedded'
        f'<span class="rl-badge neutral">{model}</span></div>',
        unsafe_allow_html=True,
    )
    action, option = st.columns([1, 2])
    rebuild = option.checkbox(
        "Rebuild (drop every vector of this collection and re-embed)", value=False
    )
    if action.button("Index chunk set", type="primary"):
        with st.spinner("Embedding chunks…"):
            try:
                report = client.index_chunk_set(collection["id"], config_key, rebuild)
                st.success(
                    f"Embedded {report['embedded']} chunks in {report['seconds']} s "
                    f"({report['total']} total)"
                )
            except (api_client.ApiError, httpx.HTTPError) as error:
                st.error(f"Indexing failed: {error}")

    try:
        detail = client.get_document(collection["id"], document_id)
        chunks = client.list_chunks(collection["id"], config_key, document_id)
    except api_client.ApiError as error:
        _fail(error)

    sizes = [c["end"] - c["start"] for c in chunks]
    metrics = st.columns(3)
    metrics[0].metric("Chunks", len(chunks))
    metrics[1].metric("Avg size", round(sum(sizes) / len(sizes)) if sizes else 0)
    metrics[2].metric("Max size", max(sizes, default=0))
    preview = render_chunks(detail["text"], [(c["start"], c["end"]) for c in chunks])
    st.markdown(as_preview_block(preview), unsafe_allow_html=True)
