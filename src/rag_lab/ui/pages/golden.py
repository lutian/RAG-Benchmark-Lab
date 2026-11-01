"""Golden dataset page: generate questions with the LLM, add your own, and review them.

Only **approved** questions are benchmarked. Each card shows the evidence excerpt the
question was labelled with, so a reviewer can check that the passage really answers it.
"""

import html
import json
from typing import Any

import httpx
import streamlit as st

from rag_lab.application.reporting import TARGET_QUESTIONS
from rag_lab.ui import api_client
from rag_lab.ui.theme import apply_theme

STATUS_BADGE = {"draft": "neutral", "approved": "", "rejected": "warn"}

apply_theme()
client = api_client.get_client()


def _fail(error: Exception) -> None:
    st.error(f"API error: {error}")
    st.stop()


def _act(action: Any, *args: Any) -> None:
    try:
        action(*args)
    except (api_client.ApiError, httpx.HTTPError) as error:
        st.error(str(error))
    else:
        st.rerun()


try:
    collections = client.list_collections()
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

st.markdown("## Golden dataset")
if not collections:
    st.info("Create a collection and ingest documents on the Ingestion page first.")
    st.stop()

with st.sidebar:
    by_name = {c["name"]: c for c in collections}
    collection = by_name[st.selectbox("Collection", list(by_name))]
    status_filter = st.radio("Show", ["all", "draft", "approved", "rejected"], horizontal=True)

collection_id = collection["id"]
try:
    questions = client.golden_questions(collection_id)
    documents = client.list_documents(collection_id)
except (api_client.ApiError, httpx.HTTPError) as error:
    _fail(error)

counts = {s: sum(q["status"] == s for q in questions) for s in ("draft", "approved", "rejected")}
metrics = st.columns(4)
metrics[0].metric("Approved", f"{counts['approved']} / {TARGET_QUESTIONS}")
metrics[1].metric("Drafts to review", counts["draft"])
metrics[2].metric("Rejected", counts["rejected"])
metrics[3].metric("Total", len(questions))

generate_column, manual_column = st.columns(2, gap="large")
with generate_column, st.expander("Generate with the LLM"), st.form("generate"):
    st.caption(
        "Samples passages from the documents (seeded) and asks the LLM for one question "
        "per passage. Questions start as drafts."
    )
    count = st.number_input("Questions", 1, 200, 10)
    seed = st.number_input("Seed", 0, 1_000_000, 0, help="Same seed + documents = same passages.")
    if st.form_submit_button("Generate", type="primary"):
        with st.spinner("Asking the LLM…"):
            _act(client.golden_generate, collection_id, int(count), int(seed))

with manual_column, st.expander("Add your own question"), st.form("manual", clear_on_submit=True):
    by_id = {d["id"]: d["filename"] for d in documents}
    document_id = st.selectbox("Document", list(by_id), format_func=lambda i: by_id[i])
    text = st.text_input("Question")
    excerpt = st.text_area("Evidence (paste the text that answers it)")
    if st.form_submit_button("Add", type="primary") and document_id:
        _act(client.golden_add, collection_id, text, document_id, excerpt)

shown = [q for q in questions if status_filter == "all" or q["status"] == status_filter]
if not shown:
    st.info("No questions here yet.")
for question in shown:
    qid = question["id"]
    evidence = "".join(
        f'<div class="rl-card-meta">{html.escape(e["filename"])} · chars {e["start"]}-{e["end"]}'
        f'</div><div class="rl-snippet"><mark class="rl-evidence">{html.escape(e["excerpt"])}'
        "</mark></div>"
        for e in question["evidence"]
    )
    badge = STATUS_BADGE[question["status"]]
    st.markdown(
        f'<div class="rl-card"><div class="rl-card-title">{html.escape(question["text"])}'
        f'<span class="rl-badge {badge}">{question["status"]}</span>'
        f'<span class="rl-badge neutral">{question["origin"]}</span></div>{evidence}</div>',
        unsafe_allow_html=True,
    )
    edit, save, approve, reject, delete = st.columns([5, 1, 1, 1, 1])
    new_text = edit.text_input(
        "Edit", question["text"], key=f"text-{qid}", label_visibility="collapsed"
    )
    # Explicit save: no request on every rerun, and a blank edit is never sent.
    unchanged = not new_text.strip() or new_text == question["text"]
    if save.button("Save", key=f"save-{qid}", disabled=unchanged):
        _act(client.golden_update, collection_id, qid, new_text, None)
    if approve.button("Approve", key=f"approve-{qid}", disabled=question["status"] == "approved"):
        _act(client.golden_update, collection_id, qid, None, "approved")
    if reject.button("Reject", key=f"reject-{qid}", disabled=question["status"] == "rejected"):
        _act(client.golden_update, collection_id, qid, None, "rejected")
    if delete.button("Delete", key=f"delete-{qid}"):
        _act(client.golden_delete, collection_id, qid)

st.divider()
export_column, import_column = st.columns(2, gap="large")
with export_column:
    # Built on demand, not on every rerun of the review page.
    if st.button("Prepare export (JSON)"):
        try:
            exported = json.dumps(client.golden_export(collection_id), indent=2, ensure_ascii=False)
        except (api_client.ApiError, httpx.HTTPError) as error:
            st.error(str(error))
        else:
            st.download_button(
                "Download dataset",
                data=exported,
                file_name=f"golden-{collection['name']}.json",
                mime="application/json",
            )
with import_column:
    upload = st.file_uploader("Import dataset (JSON)", type=["json"])
    if upload is not None and st.button("Import"):
        try:
            report = client.golden_import(collection_id, json.loads(upload.getvalue()))
            st.success(f"Imported {report['imported']}, skipped {report['skipped']}")
        except (api_client.ApiError, httpx.HTTPError, ValueError) as error:
            st.error(str(error))
