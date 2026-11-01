"""Streamlit entry point: `streamlit run src/rag_lab/ui/app.py`."""

from pathlib import Path

import streamlit as st

from rag_lab.ui.theme import brand

PAGES_DIR = Path(__file__).parent / "pages"

st.set_page_config(page_title="RAG Benchmark Lab", page_icon="🔎", layout="wide")

with st.sidebar:
    brand()

navigation = st.navigation(
    [
        st.Page(PAGES_DIR / "home.py", title="Overview", default=True),
        st.Page(PAGES_DIR / "ingestion.py", title="Ingestion"),
        st.Page(PAGES_DIR / "retrieval.py", title="Retrieval"),
        st.Page(PAGES_DIR / "ask.py", title="Ask"),
        st.Page(PAGES_DIR / "golden.py", title="Golden dataset"),
        st.Page(PAGES_DIR / "benchmark.py", title="Benchmark"),
    ]
)
navigation.run()
