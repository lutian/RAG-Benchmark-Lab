"""The Streamlit shell renders the home page with the project identity."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import rag_lab.ui

HOME_PAGE = str(Path(rag_lab.ui.__file__).parent / "pages" / "home.py")


def test_home_page_renders_without_errors() -> None:
    app = AppTest.from_file(HOME_PAGE).run()

    assert not app.exception
    assert any("RAG Benchmark Lab" in md.value for md in app.markdown)
