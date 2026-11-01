"""Fixtures that build real PDF bytes so parser tests do not depend on binary files."""

from collections.abc import Callable

import pytest
from fpdf import FPDF


def _build_pdf(pages: list[str], title: str | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    if title:
        pdf.set_title(title)
        pdf.set_author("RAG Lab")
    for page_text in pages:
        pdf.add_page()
        if page_text:
            pdf.multi_cell(0, 10, page_text)
    return bytes(pdf.output())


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    return _build_pdf
