from __future__ import annotations

import logging
import os
import re
from typing import Any

import fitz  # PyMuPDF
from .utils import normalize_ws

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(?:\\d+(\\.\\d+)*\\s+)?[A-Z][^\\n]{2,120}$")
_CLAUSE_RE = re.compile(r"^(?:section\\s+)?\\d+(\\.\\d+)+", re.IGNORECASE)
_TABLE_SPACE_RE = re.compile(r"\\s{2,}")


def extract_pdf_pages(
    pdf_path: str,
    max_pages: int = None,
) -> list[dict[str, Any]]:
    """
    Extract per-page text and lightweight structural hints.
    Returns list[{page:int, text:str, headings:[str]}]
    """
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
        limit = min(n, max_pages) if max_pages else n

        for i in range(limit):
            page = doc.load_page(i)
            # "text" mode is generally stable; arXiv PDFs vary a lot
            text = page.get_text("text") or ""
            text = normalize_ws(text)

            headings: list[str] = []
            # naive heading detection: short-ish lines with heading-like shape
            for line in text.splitlines():
                ln = line.strip()
                if 4 <= len(ln) <= 120 and _HEADING_RE.match(ln):
                    headings.append(ln)

            pages.append({"page": i + 1, "text": text, "headings": headings})

    return pages


def _looks_like_table(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    spaced = sum(1 for ln in lines if _TABLE_SPACE_RE.search(ln))
    digits = sum(1 for ln in lines if re.search(r"\\d", ln))
    if spaced / len(lines) >= 0.6 and digits / len(lines) >= 0.6:
        return True
    return False


def _classify_block(text: str) -> str:
    if not text:
        return "para"
    if _CLAUSE_RE.match(text):
        return "clause"
    if _looks_like_table(text):
        return "table"
    if len(text) < 160 and re.match(r"^\\d+\\s", text):
        return "footnote"
    return "para"


def extract_pdf_to_sections(
    pdf_path: str,
    max_pages: int = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Produces a sequence of "blocks" that carry {page, section, text}.
    We keep it simple: page text with latest heading as section label.
    Returns (blocks, extraction_method).
    """
    pages = extract_pdf_pages(pdf_path, max_pages=max_pages)
    blocks: list[dict[str, Any]] = []
    current_section: str = None

    for p in pages:
        page_no = int(p["page"])
        text = p["text"]
        headings = p["headings"]

        # choose a "dominant" heading for the page if any
        if headings:
            # pick the first heading-like line on the page
            current_section = headings[0]

        paragraphs = [s.strip() for s in text.split("\\n\\n") if s.strip()]
        if not paragraphs:
            continue

        for para in paragraphs:
            block_type = _classify_block(para)
            blocks.append(
                {
                    "page": page_no,
                    "section": current_section,
                    "text": para,
                    "block_type": block_type,
                }
            )

    return blocks, "pymupdf"
