from __future__ import annotations

import re
from typing import Any


_SECTION_REF_RE = re.compile(r"\\bsection\\s+\\d+(\\.\\d+)*\\b", re.IGNORECASE)


def sample_text(blocks: list[dict[str, Any]], max_chars: int = 50000) -> str:
    parts: list[str] = []
    total = 0
    for b in blocks:
        text = b.get("text") or ""
        section = b.get("section") or ""
        if text:
            parts.append(text)
            total += len(text)
        if section:
            parts.append(section)
            total += len(section)
        if total >= max_chars:
            break
    return "\\n".join(parts)[:max_chars]


def detect_doc_type(blocks: list[dict[str, Any]]) -> str:
    if not blocks:
        return "generic"

    pages = {b.get("page") for b in blocks if b.get("page")}
    page_count = len(pages) or 1
    total_chars = sum(len(b.get("text") or "") for b in blocks)
    avg_chars = total_chars / max(1, page_count)
    if page_count >= 3 and avg_chars < 450:
        return "scan"

    sample = sample_text(blocks).lower()

    scores = {"scholarly": 0, "financial": 0, "legal": 0}

    if "abstract" in sample:
        scores["scholarly"] += 2
    if "references" in sample or "bibliography" in sample:
        scores["scholarly"] += 2
    if "arxiv" in sample or "doi" in sample:
        scores["scholarly"] += 2
    if re.search(r"\\b(?:introduction|methods?|results?|discussion|conclusion)\\b", sample):
        scores["scholarly"] += 1

    if "consolidated financial statements" in sample:
        scores["financial"] += 3
    if "balance sheet" in sample:
        scores["financial"] += 2
    if "income statement" in sample or "statement of income" in sample:
        scores["financial"] += 2
    if "cash flow" in sample or "cash flows" in sample:
        scores["financial"] += 2
    if "notes to the financial statements" in sample:
        scores["financial"] += 2
    if "fair value" in sample:
        scores["financial"] += 1
    if "assets" in sample and "liabilit" in sample:
        scores["financial"] += 1

    if "agreement" in sample:
        scores["legal"] += 2
    if "governing law" in sample:
        scores["legal"] += 2
    if "indemnif" in sample:
        scores["legal"] += 2
    if "whereas" in sample:
        scores["legal"] += 1
    if _SECTION_REF_RE.search(sample):
        scores["legal"] += 2
    if re.search(r"\\b\\d+(\\.\\d+)+\\b", sample):
        scores["legal"] += 1

    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return "generic"
    return best
