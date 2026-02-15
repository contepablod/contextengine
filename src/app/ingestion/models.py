from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestChunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    char_start: int
    char_end: int
