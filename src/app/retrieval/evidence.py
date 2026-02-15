"""Evidence item dataclass and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvidenceItem:
    """Represents a retrieved evidence chunk with metadata."""

    id: str
    source: str
    score: float
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None

    def __post_init__(self):
        """Ensure snippet defaults to text if not provided."""
        if self.snippet is None:
            self.snippet = self.text

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "score": self.score,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "snippet": self.snippet,
        }

    @classmethod
    def from_pinecone_match(cls, match: dict, index: int) -> EvidenceItem:
        """Create EvidenceItem from Pinecone match result."""
        score = float(match.get("score", 0.0))
        metadata = match.get("metadata", {}) or {}
        text = str(metadata.get("text") or metadata.get("chunk") or "")
        source = (
            metadata.get("filename")
            or metadata.get("source")
            or metadata.get("url")
            or metadata.get("doc_id")
            or "unknown"
        )
        page = metadata.get("page")
        section = metadata.get("section")

        return cls(
            id=f"e{index + 1}",
            source=str(source),
            score=score,
            text=text,
            page_start=int(page) if isinstance(page, (int, float)) else None,
            page_end=int(page) if isinstance(page, (int, float)) else None,
            section=str(section) if section else None,
        )
