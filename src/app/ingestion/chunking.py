from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.utils.helpers import sanitize_untrusted_text
from .models import IngestChunk
from .utils import normalize_ws, sha1_text


def chunk_text(
    doc_id: str,
    filename: str,
    blocks: list[dict[str, Any]],
    chunk_chars: int,
    overlap_chars: int,
) -> list[IngestChunk]:
    """
    Chunk contiguous text while preserving page/section metadata.
    Chunks will not span more than a few pages depending on chunk size.
    """
    chunks: list[IngestChunk] = []
    buf: list[str] = []
    buf_len = 0

    # metadata for current buffer window
    page_start: int = None
    page_end: int = None
    section: str = None
    char_cursor = 0  # global cursor (approx)

    def flush(final: bool = False):
        nonlocal buf, buf_len, page_start, page_end, section, char_cursor, chunks
        if not buf:
            return
        text = normalize_ws("\n\n".join(buf))
        if not text:
            return

        # safety: tripwire suspicious chunks
        text2, flags = sanitize_untrusted_text(text)
        if "possible_prompt_injection" in flags:
            # drop instead of trying to "clean"
            buf, buf_len = [], 0
            page_start = page_end = None
            section = section  # keep section context
            return
        text = text2

        chunk_hash = sha1_text(
            f"{doc_id}|{page_start}|{page_end}|{section}|{text[:2000]}"
        )
        chunk_id = f"{doc_id}-{chunk_hash[:12]}"

        start = max(0, char_cursor - len(text))
        end = start + len(text)

        chunks.append(
            IngestChunk(
                doc_id=doc_id,
                filename=filename,
                chunk_id=chunk_id,
                text=text,
                page_start=int(page_start or 1),
                page_end=int(page_end or int(page_start or 1)),
                section=section,
                char_start=start,
                char_end=end,
            )
        )

        # prepare overlap: keep last overlap_chars of current text
        if not final and overlap_chars > 0:
            tail = text[-overlap_chars:]
            buf = [tail]
            buf_len = len(tail)
        else:
            buf = []
            buf_len = 0

        page_start = None
        page_end = None
        # keep section as last-known label

    for b in blocks:
        p = int(b["page"])
        sec = b.get("section") or section
        text = b.get("text") or ""
        block_type = (b.get("block_type") or "para").lower()
        text = normalize_ws(text)

        if not text:
            continue

        if section and sec and sec != section and buf:
            flush(final=False)

        if block_type in {"table", "footnote", "reference", "clause"} and buf:
            flush(final=False)

        # set metadata window
        if page_start is None:
            page_start = p
        page_end = p
        section = sec

        if block_type in {"table", "footnote", "reference", "clause"}:
            if len(text) <= chunk_chars:
                buf = [text]
                buf_len = len(text)
                char_cursor += len(text)
                flush(final=True)
                continue

        # append with a separator
        addition = text if not buf else "\n\n" + text
        add_len = len(addition)

        if buf_len + add_len <= chunk_chars:
            buf.append(text)
            buf_len += add_len
            char_cursor += add_len
        else:
            # flush current buffer first, then start new
            flush(final=False)
            # after flush, we need to start with this block
            if page_start is None:
                page_start = p
            page_end = p
            section = sec

            # if the block itself is huge, slice it
            if len(text) > chunk_chars:
                start_idx = 0
                while start_idx < len(text):
                    piece = text[start_idx : start_idx + chunk_chars]
                    page_start = p
                    page_end = p
                    section = sec
                    buf = [piece]
                    buf_len = len(piece)
                    char_cursor += len(piece)
                    flush(final=False)
                    start_idx += max(1, chunk_chars - overlap_chars)
            else:
                buf = [text]
                buf_len = len(text)
                char_cursor += len(text)

    flush(final=True)
    return chunks


def chunk_profile_for(doc_type: str) -> dict[str, int]:
    default = {
        "chunk_chars": int(settings.chunk_chars),
        "overlap_chars": int(settings.chunk_overlap_chars),
    }
    profiles = {
        "scholarly": {"chunk_chars": 3200, "overlap_chars": 320},
        "financial": {"chunk_chars": 2400, "overlap_chars": 240},
        "legal": {"chunk_chars": 1600, "overlap_chars": 120},
        "scan": {"chunk_chars": 1400, "overlap_chars": 200},
    }
    prof = profiles.get(doc_type)
    if not prof:
        return default
    merged = default.copy()
    merged.update(prof)
    return merged
