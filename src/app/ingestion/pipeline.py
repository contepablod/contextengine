from __future__ import annotations

import logging
import os
import time
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import clamp_str, get_embedding
from app.observability.metrics import (
    VECTOR_DB_REQUESTS,
    VECTOR_DB_REQUEST_DURATION,
)
from .bm25 import load_corpus, reindex_bm25, save_corpus
from .chunking import chunk_profile_for, chunk_text
from .extractors import extract_pdf_to_sections
from .metadata import detect_doc_type
from .utils import batch_items, stable_doc_id_from_bytes

logger = logging.getLogger(__name__)


def ingest_pdf_to_pinecone(
    client: OpenAI,
    pinecone_index: Any,
    pdf_path: str,
    namespace: str = None,
    doc_id: str = None,
    chunk_chars: int = None,
    overlap_chars: int = None,
    max_pages: int = None,
    metadata_extra: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Extracts PDF -> chunks -> embeddings -> upsert to Pinecone.

    Returns:
      {doc_id, filename, namespace, chunks_upserted, pages, elapsed_s}
    """
    if pinecone_index is None:
        raise RuntimeError("Pinecone index not configured; cannot ingest.")

    namespace = namespace or settings.namespace_knowledge
    chunk_chars = int(chunk_chars) if chunk_chars is not None else None
    overlap_chars = int(overlap_chars) if overlap_chars is not None else None

    filename = os.path.basename(pdf_path)

    t0 = time.time()

    with open(pdf_path, "rb") as f:
        data = f.read()

    stable = stable_doc_id_from_bytes(data)
    doc_id = doc_id or stable

    blocks, extraction_method = extract_pdf_to_sections(pdf_path, max_pages=max_pages)
    doc_type = detect_doc_type(blocks)
    profile = chunk_profile_for(doc_type)
    if chunk_chars is None:
        chunk_chars = profile["chunk_chars"]
    if overlap_chars is None:
        overlap_chars = profile["overlap_chars"]
    chunks = chunk_text(
        doc_id=doc_id,
        filename=filename,
        blocks=blocks,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )

    vectors: list[dict[str, Any]] = []
    corpus_records: list[dict[str, Any]] = []
    for ch in chunks:
        emb = get_embedding(client, ch.text, settings.embedding_model)
        meta: dict[str, Any] = {
            "doc_id": ch.doc_id,
            "filename": ch.filename,
            "chunk_id": ch.chunk_id,
            "page": ch.page_start,
            "page_start": ch.page_start,
            "page_end": ch.page_end,
            "section": ch.section or "",
            "char_start": ch.char_start,
            "char_end": ch.char_end,
            "doc_type": doc_type,
            # Keep the raw chunk text in metadata for retrieval display
            "text": clamp_str(ch.text, 15000),
        }
        if metadata_extra:
            meta.update(metadata_extra)
        vec: dict[str, Any] = {"id": ch.chunk_id, "values": emb, "metadata": meta}
        vectors.append(vec)
        corpus_records.append(
            {
                "id": ch.chunk_id,
                "embedding": emb,
                "metadata": meta,
                "text": ch.text,
            }
        )

    if settings.enable_bm25_lexical:
        existing = load_corpus()
        existing = [
            r for r in existing if (r.get("metadata") or {}).get("doc_id") != doc_id
        ]
        all_records = existing + corpus_records
        save_corpus(all_records)
        upserted = reindex_bm25(pinecone_index, namespace, all_records)
    else:
        upserted = 0
        for b in batch_items(vectors, settings.pinecone_upsert_batch_size):
            start_s = time.perf_counter()
            try:
                pinecone_index.upsert(vectors=b, namespace=namespace)
                VECTOR_DB_REQUESTS.labels("upsert", "success").inc()
            except Exception as exc:
                VECTOR_DB_REQUESTS.labels("upsert", "error").inc()
                VECTOR_DB_REQUEST_DURATION.labels("upsert").observe(
                    max(time.perf_counter() - start_s, 0.0)
                )
                raise
            VECTOR_DB_REQUEST_DURATION.labels("upsert").observe(
                max(time.perf_counter() - start_s, 0.0)
            )
            upserted += len(b)

    elapsed = time.time() - t0
    page_count = len({b.get("page") for b in blocks if b.get("page")})
    logger.info(
        "Ingested PDF doc_id=%s file=%s pages=%d chunks=%d ns=%s doc_type=%s elapsed=%.2fs",
        doc_id,
        filename,
        page_count,
        upserted,
        namespace,
        doc_type,
        elapsed,
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "namespace": namespace,
        "chunks_upserted": upserted,
        "pages": page_count,
        "doc_type": doc_type,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "extraction_method": extraction_method,
        "elapsed_s": elapsed,
    }
