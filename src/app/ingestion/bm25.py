from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from typing import Any

from app.core.config import settings
from app.core.utils.helpers import tokenize_for_bm25
from app.observability.metrics import (
    VECTOR_DB_REQUESTS,
    VECTOR_DB_REQUEST_DURATION,
)
from .utils import batch_items, ensure_dir

logger = logging.getLogger(__name__)


def _corpus_dir() -> str:
    return ensure_dir(settings.corpus_dir)


def corpus_path() -> str:
    return os.path.join(_corpus_dir(), "bm25_corpus.json")


def stats_path() -> str:
    return os.path.join(_corpus_dir(), "bm25_stats.json")


def load_corpus() -> list[dict[str, Any]]:
    path = corpus_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        logger.warning("Failed to load BM25 corpus from %s", path)
    return []


def save_corpus(records: list[dict[str, Any]]) -> None:
    path = corpus_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=True)


def save_stats(stats: dict[str, Any]) -> None:
    path = stats_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True)


def _build_bm25_stats(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Counter[str], int]]]:
    df: dict[str, int] = {}
    tokenized: list[tuple[dict[str, Any], Counter[str], int]] = []
    total_len = 0

    for r in records:
        tokens = tokenize_for_bm25(r.get("text", ""))
        tf = Counter(tokens)
        doc_len = sum(tf.values())
        total_len += doc_len
        tokenized.append((r, tf, doc_len))
        for term in tf.keys():
            df[term] = df.get(term, 0) + 1

    doc_count = len(records)
    avgdl = (total_len / doc_count) if doc_count else 0.0

    vocab = {term: i for i, term in enumerate(sorted(df.keys()))}
    idf: dict[str, float] = {}
    for term, dfi in df.items():
        idf[term] = math.log(1.0 + (doc_count - dfi + 0.5) / (dfi + 0.5))

    stats = {
        "doc_count": doc_count,
        "avgdl": avgdl,
        "vocab": vocab,
        "idf": idf,
    }
    return stats, tokenized


def _bm25_sparse_for_doc(
    tf: Counter[str],
    doc_len: int,
    avgdl: float,
    idf: dict[str, float],
    vocab: dict[str, int],
    k1: float,
    b: float,
) -> dict[str, list[float] | list[int]]:
    if not tf or not vocab or doc_len <= 0:
        return {"indices": [], "values": []}

    denom_base = k1 * (1.0 - b + b * (float(doc_len) / avgdl)) if avgdl else k1
    indices: list[int] = []
    values: list[float] = []
    for term, freq in tf.items():
        term_idf = idf.get(term)
        idx = vocab.get(term)
        if term_idf is None or idx is None:
            continue
        denom = float(freq) + denom_base
        weight = float(term_idf) * ((float(freq) * (k1 + 1.0)) / denom)
        if weight <= 0.0:
            continue
        indices.append(int(idx))
        values.append(float(weight))

    if not indices:
        return {"indices": [], "values": []}

    order = sorted(range(len(indices)), key=lambda i: indices[i])
    return {
        "indices": [indices[i] for i in order],
        "values": [values[i] for i in order],
    }


def reindex_bm25(
    pinecone_index: Any,
    namespace: str,
    records: list[dict[str, Any]],
) -> int:
    stats, tokenized = _build_bm25_stats(records)
    save_stats(stats)

    vocab = stats["vocab"]
    idf = stats["idf"]
    avgdl = float(stats["avgdl"])
    k1 = float(settings.bm25_k1)
    b = float(settings.bm25_b)

    vectors: list[dict[str, Any]] = []
    for r, tf, doc_len in tokenized:
        sparse = _bm25_sparse_for_doc(tf, doc_len, avgdl, idf, vocab, k1, b)
        vec: dict[str, Any] = {
            "id": r["id"],
            "values": r["embedding"],
            "metadata": r["metadata"],
        }
        if sparse["indices"]:
            vec["sparse_values"] = sparse
        vectors.append(vec)

    upserted = 0
    for bch in batch_items(vectors, settings.pinecone_upsert_batch_size):
        start_s = time.perf_counter()
        try:
            pinecone_index.upsert(vectors=bch, namespace=namespace)
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
        upserted += len(bch)
    return upserted
