from __future__ import annotations

import logging
from typing import Any, Iterable

from pinecone import Pinecone


logger = logging.getLogger(__name__)


def describe_index(pinecone_index: Any) -> tuple[set[str], int | None]:
    try:
        stats = pinecone_index.describe_index_stats()
    except Exception as exc:
        logger.warning("Pinecone stats unavailable: %s", exc)
        return set(), None

    if isinstance(stats, dict):
        namespaces = stats.get("namespaces") or {}
        dimension = stats.get("dimension")
    else:
        namespaces = getattr(stats, "namespaces", {}) or {}
        dimension = getattr(stats, "dimension", None)
    return set(namespaces.keys()), dimension


def ensure_namespaces(pinecone_index: Any, namespaces: Iterable[str]) -> None:
    existing, dimension = describe_index(pinecone_index)
    if dimension is None:
        logger.warning("Pinecone dimension unknown; namespace creation skipped")
        return

    for ns in namespaces:
        if not ns or ns in existing:
            continue
        try:
            dummy_id = "__namespace_init__"
            zero_vec = [0.0] * int(dimension)
            zero_vec[0] = 1e-6  # Pinecone rejects all-zero vectors
            pinecone_index.upsert(
                vectors=[(dummy_id, zero_vec, {"_system": True})],
                namespace=ns,
            )
            pinecone_index.delete(ids=[dummy_id], namespace=ns)
            logger.info("Initialized Pinecone namespace=%s", ns)
        except Exception as exc:
            logger.warning("Failed to init Pinecone namespace %s: %s", ns, exc)


def init_index(api_key: str, index_name: str) -> Any | None:
    if not api_key or not index_name:
        return None
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)
