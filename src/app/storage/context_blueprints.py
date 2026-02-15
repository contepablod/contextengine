from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.core.config import settings
from app.core.utils.helpers import get_embedding

logger = logging.getLogger(__name__)


def load_context_blueprints(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Failed to load context blueprints from %s: %s", path, exc)
        return []


def seed_context_blueprints(
    client: Any,
    pinecone_index: Any,
    namespace: str,
    path: str | None = None,
) -> int:
    if pinecone_index is None:
        return 0
    if client is None:
        logger.warning("OpenAI client missing; skipping context blueprint seed.")
        return 0

    path = path or settings.context_blueprint_path
    entries = load_context_blueprints(path)
    if not entries:
        return 0

    vectors: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blueprint_id = (entry.get("id") or "").strip()
        description = (entry.get("description") or "").strip()
        blueprint = entry.get("blueprint") or {}
        if not blueprint_id or not description or not isinstance(blueprint, dict):
            continue

        emb = get_embedding(client, description, settings.embedding_model)
        blueprint_json = json.dumps(blueprint, ensure_ascii=True)
        meta: dict[str, Any] = {
            "description": description,
            "blueprint": blueprint_json,
        }

        vectors.append({"id": blueprint_id, "values": emb, "metadata": meta})

    if not vectors:
        return 0

    pinecone_index.upsert(vectors=vectors, namespace=namespace)
    return len(vectors)
