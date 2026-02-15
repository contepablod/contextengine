"""Wrapper around Pinecone for consistent retrieval logic."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import (
    get_embedding,
    sanitize_untrusted_text,
    clamp_str,
    lexical_overlap_score,
    load_bm25_stats,
    build_bm25_query_vector,
)
from app.observability.metrics import (
    VECTOR_DB_REQUESTS,
    VECTOR_DB_REQUEST_DURATION,
    VECTOR_DB_RESULTS,
)
from .evidence import EvidenceItem

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """Encapsulates Pinecone query logic with metadata handling."""

    def __init__(self, pinecone_index: Any, client: OpenAI):
        """
        Initialize retriever.

        Args:
            pinecone_index: Pinecone index instance
            client: OpenAI client for embeddings
        """
        self.pinecone_index = pinecone_index
        self.client = client

    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = 6,
        doc_id: Optional[str] = None,
        meta_filter: Optional[dict[str, Any]] = None,
        enable_lexical_scoring: bool = True,
    ) -> list[EvidenceItem]:
        """
        Retrieve evidence chunks from Pinecone.

        Args:
            query: Search query string
            namespace: Pinecone namespace to search
            top_k: Number of results to retrieve
            doc_id: Optional filter by document ID
            enable_lexical_scoring: Include lexical overlap in scoring

        Returns:
            List of EvidenceItem objects ranked by relevance
        """
        if self.pinecone_index is None:
            logger.warning("Pinecone index not available, returning empty results")
            return []

        try:
            # Get embedding
            embedding = get_embedding(self.client, query, settings.embedding_model)

            # Build query
            query_kwargs: dict[str, Any] = {
                "namespace": namespace,
                "vector": embedding,
                "top_k": int(top_k),
                "include_metadata": True,
            }

            filter_obj: dict[str, Any] = {}
            if doc_id:
                filter_obj["doc_id"] = {"$eq": doc_id}
            if meta_filter:
                for k, v in meta_filter.items():
                    if k == "doc_id":
                        continue
                    filter_obj[k] = v
            if filter_obj:
                query_kwargs["filter"] = filter_obj

            # Hybrid BM25 sparse vector (if enabled and stats available)
            has_sparse = False
            if settings.enable_bm25_lexical:
                stats = load_bm25_stats(settings.corpus_dir)
                if stats:
                    sparse_vec = build_bm25_query_vector(query, stats)
                    if sparse_vec.get("indices"):
                        query_kwargs["sparse_vector"] = sparse_vec
                        has_sparse = True

            # Execute query
            start_s = time.perf_counter()
            try:
                res = self.pinecone_index.query(**query_kwargs)
                VECTOR_DB_REQUESTS.labels("query", "success").inc()
            except Exception as exc:
                VECTOR_DB_REQUESTS.labels("query", "error").inc()
                VECTOR_DB_REQUEST_DURATION.labels("query").observe(
                    max(time.perf_counter() - start_s, 0.0)
                )
                raise
            VECTOR_DB_REQUEST_DURATION.labels("query").observe(
                max(time.perf_counter() - start_s, 0.0)
            )
            matches = self._extract_matches(res)
            VECTOR_DB_RESULTS.labels("query").observe(len(matches))

            # Convert to EvidenceItem
            candidates = []
            total_chars = 0

            for i, match in enumerate(matches):
                evidence = EvidenceItem.from_pinecone_match(match, i)

                # Safety check: flag suspicious content
                text, flags = sanitize_untrusted_text(evidence.text)
                if "possible_prompt_injection" in flags:
                    logger.warning(f"Suspicious content detected in evidence {evidence.id}")
                    continue

                evidence.text = clamp_str(text, 9000)
                total_chars += len(evidence.text)

                # Stop if we've accumulated enough context
                if total_chars > settings.max_context_chars:
                    break

                # Apply lexical scoring boost if enabled
                if enable_lexical_scoring and not has_sparse:
                    bonus = lexical_overlap_score(query, evidence.text)
                    evidence.score += settings.lexical_overlap_weight * bonus

                candidates.append(evidence)

            # Sort by combined score
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates

        except Exception as e:
            logger.error(f"Failed to retrieve from Pinecone: {e}", exc_info=True)
            raise

    @staticmethod
    def _extract_matches(res: Any) -> list[dict]:
        """Extract matches from Pinecone response (handles both dict and object responses)."""
        if isinstance(res, dict):
            return res.get("matches", []) or []
        return getattr(res, "matches", []) or []
