"""
Caching utilities for embeddings, documents, and LLM responses.
Uses an in-memory cache with TTL. For production, integrate with Redis.
"""

import hashlib
import time
from typing import Any, Optional


class SimpleCache:
    """In-memory cache with TTL."""

    def __init__(self, max_size: int = 10000, default_ttl_s: int = 3600):
        self.cache: dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl_s = default_ttl_s

    def _make_key(self, key: str) -> str:
        """Hash key for consistent lookups."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        hashed_key = self._make_key(key)
        if hashed_key not in self.cache:
            return None

        value, expiry = self.cache[hashed_key]
        if time.time() > expiry:
            del self.cache[hashed_key]
            return None

        return value

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        """Set value with optional TTL."""
        hashed_key = self._make_key(key)
        ttl = ttl_s or self.default_ttl_s
        expiry = time.time() + ttl

        # Simple eviction: clear half cache if full
        if len(self.cache) >= self.max_size:
            keys_to_remove = list(self.cache.keys())[: self.max_size // 2]
            for k in keys_to_remove:
                del self.cache[k]

        self.cache[hashed_key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()


class EmbeddingCache(SimpleCache):
    """Specialized cache for embeddings."""

    def get_embedding(self, text: str, model: str) -> Optional[list[float]]:
        key = f"emb:{model}:{text[:100]}"
        return self.get(key)

    def set_embedding(
        self, text: str, model: str, embedding: list[float], ttl_s: int = 86400
    ) -> None:
        key = f"emb:{model}:{text[:100]}"
        self.set(key, embedding, ttl_s=ttl_s)


class ResponseCache(SimpleCache):
    """Specialized cache for LLM responses."""

    def get_response(self, prompt: str, model: str) -> Optional[str]:
        key = f"resp:{model}:{hashlib.md5(prompt.encode()).hexdigest()}"
        return self.get(key)

    def set_response(
        self, prompt: str, model: str, response: str, ttl_s: int = 86400
    ) -> None:
        key = f"resp:{model}:{hashlib.md5(prompt.encode()).hexdigest()}"
        self.set(key, response, ttl_s=ttl_s)


# Global cache instances
embedding_cache = EmbeddingCache(max_size=5000)
response_cache = ResponseCache(max_size=1000)
