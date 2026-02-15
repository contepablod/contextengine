from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv(override=False)

_BASE_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))


def _get_env(name: str, default: str = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str = _get_env("OPENAI_API_KEY", "")
    openai_base_url: str = _get_env("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    generation_model: str = _get_env(
        "GENERATION_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    planning_model: str = _get_env("PLANNING_MODEL", "xiaomi/mimo-v2-flash:free")
    embedding_model: str = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    moderation_model: str = _get_env(
        "MODERATION_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free"
    )

    # Pinecone
    pinecone_api_key: str = _get_env("PINECONE_API_KEY", "")
    pinecone_index: str = _get_env("PINECONE_INDEX", "genai-mas-mcp-ch3")
    # Namespaces
    namespace_context: str = _get_env(
        "PINECONE_NAMESPACE_CONTEXT", "ContextLibrary"
    )
    namespace_knowledge: str = _get_env(
        "PINECONE_NAMESPACE_KNOWLEDGE", "KnowledgeStore"
    )

    # Execution bounds
    max_steps: int = int(_get_env("MAX_STEPS", "6"))
    max_input_chars: int = int(_get_env("MAX_INPUT_CHARS", "12000"))
    max_context_chars: int = int(
        _get_env("MAX_CONTEXT_CHARS", "40000")
    )  # retrieved chunks total
    max_output_chars: int = int(_get_env("MAX_OUTPUT_CHARS", "20000"))
    max_tokens_per_call: int = int(_get_env("MAX_TOKENS_PER_CALL", "1500"))

    # API behavior
    request_timeout_s: float = float(_get_env("REQUEST_TIMEOUT_S", "30"))
    enable_input_moderation: bool = (
        _get_env("ENABLE_INPUT_MODERATION", "false").lower() == "true"
    )

    # Auth
    enable_auth: bool = _get_env("ENABLE_AUTH", "true").lower() == "true"
    auth_username: str = _get_env("AUTH_USERNAME", "")
    auth_password_hash: str = _get_env("AUTH_PASSWORD_HASH", "")
    auth_cookie_name: str = _get_env("AUTH_COOKIE_NAME", "contextengine_session")
    auth_cookie_secure: bool = _get_env("AUTH_COOKIE_SECURE", "false").lower() == "true"
    auth_session_ttl_s: int = int(_get_env("AUTH_SESSION_TTL_S", "86400"))

    # Retrieval + chunking
    chunk_chars: int = int(_get_env("CHUNK_CHARS", "1800"))
    chunk_overlap_chars: int = int(_get_env("CHUNK_OVERLAP_CHARS", "200"))
    rerank_top_n: int = int(_get_env("RERANK_TOP_N", "8"))
    enable_llm_rerank: bool = (
        _get_env("ENABLE_LLM_RERANK", "false").lower() == "true"
    )
    reranker_model: str = _get_env(
        "RERANKER_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free"
    )
    doc_top_k: int = int(_get_env("DOC_TOP_K", "8"))
    enable_bm25_lexical: bool = (
        _get_env("ENABLE_BM25_LEXICAL", "false").lower() == "true"
    )
    bm25_k1: float = float(_get_env("BM25_K1", "1.2"))
    bm25_b: float = float(_get_env("BM25_B", "0.75"))
    lexical_overlap_weight: float = float(_get_env("LEXICAL_OVERLAP_WEIGHT", "0.2"))
    corpus_dir: str = _get_env(
        "CORPUS_DIR", os.path.join(_PROJECT_ROOT, "uploads", "bm25")
    )
    pinecone_upsert_batch_size: int = int(
        _get_env("PINECONE_UPSERT_BATCH_SIZE", "100")
    )

    # Rate limiting (simple, in-memory)
    rate_limit_per_minute: int = int(_get_env("RATE_LIMIT_PER_MINUTE", "60"))

    # Context blueprints
    context_blueprint_path: str = _get_env(
        "CONTEXT_BLUEPRINT_PATH",
        os.path.join(_PROJECT_ROOT, "context.json"),
    )
    seed_context_blueprints: bool = (
        _get_env("SEED_CONTEXT_BLUEPRINTS", "true").lower() == "true"
    )

    # Observability
    enable_metrics: bool = _get_env("ENABLE_METRICS", "true").lower() == "true"
    metrics_path: str = _get_env("METRICS_PATH", "/metrics")
    enable_tracing: bool = _get_env("ENABLE_TRACING", "false").lower() == "true"
    otel_exporter_otlp_endpoint: str = _get_env(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    )
    otel_exporter_otlp_headers: str = _get_env("OTEL_EXPORTER_OTLP_HEADERS", "")
    otel_service_name: str = _get_env("OTEL_SERVICE_NAME", "contextengine")

    # PDF extraction
    # PyMuPDF only (GROBID removed)


settings = Settings()
