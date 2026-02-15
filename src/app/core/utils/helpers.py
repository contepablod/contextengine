from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import tiktoken
from openai import OpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.observability.metrics import (
    LLM_ERRORS,
    LLM_REQUESTS,
    LLM_REQUEST_DURATION,
    LLM_TOKENS,
)


logger = logging.getLogger(__name__)


# -----------------------------
# Untrusted-content boxing
# -----------------------------


def _box_untrusted(text: str) -> str:
    """
    Put untrusted content in a delimited block; models should treat it as data only.
    Use this when injecting retrieved chunks / user-uploaded doc snippets.
    """
    return f"<UNTRUSTED_DATA>\n{text}\n</UNTRUSTED_DATA>"


# -----------------------------
# Prompt-injection tripwires (cheap filter)
# -----------------------------

INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"system prompt",
    r"developer message",
    r"reveal (the )?prompt",
    r"exfiltrate",
    r"do not follow",
    r"jailbreak",
    r"you are now",
    r"BEGIN SYSTEM PROMPT",
    r"###\s*SYSTEM",
]
_injection_regex = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_untrusted_text(text: str) -> tuple[str, list[str]]:
    """
    Returns (text, flags).
    We do NOT "clean" content (lossy/dangerous); we only flag suspicious chunks.
    """
    flags: list[str] = []
    if _injection_regex.search(text or ""):
        flags.append("possible_prompt_injection")
    return text, flags


def clamp_str(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[TRUNCATED]"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def hash_password(password: str, *, iterations: int = 200_000, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64encode(salt)}${_b64encode(dk)}"


def verify_password(password: str, encoded: str) -> bool:
    if not password or not encoded:
        return False
    try:
        scheme, iter_str, salt_b64, hash_b64 = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iter_str)
        salt = _b64decode(salt_b64)
        expected = _b64decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def count_tokens(text: str, model: str | None = None) -> int:
    model = model or settings.generation_model
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def _record_llm_metrics(
    *,
    model: str,
    operation: str,
    start_s: float,
    resp: Any | None = None,
    error: Exception | None = None,
    prompt_text: str | None = None,
    completion_text: str | None = None,
) -> None:
    duration = max(time.perf_counter() - start_s, 0.0)
    LLM_REQUEST_DURATION.labels(model, operation).observe(duration)
    status = "success" if error is None else "error"
    LLM_REQUESTS.labels(model, operation, status).inc()
    if error is not None:
        LLM_ERRORS.labels(model, operation, type(error).__name__).inc()
        return

    usage = getattr(resp, "usage", None)
    if usage:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if prompt is not None:
            LLM_TOKENS.labels(model, "prompt").inc(int(prompt))
        if completion is not None:
            LLM_TOKENS.labels(model, "completion").inc(int(completion))
        if total is not None:
            LLM_TOKENS.labels(model, "total").inc(int(total))
        return

    if prompt_text is None and completion_text is None:
        return
    prompt_tokens = count_tokens(prompt_text or "", model)
    completion_tokens = count_tokens(completion_text or "", model)
    if prompt_tokens:
        LLM_TOKENS.labels(model, "prompt").inc(int(prompt_tokens))
    if completion_tokens:
        LLM_TOKENS.labels(model, "completion").inc(int(completion_tokens))
    if prompt_tokens or completion_tokens:
        LLM_TOKENS.labels(model, "total").inc(int(prompt_tokens + completion_tokens))


def make_openai_client() -> OpenAI:
    """
    Supports both OpenAI and OpenRouter-style usage.
    - OPENAI_API_KEY is required
    - OPENAI_BASE_URL optional (e.g. https://openrouter.ai/api/v1)
    """
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


# -----------------------------
# Lightweight lexical similarity
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_for_overlap(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 3}


def lexical_overlap_score(query: str, text: str) -> float:
    """
    Simple lexical overlap score in [0,1].
    Useful as a cheap heuristic rerank signal or debugging check.

    score = |tokens(query) ∩ tokens(text)| / |tokens(query)|
    """
    q = _tokenize_for_overlap(query)
    if not q:
        return 0.0
    t = _tokenize_for_overlap(text)
    return len(q.intersection(t)) / float(len(q))


def tokenize_for_bm25(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 3]


def load_bm25_stats(corpus_dir: str) -> dict[str, Any] | None:
    path = os.path.join(corpus_dir, "bm25_stats.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        logger.warning("Failed to load BM25 stats from %s", path)
        return None


def build_bm25_query_vector(
    query: str, stats: dict[str, Any]
) -> dict[str, list[float] | list[int]]:
    vocab = stats.get("vocab") or {}
    idf = stats.get("idf") or {}
    if not vocab or not idf:
        return {"indices": [], "values": []}

    counts = Counter(tokenize_for_bm25(query))
    if not counts:
        return {"indices": [], "values": []}

    indices: list[int] = []
    values: list[float] = []
    for term, tf in counts.items():
        idx = vocab.get(term)
        term_idf = idf.get(term)
        if idx is None or term_idf is None:
            continue
        indices.append(int(idx))
        values.append(float(tf) * float(term_idf))

    if not indices:
        return {"indices": [], "values": []}

    order = sorted(range(len(indices)), key=lambda i: indices[i])
    return {
        "indices": [indices[i] for i in order],
        "values": [values[i] for i in order],
    }


# -----------------------------
# LLM + embeddings
# -----------------------------


@retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=4)
)
def call_chat_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    *,
    temperature: float = 0.2,
    response_format: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> str:
    timeout_s = timeout_s or settings.request_timeout_s

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        start_s = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        _record_llm_metrics(
            model=model,
            operation="chat",
            start_s=start_s,
            resp=resp,
            prompt_text=f"{system}\n{user}",
            completion_text=content,
        )
        return content
    except BadRequestError as exc:
        _record_llm_metrics(
            model=model, operation="chat", start_s=start_s, error=exc
        )
        msg = str(exc).lower()
        if response_format is not None and "response_format" in msg and "not supported" in msg:
            logger.warning(
                "response_format unsupported by model=%s; retrying without it",
                model,
            )
            kwargs.pop("response_format", None)
            start_s = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout_s,
                    **kwargs,
                )
                content = resp.choices[0].message.content or ""
                _record_llm_metrics(
                    model=model,
                    operation="chat",
                    start_s=start_s,
                    resp=resp,
                    prompt_text=f"{system}\n{user}",
                    completion_text=content,
                )
            except Exception as exc_retry:
                _record_llm_metrics(
                    model=model,
                    operation="chat",
                    start_s=start_s,
                    error=exc_retry,
                )
                raise
            return content
        raise
    except Exception as exc:
        _record_llm_metrics(
            model=model, operation="chat", start_s=start_s, error=exc
        )
        raise


@retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=4)
)
def get_embedding(client: OpenAI, text: str, model: str) -> list[float]:
    text = (text or "").replace("\n", " ")
    start_s = time.perf_counter()
    try:
        resp = client.embeddings.create(
            model=model,
            input=text,
            timeout=settings.request_timeout_s,
        )
        _record_llm_metrics(
            model=model,
            operation="embedding",
            start_s=start_s,
            resp=resp,
            prompt_text=text,
        )
    except Exception as exc:
        _record_llm_metrics(
            model=model, operation="embedding", start_s=start_s, error=exc
        )
        raise
    return resp.data[0].embedding


# -----------------------------
# JSON helpers
# -----------------------------


def safe_json_loads(s: str) -> Any:
    """
    Robust JSON parser:
    - strips whitespace
    - removes ``` fenced blocks if the model returned them
    - tries strict json.loads
    """
    s = (s or "").strip()

    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    return json.loads(s)


# -----------------------------
# Evidence item (shared struct)
# -----------------------------


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    source: str
    score: float
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


# -----------------------------
# Moderation
# -----------------------------

_CATS = [
    "hate",
    "hate_threatening",
    "harassment",
    "harassment_threatening",
    "self_harm",
    "self_harm_intent",
    "self_harm_instructions",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
]


def _empty_categories() -> dict[str, bool]:
    return {c: False for c in _CATS}


def _empty_scores() -> dict[str, float]:
    return {c: 0.0 for c in _CATS}


def _map_reasons_to_categories(reasons: list[Any]) -> dict[str, bool]:
    cats = _empty_categories()
    for reason in reasons or []:
        r = str(reason).lower()
        if "violence" in r:
            cats["violence"] = True
        elif "hate" in r:
            cats["hate"] = True
        elif "sexual" in r:
            cats["sexual"] = True
        elif "harass" in r:
            cats["harassment"] = True
        elif ("self" in r) and ("harm" in r):
            cats["self_harm"] = True
    return cats


@retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=4)
)
def create_moderation_response(
    text_to_moderate: str, client: OpenAI
) -> SimpleNamespace:
    """
    Return an OpenAI-moderation-like object:
      resp.model (str)
      resp.results[0].flagged (bool)
      resp.results[0].categories (dict[str,bool])
      resp.results[0].category_scores (dict[str,float])

    Uses provider built-in moderation behavior by making a minimal chat request.
    """
    model_name = os.getenv("MODERATION_MODEL", settings.moderation_model)

    try:
        start_s = time.perf_counter()
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": text_to_moderate}],
            max_tokens=1,
            timeout=settings.request_timeout_s,
        )
        _record_llm_metrics(
            model=model_name,
            operation="moderation",
            start_s=start_s,
            resp=None,
            prompt_text=text_to_moderate,
        )
        r0 = SimpleNamespace(
            flagged=False,
            categories=_empty_categories(),
            category_scores=_empty_scores(),
        )
        return SimpleNamespace(model=model_name, results=[r0])

    except Exception as e:
        _record_llm_metrics(
            model=model_name,
            operation="moderation",
            start_s=start_s,
            error=e,
            prompt_text=text_to_moderate,
        )
        status_code = getattr(e, "status_code", None)
        response = getattr(e, "response", None)

        if status_code == 403 and response is not None:
            error_data: dict[str, Any] = {}
            try:
                error_data = response.json()
            except Exception:
                error_data = {}

            reasons = error_data.get("error", {}).get("metadata", {}).get("reasons", [])

            cats = _map_reasons_to_categories(reasons)
            scores = {k: (1.0 if v else 0.0) for k, v in cats.items()}
            r0 = SimpleNamespace(flagged=True, categories=cats, category_scores=scores)
            return SimpleNamespace(model=model_name, results=[r0])

        raise


def moderate_text(client: OpenAI, text: str) -> dict[str, Any]:
    """Moderate and return a compact report."""
    text = clamp_str(text or "", 20000)
    resp = create_moderation_response(text, client)
    r0 = resp.results[0]

    categories = (
        r0.categories if isinstance(r0.categories, dict) else vars(r0.categories)
    )
    scores = (
        r0.category_scores
        if isinstance(r0.category_scores, dict)
        else vars(r0.category_scores)
    )

    return {
        "flagged": bool(getattr(r0, "flagged", False)),
        "categories": categories,
        "category_scores": scores,
        "model": getattr(resp, "model", settings.moderation_model),
    }
