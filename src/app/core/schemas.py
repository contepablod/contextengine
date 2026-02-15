from __future__ import annotations

import ast
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, conint, field_validator

from app.core.config import settings


# -----------------------------
# Agent names
# -----------------------------
AgentName = Literal["Librarian", "Researcher", "Summarizer", "Writer", "Verifier"]


# -----------------------------
# Planning schemas
# -----------------------------
class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: conint(ge=1)
    agent: AgentName
    input: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: list[PlanStep] = Field(min_length=1)


# -----------------------------
# Agent input schemas (strict)
# -----------------------------
class LibrarianInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent_query: str = Field(min_length=1, max_length=5000)


class ResearcherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_query: str = Field(min_length=1, max_length=5000)
    top_k: conint(ge=1, le=20) = 6
    doc_id: str | None = Field(default=None, max_length=200)


class SummarizerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_to_summarize: str = Field(min_length=1, max_length=300000)
    max_words: conint(ge=50, le=2000) = 300


def _coerce_to_dict(v: Any) -> dict[str, Any]:
    """
    Accept:
      - dict -> dict
      - JSON string -> dict
      - Python-literal dict string (single quotes) -> dict
    Reject everything else.
    """
    if isinstance(v, dict):
        return v

    if isinstance(v, str):
        s = v.strip()

        # 1) strict JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2) python-literal dict (handles single quotes)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise ValueError(f"Expected dict or JSON-string dict, got: {type(v).__name__}")


class WriterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_json: dict[str, Any]
    facts: dict[str, Any]
    style_notes: str | None = None

    @field_validator("blueprint_json", mode="before")
    @classmethod
    def parse_blueprint(cls, v: Any) -> dict[str, Any]:
        return _coerce_to_dict(v)

    @field_validator("facts", mode="before")
    @classmethod
    def parse_facts(cls, v: Any) -> dict[str, Any]:
        return _coerce_to_dict(v)


class VerifierInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft: str = Field(min_length=1, max_length=200000)
    reference: str = Field(min_length=1, max_length=50000)
    verification_objective: str | None = Field(
        default="Check for factual inaccuracies, unsupported claims, missing citations, and contradictions.",
        max_length=5000,
    )


# -----------------------------
# API schemas (existing)
# -----------------------------
class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=12000)
    namespace_context: str | None = None
    namespace_knowledge: str | None = None


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    output: str
    blocked: bool = False
    moderation: dict[str, Any] | None = None
    trace: dict[str, Any]


# -----------------------------
# Document upload + chat schemas (missing)
# -----------------------------
class UploadResponse(BaseModel):
    """
    Returned by /upload (or equivalent).
    - doc_id: stable identifier used to filter retrieval to a specific uploaded doc
    - filename: original filename
    - pages/chunks: optional ingestion stats
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    file_size_bytes: int | None = None
    pages: int | None = None
    chunks: int | None = None
    namespace: str | None = None  # if you store uploads in a dedicated namespace
    doc_type: str | None = None
    chunk_chars: int | None = None
    overlap_chars: int | None = None
    extraction_method: str | None = None


class ChatDocRequest(BaseModel):
    """
    Request for chatting against a single uploaded document.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    doc_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=12000, validation_alias="message")

    # retrieval controls
    top_k: conint(ge=1, le=20) = settings.doc_top_k
    # optional: if your retrieval supports namespaces per-doc
    namespace_knowledge: str | None = None
    # optional: filter context retrieval by type
    context_types: list[str] | None = None
    # optional: metadata filters
    section: str | None = Field(default=None, max_length=200)
    page_start: conint(ge=1) | None = None
    page_end: conint(ge=1) | None = None

    # optional: conversation/threading
    thread_id: str | None = Field(default=None, max_length=200)
    style_notes: str | None = Field(default=None, max_length=2000)


class ChatDocResponse(BaseModel):
    """
    Returned by /chat-doc (or equivalent).
    - answer: assistant response
    - citations: evidence used (if any)
    - trace_id: optional engine trace
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    doc_id: str
    thread_id: str | None = None

    # lightweight citations object (compatible with your Researcher evidence shape)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    trace_id: str | None = None
    trace: dict[str, Any] | None = None


# -----------------------------
# Context store schemas
# -----------------------------
class ContextUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200000)
    source: str | None = Field(default=None, max_length=200)
    context_type: str | None = Field(default=None, max_length=100)


class ContextUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str
    namespace: str


class ContextBlueprintUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=200000)
    blueprint: dict[str, Any]

    @field_validator("blueprint", mode="before")
    @classmethod
    def parse_blueprint(cls, v: Any) -> dict[str, Any]:
        return _coerce_to_dict(v)


class ContextBlueprintUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprint_id: str
    namespace: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authenticated: bool
    username: str | None = None


class AuthStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authenticated: bool
    username: str | None = None


class DeleteDocRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1, max_length=200)


class DeleteContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str = Field(min_length=1, max_length=200)
