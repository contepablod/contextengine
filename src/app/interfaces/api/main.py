from __future__ import annotations

import json
import hmac
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.registry import get_agent_factory
from app.core.config import settings
from app.runtime.engine import run_engine
from app.core.utils.helpers import (
    clamp_str,
    get_embedding,
    make_openai_client,
    verify_password,
)
from app.ingestion import ingest_pdf_to_pinecone
from app.core.schemas import (
    GenerateRequest,
    GenerateResponse,
    UploadResponse,
    ChatDocRequest,
    ChatDocResponse,
    ContextUploadRequest,
    ContextUploadResponse,
    ContextBlueprintUploadRequest,
    ContextBlueprintUploadResponse,
    LoginRequest,
    LoginResponse,
    AuthStatusResponse,
    DeleteDocRequest,
    DeleteContextRequest,
)
from app.runtime.middleware import (
    AuthMiddleware,
    CorrelationIDMiddleware,
    RequestLoggingMiddleware,
    auth_store,
)
from app.observability.metrics import MetricsMiddleware
from app.observability.tracing import setup_tracing
from app.core.environment import EnvironmentConfig
from app.storage import (
    embedding_cache,
    response_cache,
    ensure_namespaces,
    init_index,
    seed_context_blueprints,
    remove_temp_file,
    write_temp_file,
)

# ----------------------------
# Logging
# ----------------------------

# Load environment-specific config
env_config = EnvironmentConfig.from_env()

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _configure_logging(log_level: str) -> None:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    log_file = os.getenv(
        "LOG_FILE",
        os.path.join(project_root, "logs", "app.log"),
    )
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_handler: RotatingFileHandler | None = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        try:
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=log_max_bytes, backupCount=log_backup_count
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            handlers.append(file_handler)
        except Exception:
            # Fall back to console logging if file logging fails.
            pass

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    if file_handler:
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(logger_name).addHandler(file_handler)


_configure_logging(env_config.log_level)
logger = logging.getLogger("app")

# ----------------------------
# Simple in-memory rate limiter
# ----------------------------
# NOTE: This is per-process and resets on restart.
# For real production, use Redis (e.g., slowapi + redis) or a gateway rate limiter.

_rate: dict[str, list[float]] = {}
_RATE_MAX_KEYS = 50_000  # safety cap to avoid memory blowups


def _rate_limit_key(x_api_key: str) -> str:
    return (x_api_key or "anonymous").strip() or "anonymous"


def _check_rate_limit(key: str) -> None:
    now = time.time()
    window_s = 60.0
    limit = settings.rate_limit_per_minute

    # quick cleanup if keyspace explodes
    if len(_rate) > _RATE_MAX_KEYS:
        # drop oldest keys crudely: reset everything (best effort)
        _rate.clear()

    ts = _rate.get(key, [])
    ts = [t for t in ts if now - t < window_s]
    if len(ts) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    ts.append(now)
    _rate[key] = ts


# ----------------------------
# App lifespan (startup/shutdown)
# ----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_auth:
        if not settings.auth_username or not settings.auth_password_hash:
            raise RuntimeError(
                "AUTH_USERNAME and AUTH_PASSWORD_HASH must be set when ENABLE_AUTH=true."
            )

    # Create clients once per process
    openai_client = make_openai_client()

    pinecone_index = None
    if settings.pinecone_api_key and settings.pinecone_index:
        try:
            pinecone_index = init_index(
                settings.pinecone_api_key, settings.pinecone_index
            )
            if pinecone_index:
                logger.info("Pinecone initialized index=%s", settings.pinecone_index)
                ensure_namespaces(
                    pinecone_index,
                    [settings.namespace_context, settings.namespace_knowledge],
                )
                if settings.seed_context_blueprints:
                    seeded = seed_context_blueprints(
                        openai_client,
                        pinecone_index,
                        settings.namespace_context,
                        settings.context_blueprint_path,
                    )
                    if seeded:
                        logger.info("Seeded %d context blueprints", seeded)
            else:
                logger.warning("Pinecone not configured; empty index settings.")
        except Exception as e:
            # Don't crash app: allow non-RAG fallback
            logger.exception("Failed to init Pinecone index: %s", e)
            pinecone_index = None
    else:
        logger.warning(
            "Pinecone not configured; RAG will fallback with minimal outputs."
        )

    app.state.openai_client = openai_client
    app.state.pinecone_index = pinecone_index
    # AgentFactory provides class-based agents (Phase 1 refactor)
    try:
        app.state.agent_factory = get_agent_factory(openai_client, pinecone_index)
    except Exception:
        app.state.agent_factory = None

    yield

    # If you had closeable clients, you'd close them here.
    # (OpenAI client doesn't require explicit close in typical usage.)


app = FastAPI(
    title="Assistant (Production)",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add middleware in reverse order (last added = first executed)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIDMiddleware)
if settings.enable_metrics:
    app.add_middleware(MetricsMiddleware)
if settings.enable_auth:
    app.add_middleware(AuthMiddleware)

# Add CORS with environment-specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=env_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_tracing(app)

_FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "web")
)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "ok": True,
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "environment": env_config.name.value,
    }


@app.get("/status")
def status() -> dict:
    return {
        "ok": True,
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "environment": env_config.name.value,
    }


@app.get("/health")
def health_check() -> dict:
    """
    Enhanced health check endpoint with service status.
    Returns 200 if app is operational, 503 if critical services fail.
    """
    health_status = {"status": "healthy", "services": {}}

    # Check OpenAI client
    try:
        openai_client = getattr(app.state, "openai_client", None)
        health_status["services"]["openai"] = "available" if openai_client else "unavailable"
    except Exception as e:
        logger.warning("Health check: OpenAI check failed: %s", e)
        health_status["services"]["openai"] = "error"

    # Check Pinecone
    try:
        pinecone_index = getattr(app.state, "pinecone_index", None)
        health_status["services"]["pinecone"] = (
            "available" if pinecone_index else "not_configured"
        )
    except Exception as e:
        logger.warning("Health check: Pinecone check failed: %s", e)
        health_status["services"]["pinecone"] = "error"

    # Check agent factory
    try:
        agent_factory = getattr(app.state, "agent_factory", None)
        health_status["services"]["agents"] = "available" if agent_factory else "unavailable"
    except Exception as e:
        logger.warning("Health check: Agent factory check failed: %s", e)
        health_status["services"]["agents"] = "error"

    # Set overall status to unhealthy if critical services are down
    if health_status["services"].get("openai") != "available":
        health_status["status"] = "unhealthy"
    if (
        health_status["services"].get("pinecone") == "error"
        or health_status["services"].get("agents") == "error"
    ):
        health_status["status"] = "degraded"

    health_status["environment"] = env_config.name.value
    health_status["version"] = "2.0.0"

    return health_status


if settings.enable_metrics:
    @app.get(settings.metrics_path)
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
def models() -> dict:
    return {
        "generation_model": settings.generation_model,
        "planning_model": settings.planning_model,
        "moderation_model": settings.moderation_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "enable_llm_rerank": settings.enable_llm_rerank,
        "rerank_top_n": settings.rerank_top_n,
        "doc_top_k": settings.doc_top_k,
        "enable_bm25_lexical": settings.enable_bm25_lexical,
    }


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, response: Response) -> LoginResponse:
    if not settings.enable_auth:
        return LoginResponse(authenticated=True, username=settings.auth_username or None)

    username_ok = hmac.compare_digest(req.username, settings.auth_username)
    password_ok = verify_password(req.password, settings.auth_password_hash)
    if not (username_ok and password_ok):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = auth_store.create()
    response.set_cookie(
        settings.auth_cookie_name,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        max_age=settings.auth_session_ttl_s,
        path="/",
    )
    return LoginResponse(authenticated=True, username=settings.auth_username)


@app.post("/auth/logout", response_model=LoginResponse)
def logout(request: Request, response: Response) -> LoginResponse:
    token = request.cookies.get(settings.auth_cookie_name)
    auth_store.revoke(token)
    response.delete_cookie(settings.auth_cookie_name, path="/")
    return LoginResponse(authenticated=False, username=None)


@app.get("/auth/me", response_model=AuthStatusResponse)
def auth_status(request: Request) -> AuthStatusResponse:
    if not settings.enable_auth:
        return AuthStatusResponse(authenticated=True, username=settings.auth_username or None)
    token = request.cookies.get(settings.auth_cookie_name)
    authenticated = auth_store.is_valid(token)
    return AuthStatusResponse(
        authenticated=authenticated,
        username=settings.auth_username if authenticated else None,
    )
def _context_id(text: str, source: str | None, context_type: str | None) -> str:
    seed = f"{source or ''}|{context_type or ''}|{text}"
    return "ctx-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()


@app.post("/context", response_model=ContextUploadResponse)
def upload_context(req: ContextUploadRequest) -> ContextUploadResponse:
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot upload context.",
        )

    openai_client = getattr(app.state, "openai_client", None)
    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not configured; cannot embed context.",
        )

    context_type = (req.context_type or "").strip() or "general"
    text = clamp_str(req.text, settings.max_context_chars)
    emb = get_embedding(openai_client, text, settings.embedding_model)
    context_id = _context_id(text, req.source, context_type)
    meta = {
        "source": req.source or context_type or "manual",
        "context_type": context_type,
        "text": clamp_str(text, 15000),
    }

    pinecone_index.upsert(
        vectors=[{"id": context_id, "values": emb, "metadata": meta}],
        namespace=settings.namespace_context,
    )

    return ContextUploadResponse(
        context_id=context_id,
        namespace=settings.namespace_context,
    )


@app.post("/context-blueprint", response_model=ContextBlueprintUploadResponse)
def upload_context_blueprint(
    req: ContextBlueprintUploadRequest,
) -> ContextBlueprintUploadResponse:
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot upload context blueprint.",
        )

    openai_client = getattr(app.state, "openai_client", None)
    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not configured; cannot embed context blueprint.",
        )

    blueprint_id = (req.id or "").strip()
    description = clamp_str(req.description, settings.max_context_chars)
    blueprint = req.blueprint
    if not blueprint_id:
        raise HTTPException(status_code=422, detail="Blueprint id is required.")
    if not description:
        raise HTTPException(status_code=422, detail="Blueprint description is required.")
    if not isinstance(blueprint, dict) or not blueprint:
        raise HTTPException(status_code=422, detail="Blueprint must be a non-empty object.")

    emb = get_embedding(openai_client, description, settings.embedding_model)
    blueprint_json = json.dumps(blueprint, ensure_ascii=True)
    meta = {
        "description": description,
        "blueprint": blueprint_json,
    }

    pinecone_index.upsert(
        vectors=[{"id": blueprint_id, "values": emb, "metadata": meta}],
        namespace=settings.namespace_context,
    )

    return ContextBlueprintUploadResponse(
        blueprint_id=blueprint_id,
        namespace=settings.namespace_context,
    )


@app.post("/reset-context")
def reset_context_store() -> dict:
    """
    Clears the context namespace in Pinecone.
    """
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot reset context store.",
        )

    namespace = settings.namespace_context
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        logger.exception("Failed to reset context namespace %s: %s", namespace, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset context store: {e}",
        )

    return {"ok": True, "namespace": namespace}


@app.post("/delete-context")
def delete_context(req: DeleteContextRequest) -> dict:
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot delete context.",
        )

    namespace = settings.namespace_context
    try:
        pinecone_index.delete(ids=[req.context_id], namespace=namespace)
    except Exception as e:
        logger.exception("Failed to delete context %s: %s", req.context_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete context: {e}",
        )

    return {"ok": True, "context_id": req.context_id, "namespace": namespace}


@app.post("/reset-knowledge")
def reset_knowledge_store() -> dict:
    """
    Clears the knowledge namespace in Pinecone.
    """
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot reset knowledge store.",
        )

    namespace = settings.namespace_knowledge
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        logger.exception("Failed to reset knowledge namespace %s: %s", namespace, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset knowledge store: {e}",
        )

    return {"ok": True, "namespace": namespace}


@app.post("/delete-doc")
def delete_doc(req: DeleteDocRequest) -> dict:
    pinecone_index = getattr(app.state, "pinecone_index", None)
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured; cannot delete document.",
        )

    namespace = settings.namespace_knowledge
    try:
        pinecone_index.delete(
            delete_all=False,
            namespace=namespace,
            filter={"doc_id": {"$eq": req.doc_id}},
        )
    except Exception as e:
        logger.exception("Failed to delete doc %s: %s", req.doc_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {e}",
        )

    return {"ok": True, "doc_id": req.doc_id, "namespace": namespace}


# ----------------------------
# Exception handling
# ----------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    # Preserve intended HTTP errors (429, 400, etc.)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ----------------------------
# Routes
# ----------------------------


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, x_api_key: str = Header(default=None)):
    # Rate limit (best-effort, per-process)
    _check_rate_limit(_rate_limit_key(x_api_key))

    ns_ctx = req.namespace_context or settings.namespace_context
    ns_kn = req.namespace_knowledge or settings.namespace_knowledge

    openai_client = app.state.openai_client
    pinecone_index = app.state.pinecone_index

    result = run_engine(
        client=openai_client,
        pinecone_index=pinecone_index,
        goal=req.goal,
        namespace_context=ns_ctx,
        namespace_knowledge=ns_kn,
    )

    # Optional: enforce output shape through Pydantic (FastAPI also validates via response_model)
    return GenerateResponse(**result).model_dump()
    # return result


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...), x_api_key: str = Header(default=None)
):
    _check_rate_limit(_rate_limit_key(x_api_key))

    if app.state.pinecone_index is None:
        raise HTTPException(status_code=400, detail="Pinecone is not configured.")

    max_mb = int(os.getenv("MAX_UPLOAD_MB", "25"))
    raw = await file.read()
    file_size_bytes = len(raw)
    if file_size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{max_mb}MB).")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported for now.")

    ns_kn = settings.namespace_knowledge  # store doc chunks in your knowledge namespace
    tmp_path = None
    try:
        tmp_path = write_temp_file(raw, suffix=".pdf")
        result = ingest_pdf_to_pinecone(
            client=app.state.openai_client,
            pinecone_index=app.state.pinecone_index,
            pdf_path=tmp_path,
            namespace=ns_kn,
            metadata_extra={"filename": file.filename},
        )
    finally:
        remove_temp_file(tmp_path)

    return UploadResponse(
        doc_id=result.get("doc_id", ""),
        filename=file.filename,
        file_size_bytes=file_size_bytes,
        pages=result.get("pages"),
        chunks=result.get("chunks_upserted"),
        namespace=result.get("namespace"),
        doc_type=result.get("doc_type"),
        chunk_chars=result.get("chunk_chars"),
        overlap_chars=result.get("overlap_chars"),
        extraction_method=result.get("extraction_method"),
    ).model_dump()


@app.post("/chat", response_model=ChatDocResponse)
def chat_doc(req: ChatDocRequest, x_api_key: str = Header(default=None)):
    _check_rate_limit(_rate_limit_key(x_api_key))

    if app.state.pinecone_index is None:
        raise HTTPException(status_code=400, detail="Pinecone is not configured.")

    # 1) retrieve + synthesize answer with citations via AgentFactory
    factory = getattr(app.state, "agent_factory", None)
    if factory is None:
        raise HTTPException(status_code=500, detail="AgentFactory not initialized")

    researcher = factory.create_agent("Researcher")
    effective_top_k = int(req.top_k or settings.doc_top_k)
    if settings.enable_llm_rerank:
        min_k = max(1, int(settings.rerank_top_n))
        effective_top_k = max(effective_top_k, min_k)
        effective_top_k = min(effective_top_k, min_k * 2)

    facts = researcher.execute(
        topic_query=req.question,
        namespace_knowledge=req.namespace_knowledge or settings.namespace_knowledge,
        top_k=effective_top_k,
        doc_id=req.doc_id,
        section=req.section,
        page_start=req.page_start,
        page_end=req.page_end,
    )

    # 2) (optional) pass through Writer for nice formatting using your Librarian blueprint
    librarian = factory.create_agent("Librarian")
    blueprint = librarian.execute(
        intent_query=(
            f"Answer questions about an uploaded PDF with citations. Question: {req.question}"
        ),
        context_types=req.context_types,
    )

    writer = factory.create_agent("Writer")
    final = writer.execute(blueprint_json=blueprint, facts=facts, style_notes=req.style_notes)

    return {
        "doc_id": req.doc_id,
        "answer": final["final"],
        "citations": facts.get("evidence", []),
        "evidence": facts.get("evidence", []),
        "thread_id": req.thread_id,
    }
