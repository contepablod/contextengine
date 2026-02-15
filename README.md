# Context Engine

Context Engine is a PDF Q&A assistant with citations. It ingests PDFs into Pinecone, extracts text with PyMuPDF, and answers questions via a FastAPI backend and a static HTML frontend.

## Features

- PDF upload + ingestion into a knowledge store (Pinecone)
- Context store for reusable notes (tagged by type)
- Q&A with citations and evidence snippets
- PyMuPDF-based extraction with lightweight layout heuristics
- Hybrid chunking with doc-type detection (scholarly, financial, legal, scan)

## Project layout

- `src/app/core`: Configuration, schemas, registry, and shared utilities
- `src/app/runtime`: Engine and middleware
- `src/app/agents`: Librarian, Researcher, Writer, Summarizer, Verifier
- `src/app/retrieval`: Pinecone retriever and reranker
- `src/app/ingestion`: Extractors, chunking, BM25, ingestion pipeline
- `src/app/storage`: Pinecone init, caches, local file helpers
- `src/app/observability`: Metrics + tracing
- `src/app/interfaces/api`: FastAPI entrypoint
- `src/app/interfaces/web`: Static frontend (served at `/`)
- `tests`: Test suite (see `tests/README.md`)
- `uploads`: Upload staging area
- `docker`: Dockerfiles + compose + observability configs

## Requirements

- Python 3.12 (3.10+ likely OK)
- Pinecone index + API key
- OpenAI-compatible embeddings API (default uses OpenRouter base URL)
  No external PDF extraction service is required.

## Setup

```bash
cd /home/pdconte/Desktop/contextengine
uv venv .venv
source .venv/bin/activate
uv sync --no-dev
```

Optional: pip fallback

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lockfile workflow (when dependencies change):

```bash
cd /home/pdconte/Desktop/contextengine
uv lock
```

## Environment variables

Minimum:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`

Common (optional):

- `OPENAI_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `GENERATION_MODEL`, `PLANNING_MODEL`, `MODERATION_MODEL`
- `RERANKER_MODEL`
- `PINECONE_NAMESPACE_CONTEXT` (default: `ContextLibrary`)
- `PINECONE_NAMESPACE_KNOWLEDGE` (default: `KnowledgeStore`)
- `CHUNK_CHARS`, `CHUNK_OVERLAP_CHARS`
- `ENABLE_LLM_RERANK`, `RERANK_TOP_N`
- `DOC_TOP_K`
- `LOG_FILE` (default: `logs/app.log`)
- `LOG_MAX_BYTES` (default: `10485760`)
- `LOG_BACKUP_COUNT` (default: `5`)
- `ENABLE_METRICS` (default: `true`)
- `METRICS_PATH` (default: `/metrics`)
- `ENABLE_TRACING` (default: `false`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default: empty)
- `OTEL_EXPORTER_OTLP_HEADERS` (default: empty)
- `OTEL_SERVICE_NAME` (default: `contextengine`)
- `CONTEXT_BLUEPRINT_PATH` (default: `context.json`)
- `SEED_CONTEXT_BLUEPRINTS` (default: `true`)

## Run the backend

```bash
cd /home/pdconte/Desktop/contextengine
PYTHONPATH=src uvicorn app.interfaces.api.main:app --host 0.0.0.0 --port 8000
```

Health check: `http://127.0.0.1:8000/health`

## Run the frontend

The frontend is served by the backend at `http://127.0.0.1:8000/`.

## Docker Compose (app)

```bash
cd /home/pdconte/Desktop/contextengine
docker compose -f docker/docker-compose.yml up --build
```

## Docker Compose (app + observability)

This starts Prometheus, Grafana, and Tempo with tracing enabled.

```bash
cd /home/pdconte/Desktop/contextengine
docker compose -f docker/docker-compose.observability.yml up --build
```

Then open:

- App: `http://127.0.0.1:8000`
- Metrics: `http://127.0.0.1:8000/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3001` (default login: `admin` / `admin`)
- Tempo: `http://127.0.0.1:3200`

Grafana dashboards (auto-provisioned):

- `Context Engine - API Overview` (traffic, latency, errors, top endpoints)
- `Context Engine - Traces` (Tempo trace explorer)
- `Context Engine - AI & Vector Metrics` (LLM latency/tokens, vector DB latency)
- `Context Engine - System Metrics` (CPU, memory, network I/O)

## API overview

- `POST /upload` - Upload a PDF and ingest into the knowledge store
- `POST /chat` - Ask a question against an uploaded PDF (returns citations + evidence)
- `POST /context` - Add reusable context text to the context store
- `POST /context-blueprint` - Add a context blueprint (description + blueprint JSON)
- `POST /delete-context` - Delete a single context entry by id
- `POST /reset-context` - Clear the context namespace
- `POST /reset-knowledge` - Clear the knowledge namespace
- `POST /delete-doc` - Delete a single document’s chunks by doc_id
- `GET /health` - Service status
- `GET /models` - Active model configuration and retrieval toggles
- `GET /metrics` - Prometheus metrics (configurable via `METRICS_PATH`)
- `POST /auth/login` - Log in and receive a session cookie
- `POST /auth/logout` - Clear the active session cookie
- `GET /auth/me` - Check authentication status

## Tests

See `tests/README.md` for test commands and structure.

## Notes

- The frontend expects answers to include inline citations like `e1`, `e2`, etc.
- Chunking is document-type aware and stores `doc_type` in metadata for debugging.
- When `ENABLE_BM25_LEXICAL=true`, retrieval uses Pinecone sparse vectors if BM25 stats are available.
- Logs are written to `logs/app.log` by default (rotating).
- Metrics are exposed on `/metrics` when `ENABLE_METRICS=true`.
- Tracing is enabled when `ENABLE_TRACING=true` with an OTLP endpoint.
- System metrics (CPU/memory/network) are collected by node-exporter in the observability stack.
- AI metrics are exposed as `llm_*` (latency, errors, tokens) and vector DB metrics as `vector_db_*`.
