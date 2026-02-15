from __future__ import annotations

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from prometheus_client import Counter, Gauge, Histogram


REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
REQUEST_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "In-progress HTTP requests",
    ["method", "path"],
)
REQUEST_EXCEPTIONS = Counter(
    "http_request_exceptions_total",
    "Total HTTP request exceptions",
    ["method", "path", "exception_type"],
)

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "operation", "status"],
)
LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model", "operation"],
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "type"],
)
LLM_ERRORS = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["model", "operation", "error_type"],
)

VECTOR_DB_REQUESTS = Counter(
    "vector_db_requests_total",
    "Vector database requests",
    ["operation", "status"],
)
VECTOR_DB_REQUEST_DURATION = Histogram(
    "vector_db_request_duration_seconds",
    "Vector database request duration in seconds",
    ["operation"],
)
VECTOR_DB_RESULTS = Histogram(
    "vector_db_query_results",
    "Vector database results per query",
    ["operation"],
)


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    return request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = _route_label(request)
        method = request.method
        REQUEST_IN_PROGRESS.labels(method, path).inc()
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            REQUEST_EXCEPTIONS.labels(method, path, type(exc).__name__).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            status = response.status_code if response is not None else 500
            REQUESTS_TOTAL.labels(method, path, str(status)).inc()
            REQUEST_DURATION.labels(method, path).observe(duration)
            REQUEST_IN_PROGRESS.labels(method, path).dec()
