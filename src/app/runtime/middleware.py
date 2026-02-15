"""
Custom middleware for structured logging, correlation IDs, and observability.
"""

import logging
import secrets
import threading
import time
import uuid
from datetime import datetime
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.config import settings

logger = logging.getLogger("app.middleware")


class AuthSessionStore:
    def __init__(self, ttl_s: int):
        self.ttl_s = max(int(ttl_s), 60)
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.ttl_s
        with self._lock:
            self._sessions[token] = expires_at
        return token

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def is_valid(self, token: str | None) -> bool:
        if not token:
            return False
        now = time.time()
        with self._lock:
            expires_at = self._sessions.get(token)
            if expires_at is None:
                return False
            if expires_at < now:
                self._sessions.pop(token, None)
                return False
            return True


auth_store = AuthSessionStore(settings.auth_session_ttl_s)


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforces session auth for protected endpoints."""

    _allowed_paths = {
        "/",
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.enable_auth:
            return await call_next(request)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        path = request.url.path or "/"
        if path in self._allowed_paths or path.startswith("/auth/") or path.startswith("/static/"):
            return await call_next(request)

        token = request.cookies.get(settings.auth_cookie_name)
        if not auth_store.is_valid(token):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)

        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Adds correlation IDs to all requests for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        # Add to response headers
        response = await call_next(request)
        response.headers["x-correlation-id"] = correlation_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs requests and responses with structured data."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        start_time = datetime.utcnow()

        # Log request
        logger.info(
            f"HTTP {request.method} {request.url.path}",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                f"Request failed: {exc}",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                },
                exc_info=True,
            )
            raise

        # Log response
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"HTTP {response.status_code}",
            extra={
                "correlation_id": correlation_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response
