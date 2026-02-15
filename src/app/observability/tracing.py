from __future__ import annotations

import logging

from fastapi import FastAPI

from app.core.config import settings


logger = logging.getLogger(__name__)


def _parse_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def setup_tracing(app: FastAPI) -> None:
    if not settings.enable_tracing:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception as exc:
        logger.warning("OpenTelemetry unavailable: %s", exc)
        return

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint or None,
        headers=_parse_headers(settings.otel_exporter_otlp_headers),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
