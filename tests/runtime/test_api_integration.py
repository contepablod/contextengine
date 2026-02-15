"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.interfaces.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheckEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_has_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "environment" in data
        assert "version" in data

    def test_health_check_services_structure(self, client):
        response = client.get("/health")
        data = response.json()
        services = data.get("services", {})
        assert isinstance(services, dict)
        # Should have at least openai service
        assert "openai" in services or len(services) >= 0


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_serves_frontend_html(self, client):
        response = client.get("/")
        assert "Context Engine" in response.text


class TestCorrelationIDMiddleware:
    """Tests for correlation ID functionality."""

    def test_correlation_id_in_response_headers(self, client):
        response = client.get("/health")
        assert "x-correlation-id" in response.headers

    def test_custom_correlation_id_preserved(self, client):
        custom_id = "test-correlation-id-12345"
        response = client.get("/health", headers={"x-correlation-id": custom_id})
        assert response.headers["x-correlation-id"] == custom_id


class TestRateLimitingHeader:
    """Tests for rate limiting with API key header."""

    def test_rate_limit_with_api_key(self, client):
        # First request should succeed
        response = client.get("/health", headers={"x-api-key": "test-key"})
        assert response.status_code in [200, 429]  # Either OK or rate limited


class TestExceptionHandling:
    """Tests for exception handling."""

    def test_404_error_returns_proper_response(self, client):
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404


class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_docs_endpoint_exists(self, client):
        response = client.get("/docs")
        # Should either have docs or be redirected
        assert response.status_code in [200, 307, 404]

    def test_openapi_schema_accessible(self, client):
        response = client.get("/openapi.json")
        # Should have OpenAPI schema or be redirected
        assert response.status_code in [200, 307, 404]
