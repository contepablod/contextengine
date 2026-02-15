"""
Unit tests for core components.
"""

import pytest
from app.storage.cache import SimpleCache, EmbeddingCache, ResponseCache
from app.core.environment import EnvironmentConfig, Environment
from app.core.errors import CircuitBreaker, DetailedError


class TestSimpleCache:
    """Tests for SimpleCache."""

    def test_cache_set_and_get(self):
        cache = SimpleCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        cache = SimpleCache()
        assert cache.get("nonexistent") is None

    def test_cache_expiry(self, monkeypatch):
        cache = SimpleCache(default_ttl_s=0)
        cache.set("key1", "value1")
        # Simulate time passing
        import time

        monkeypatch.setattr(time, "time", lambda: time.time() + 1)
        assert cache.get("key1") is None

    def test_cache_clear(self):
        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_embedding_cache_set_and_get(self):
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]
        cache.set_embedding("test text", "model-v1", embedding)
        assert cache.get_embedding("test text", "model-v1") == embedding

    def test_embedding_cache_model_isolation(self):
        cache = EmbeddingCache()
        embedding1 = [0.1, 0.2]
        embedding2 = [0.3, 0.4]
        cache.set_embedding("same text", "model-v1", embedding1)
        cache.set_embedding("same text", "model-v2", embedding2)
        assert cache.get_embedding("same text", "model-v1") == embedding1
        assert cache.get_embedding("same text", "model-v2") == embedding2


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_breaker_success(self):
        breaker = CircuitBreaker(failure_threshold=3)

        def success_func():
            return "ok"

        result = breaker.call(success_func)
        assert result == "ok"

    def test_circuit_breaker_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=2)

        def fail_func():
            raise Exception("Service unavailable")

        # First two failures
        with pytest.raises(Exception):
            breaker.call(fail_func)
        with pytest.raises(Exception):
            breaker.call(fail_func)

        # Third call should fail with circuit breaker error
        with pytest.raises(Exception, match="Circuit breaker"):
            breaker.call(fail_func)

    def test_circuit_breaker_recovery(self, monkeypatch):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=1)

        def fail_func():
            raise Exception("Service unavailable")

        def success_func():
            return "ok"

        # Trigger open state
        with pytest.raises(Exception):
            breaker.call(fail_func)

        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker"):
            breaker.call(success_func)

        # Simulate time passing and recovery
        import time

        current_time = time.time()
        monkeypatch.setattr(time, "time", lambda: current_time + 2)

        # Should attempt recovery
        result = breaker.call(success_func)
        assert result == "ok"
        assert breaker.state.value == "closed"


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_dev_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "dev")
        config = EnvironmentConfig.from_env()
        assert config.name == Environment.DEV
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.rate_limit_per_minute == 100

    def test_prod_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "prod")
        config = EnvironmentConfig.from_env()
        assert config.name == Environment.PROD
        assert config.debug is False
        assert config.log_level == "WARNING"
        assert config.rate_limit_per_minute == 30

    def test_staging_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "staging")
        config = EnvironmentConfig.from_env()
        assert config.name == Environment.STAGING
        assert config.enable_input_moderation is True


class TestDetailedError:
    """Tests for DetailedError."""

    def test_error_creation(self):
        error = DetailedError(
            message="Service failed",
            error_type="SERVICE_ERROR",
            context={"service": "openai", "code": 500},
        )
        assert error.message == "Service failed"
        assert error.error_type == "SERVICE_ERROR"

    def test_error_to_dict(self):
        error = DetailedError(
            message="Service failed",
            error_type="SERVICE_ERROR",
            context={"service": "openai"},
        )
        error_dict = error.to_dict()
        assert error_dict["error"] == "Service failed"
        assert error_dict["type"] == "SERVICE_ERROR"
        assert error_dict["context"]["service"] == "openai"
