"""
Environment profiles for dev/staging/prod configurations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Environment(Enum):
    """Deployment environments."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


@dataclass
class EnvironmentConfig:
    """Configuration per environment."""

    name: Environment
    debug: bool
    cors_origins: list[str]
    cache_ttl_s: int
    rate_limit_per_minute: int
    max_retries: int
    circuit_breaker_threshold: int
    request_timeout_s: float
    log_level: str
    enable_input_moderation: bool

    @classmethod
    def from_env(cls, env_name: Optional[str] = None) -> "EnvironmentConfig":
        """Create config from environment name."""
        import os

        env_name = env_name or os.getenv("ENVIRONMENT", "dev").lower()

        configs = {
            "dev": EnvironmentConfig(
                name=Environment.DEV,
                debug=True,
                cors_origins=[
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:5500",
                    "http://127.0.0.1:5500",
                    "http://localhost:8080",
                    "http://127.0.0.1:8080",
                ],
                cache_ttl_s=300,
                rate_limit_per_minute=100,
                max_retries=3,
                circuit_breaker_threshold=5,
                request_timeout_s=30.0,
                log_level="DEBUG",
                enable_input_moderation=False,
            ),
            "staging": EnvironmentConfig(
                name=Environment.STAGING,
                debug=False,
                cors_origins=[
                    "https://staging.example.com",
                    "http://localhost:3000",
                ],
                cache_ttl_s=1800,
                rate_limit_per_minute=50,
                max_retries=2,
                circuit_breaker_threshold=3,
                request_timeout_s=20.0,
                log_level="INFO",
                enable_input_moderation=True,
            ),
            "prod": EnvironmentConfig(
                name=Environment.PROD,
                debug=False,
                cors_origins=[
                    "https://example.com",
                    "https://www.example.com",
                ],
                cache_ttl_s=3600,
                rate_limit_per_minute=30,
                max_retries=2,
                circuit_breaker_threshold=2,
                request_timeout_s=15.0,
                log_level="WARNING",
                enable_input_moderation=True,
            ),
        }

        return configs.get(env_name, configs["dev"])
