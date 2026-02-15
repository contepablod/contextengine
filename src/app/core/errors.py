"""
Enhanced error handling with circuit breaker pattern and detailed error context.
"""

import logging
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States for circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for handling external service failures.
    Prevents cascading failures by temporarily rejecting requests.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 60,
        name: str = "circuit_breaker",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"{self.name}: Entering HALF_OPEN state")
            else:
                raise Exception(
                    f"Circuit breaker {self.name} is OPEN. Service unavailable."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            logger.error(f"{self.name}: Call failed", exc_info=True)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to retry."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout_s

    def _on_success(self) -> None:
        """Handle successful call."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_success_time = time.time()
        logger.info(f"{self.name}: Call succeeded, circuit CLOSED")

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"{self.name}: Failure threshold reached ({self.failure_count}), circuit OPEN"
            )


class DetailedError(Exception):
    """Enhanced error with context information."""

    def __init__(
        self,
        message: str,
        error_type: str,
        context: Optional[dict] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.context = context or {}
        self.original_error = original_error

        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "error": self.message,
            "type": self.error_type,
            "context": self.context,
        }


# Common circuit breakers
openai_breaker = CircuitBreaker(
    failure_threshold=3, recovery_timeout_s=30, name="openai"
)
pinecone_breaker = CircuitBreaker(
    failure_threshold=5, recovery_timeout_s=60, name="pinecone"
)
