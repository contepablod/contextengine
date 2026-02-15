"""Base agent class with common functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the Context Engine."""

    def __init__(self, client: OpenAI):
        """
        Initialize the agent.

        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.agent_name = self.__class__.__name__

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with given inputs.

        Returns:
            Dictionary containing agent output
        """
        pass

    def log_execution(self, duration_s: float, success: bool, error: str | None = None):
        """Log agent execution metrics."""
        status = "success" if success else "error"
        logger.info(
            f"Agent {self.agent_name} finished",
            extra={
                "agent": self.agent_name,
                "duration_s": duration_s,
                "status": status,
                "error": error,
            },
        )

    def execute_with_timing(self, **kwargs) -> Dict[str, Any]:
        """Execute agent and track timing."""
        start_time = time.time()
        try:
            result = self.execute(**kwargs)
            duration = time.time() - start_time
            self.log_execution(duration, success=True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.log_execution(duration, success=False, error=str(e))
            raise
