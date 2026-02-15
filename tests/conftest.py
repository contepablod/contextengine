"""Test fixtures and configuration."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
from openai import OpenAI

# Add src directory to path (package root)
app_dir = Path(__file__).parent.parent / "src" / "app"
sys.path.insert(0, str(app_dir.parent))


@pytest.fixture
def openai_client():
    """Provide a mock OpenAI client."""
    return Mock(spec=OpenAI)


@pytest.fixture
def pinecone_index():
    """Provide a mock Pinecone index."""
    index = Mock()
    index.query.return_value = {
        "matches": [
            {
                "id": "chunk_1",
                "score": 0.95,
                "metadata": {"text": "Sample text", "source": "test.pdf"},
            }
        ]
    }
    return index


@pytest.fixture
def environment_config():
    """Provide environment config fixture."""
    from app.core.environment import EnvironmentConfig

    return EnvironmentConfig.from_env("dev")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
