"""Test suite for Context Engine agents."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from openai import OpenAI

from app.agents import (
    LibrarianAgent,
    ResearcherAgent,
    WriterAgent,
    SummarizerAgent,
    VerifierAgent,
)
from app.retrieval import EvidenceItem


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    return Mock(spec=OpenAI)


@pytest.fixture
def mock_pinecone():
    """Create a mock Pinecone index."""
    return MagicMock()


class TestLibrarianAgent:
    """Test Librarian agent."""

    def test_initialization(self, mock_client):
        """Test agent initialization."""
        agent = LibrarianAgent(mock_client)
        assert agent is not None
        assert agent.agent_name == "LibrarianAgent"

    def test_execute_without_pinecone(self, mock_client):
        """Test execution without Pinecone."""
        agent = LibrarianAgent(mock_client, pinecone_index=None)

        # Mock the LLM call
        with patch("app.agents.librarian.call_chat_completion") as mock_call:
            mock_call.return_value = (
                '{"purpose": "test", "tone": "clear", "format": ["summary"], '
                '"constraints": ["accurate"]}'
            )

            result = agent.execute(intent_query="Test query")

            assert isinstance(result, dict)
            assert "purpose" in result
            assert "tone" in result
            assert "format" in result
            assert "constraints" in result

    def test_default_blueprint(self, mock_client):
        """Test default blueprint fallback."""
        agent = LibrarianAgent(mock_client)
        blueprint = agent._default_blueprint()

        assert blueprint["purpose"] == "paper_qa_assistant"
        assert blueprint["tone"] == "clear, technical, and cautious"
        assert isinstance(blueprint["format"], list)
        assert isinstance(blueprint["constraints"], list)


class TestResearcherAgent:
    """Test Researcher agent."""

    def test_initialization(self, mock_client, mock_pinecone):
        """Test agent initialization."""
        agent = ResearcherAgent(mock_client, mock_pinecone)
        assert agent is not None
        assert agent.retriever is not None
        assert agent.reranker is not None

    def test_execute_without_pinecone(self, mock_client):
        """Test execution without Pinecone."""
        agent = ResearcherAgent(mock_client, pinecone_index=None)
        result = agent.execute(topic_query="Test query")

        assert result["answer"] == "No retrieval backend configured."
        assert result["evidence"] == []
        assert result["claims"] == []

    def test_format_evidence(self):
        """Test evidence formatting."""
        evidence = [
            EvidenceItem(
                id="e1",
                source="paper.pdf",
                score=0.95,
                text="This is test evidence.",
            ),
            EvidenceItem(
                id="e2",
                source="doc.txt",
                score=0.87,
                text="More evidence here.",
            ),
        ]

        formatted = ResearcherAgent._format_evidence(evidence)
        assert "e1" in formatted
        assert "e2" in formatted
        assert "paper.pdf" in formatted
        assert "0.95" in formatted


class TestWriterAgent:
    """Test Writer agent."""

    def test_initialization(self, mock_client):
        """Test agent initialization."""
        agent = WriterAgent(mock_client)
        assert agent is not None

    def test_execute(self, mock_client):
        """Test output generation."""
        agent = WriterAgent(mock_client)

        blueprint = {
            "purpose": "Summarize findings",
            "tone": "academic",
            "format": ["summary"],
            "constraints": ["cite sources"],
        }
        facts = {"key_finding": "Test result"}

        with patch("app.agents.writer.call_chat_completion") as mock_call:
            mock_call.return_value = "Generated output based on blueprint."

            result = agent.execute(blueprint_json=blueprint, facts=facts)

            assert "final" in result
            assert result["blueprint_applied"] is True

    def test_build_system_prompt(self):
        """Test system prompt generation."""
        prompt = WriterAgent._build_system_prompt(
            purpose="Test purpose",
            tone="casual",
            format_items=["summary", "bullets"],
            constraints=["be brief"],
        )

        assert "Test purpose" in prompt
        assert "casual" in prompt
        assert "summary" in prompt
        assert "be brief" in prompt


class TestSummarizerAgent:
    """Test Summarizer agent."""

    def test_initialization(self, mock_client):
        """Test agent initialization."""
        agent = SummarizerAgent(mock_client)
        assert agent is not None

    def test_execute(self, mock_client):
        """Test summarization."""
        agent = SummarizerAgent(mock_client)
        long_text = "This is a long text. " * 100

        with patch("app.agents.summarizer.call_chat_completion") as mock_call:
            mock_call.return_value = "Short summary."

            result = agent.execute(text_to_summarize=long_text, max_words=50)

            assert "summary" in result
            assert result["original_length"] > result["summary_length"]


class TestVerifierAgent:
    """Test Verifier agent."""

    def test_initialization(self, mock_client):
        """Test agent initialization."""
        agent = VerifierAgent(mock_client)
        assert agent is not None

    def test_execute(self, mock_client):
        """Test verification."""
        agent = VerifierAgent(mock_client)

        with patch("app.agents.verifier.call_chat_completion") as mock_call:
            mock_call.return_value = (
                '{"is_valid": true, "issues": [], "suggestions": []}'
            )

            result = agent.execute(draft="Test draft", reference="Test reference")

            assert "is_valid" in result
            assert "issues" in result
            assert "suggestions" in result


class TestEvidenceItem:
    """Test EvidenceItem dataclass."""

    def test_creation(self):
        """Test evidence item creation."""
        evidence = EvidenceItem(
            id="e1",
            source="test.pdf",
            score=0.95,
            text="Test text",
            page_start=1,
            page_end=2,
        )

        assert evidence.id == "e1"
        assert evidence.source == "test.pdf"
        assert evidence.score == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        evidence = EvidenceItem(
            id="e1",
            source="test.pdf",
            score=0.95,
            text="Test text",
        )

        d = evidence.to_dict()
        assert d["id"] == "e1"
        assert d["source"] == "test.pdf"
        assert d["score"] == 0.95

    def test_from_pinecone_match(self):
        """Test creation from Pinecone match."""
        match = {
            "score": 0.92,
            "metadata": {
                "text": "Match text",
                "filename": "doc.pdf",
                "page": 5,
                "section": "Introduction",
            },
        }

        evidence = EvidenceItem.from_pinecone_match(match, index=0)

        assert evidence.id == "e1"
        assert evidence.source == "doc.pdf"
        assert evidence.score == 0.92
        assert evidence.page_start == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
