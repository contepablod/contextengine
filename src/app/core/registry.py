"""Agent registry for instantiating and managing agent classes."""

from __future__ import annotations

from typing import Any, Protocol

from openai import OpenAI

from app.agents.librarian import LibrarianAgent
from app.agents.researcher import ResearcherAgent
from app.agents.writer import WriterAgent
from app.agents.summarizer import SummarizerAgent
from app.agents.verifier import VerifierAgent
from app.core.schemas import (
    AgentName,
    LibrarianInput,
    ResearcherInput,
    SummarizerInput,
    WriterInput,
    VerifierInput,
)


# Map agent name -> Pydantic input schema (strict validation)
AGENT_INPUT_SCHEMAS: dict[str, type] = {
    "Librarian": LibrarianInput,
    "Researcher": ResearcherInput,
    "Summarizer": SummarizerInput,
    "Writer": WriterInput,
    "Verifier": VerifierInput,
}


def validate_agent_input(agent: AgentName, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate agent input using Pydantic schema."""
    schema = AGENT_INPUT_SCHEMAS[agent]
    obj = schema(**payload)  # raises ValidationError if invalid
    return obj.model_dump()


class AgentFactory:
    """Factory for creating agent instances."""

    def __init__(self, client: OpenAI, pinecone_index: Any = None):
        """
        Initialize agent factory.

        Args:
            client: OpenAI client instance
            pinecone_index: Optional Pinecone index
        """
        self.client = client
        self.pinecone_index = pinecone_index

    def create_agent(self, agent_name: AgentName):
        """Create an agent instance by name."""
        if agent_name == "Librarian":
            return LibrarianAgent(self.client, self.pinecone_index)
        elif agent_name == "Researcher":
            return ResearcherAgent(self.client, self.pinecone_index)
        elif agent_name == "Summarizer":
            return SummarizerAgent(self.client)
        elif agent_name == "Writer":
            return WriterAgent(self.client)
        elif agent_name == "Verifier":
            return VerifierAgent(self.client)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")


# Backward compatibility: legacy function interface
def get_agent_factory(client: OpenAI, pinecone_index: Any = None) -> AgentFactory:
    """Get an agent factory instance."""
    return AgentFactory(client, pinecone_index)
