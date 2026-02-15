"""Integration-style test for `run_engine()` using mocked planning and agents.

This test verifies that the engine executes a simple 3-step plan (Librarian -> Researcher -> Writer)
by patching `plan_steps` and returning mocked agent outputs via a fake AgentFactory.
"""

import pytest

from app.runtime.engine import run_engine
from app.core.schemas import ExecutionPlan, PlanStep


class _MockLibrarian:
    def execute(self, intent_query: str):
        return {"purpose": "test", "tone": "neutral", "format": ["summary"], "constraints": []}


class _MockResearcher:
    def execute(self, topic_query: str, namespace_knowledge: str = None, top_k: int = 5, doc_id: str | None = None):
        return {
            "answer": "evidence-based answer",
            "claims": ["Claim A [e1]"],
            "evidence": [{"id": "e1", "source": "doc.pdf", "score": 0.9, "page_start": 1, "text": "Evidence text"}],
        }


class _MockWriter:
    def execute(self, blueprint_json, facts, style_notes=None):
        return {"final": "Final composed answer from writer"}


class _MockFactory:
    def create_agent(self, agent_name: str):
        if agent_name == "Librarian":
            return _MockLibrarian()
        if agent_name == "Researcher":
            return _MockResearcher()
        if agent_name == "Writer":
            return _MockWriter()
        raise ValueError("Unexpected agent")


def test_run_engine_with_mocked_agents(monkeypatch):
    # 1) Patch planning to return deterministic 3-step plan
    plan = ExecutionPlan(plan=[
        PlanStep(step=1, agent="Librarian", input={"intent_query": "Make a blueprint"}),
        PlanStep(step=2, agent="Researcher", input={"topic_query": "What is X?", "top_k": 3}),
        PlanStep(step=3, agent="Writer", input={"blueprint_json": {}, "facts": {}}),
    ])

    monkeypatch.setattr("app.runtime.engine.plan_steps", lambda client, goal: plan)

    # 2) Patch the engine's factory creator to return our mock factory
    monkeypatch.setattr(
        "app.runtime.engine.get_agent_factory",
        lambda client, pinecone_index=None: _MockFactory(),
    )
    # Prevent actual moderation calls (we're not providing a real OpenAI client)
    monkeypatch.setattr(
        "app.runtime.engine.moderate_text",
        lambda client, text: {"flagged": False},
    )

    # 3) Run engine (client and pinecone_index can be None since mocks don't use them)
    res = run_engine(client=None, pinecone_index=None, goal="Test goal", namespace_context="ctx", namespace_knowledge="kn")

    assert res["blocked"] is False
    assert "output" in res
    assert res["output"] == "Final composed answer from writer"
    # trace should have 3 steps
    assert res.get("trace", {}).get("steps") is not None
    assert len(res["trace"]["steps"]) == 3
