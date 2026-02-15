"""Integration test for retrieval flow with mocked PineconeRetriever and LLMReranker."""

from app.agents.researcher import ResearcherAgent
from app.retrieval.evidence import EvidenceItem


def test_researcher_retrieval_with_mocked_retriever(monkeypatch):
    # Prepare mock evidence items
    ev1 = EvidenceItem(id="e1", source="doc1.pdf", score=0.95, text="Evidence one.", page_start=1)
    ev2 = EvidenceItem(id="e2", source="doc2.pdf", score=0.88, text="Evidence two.", page_start=2)

    # Patch PineconeRetriever.retrieve to return our candidates
    def _mock_retrieve(self, query, namespace, top_k, doc_id=None, enable_lexical_scoring=False):
        return [ev1, ev2]

    monkeypatch.setattr("app.retrieval.pinecone_client.PineconeRetriever.retrieve", _mock_retrieve)

    # Patch LLMReranker.rerank to simply return the candidates as-is
    monkeypatch.setattr("app.retrieval.reranker.LLMReranker.rerank", lambda self, question, candidates, top_n: candidates)

    # Instantiate agent with mocked pinecone index (not used thanks to patch)
    agent = ResearcherAgent(client=None, pinecone_index=object())

    res = agent.execute(topic_query="Test query", namespace_knowledge="kn", top_k=2)

    assert "answer" in res
    assert isinstance(res["evidence"], list)
    assert len(res["evidence"]) == 2
    ids = [e["id"] for e in res["evidence"]]
    assert "e1" in ids and "e2" in ids
