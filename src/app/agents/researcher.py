"""Researcher agent - performs RAG retrieval with evidence synthesis."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import clamp_str, _box_untrusted, call_chat_completion
from app.retrieval import PineconeRetriever, LLMReranker, EvidenceItem
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Performs RAG retrieval and evidence synthesis."""

    def __init__(self, client: OpenAI, pinecone_index: Any = None):
        """
        Initialize Researcher agent.

        Args:
            client: OpenAI client instance
            pinecone_index: Pinecone index for retrieval
        """
        super().__init__(client)
        self.pinecone_index = pinecone_index
        self.retriever = PineconeRetriever(pinecone_index, client)
        self.reranker = LLMReranker(client)

    def execute(
        self,
        topic_query: str,
        namespace_knowledge: str = None,
        top_k: int = 6,
        doc_id: Optional[str] = None,
        section: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve and synthesize evidence.

        Args:
            topic_query: Search query
            namespace_knowledge: Pinecone namespace (defaults to settings)
            top_k: Number of results to retrieve
            doc_id: Optional document ID filter

        Returns:
            Dictionary with answer, claims, and evidence
        """
        if self.pinecone_index is None:
            logger.warning("Pinecone not available, returning minimal response")
            return {
                "answer": "No retrieval backend configured.",
                "claims": [],
                "evidence": [],
            }

        namespace = namespace_knowledge or settings.namespace_knowledge

        # Retrieve candidates
        meta_filter = {}
        if section:
            meta_filter["section"] = {"$eq": section}
        if page_start or page_end:
            page_range: dict[str, Any] = {}
            if page_start:
                page_range["$gte"] = int(page_start)
            if page_end:
                page_range["$lte"] = int(page_end)
            if page_range:
                meta_filter["page_start"] = page_range

        candidates = self.retriever.retrieve(
            query=topic_query,
            namespace=namespace,
            top_k=top_k,
            doc_id=doc_id,
            meta_filter=meta_filter if meta_filter else None,
            enable_lexical_scoring=settings.enable_bm25_lexical,
        )

        # Rerank if enabled
        evidence = self.reranker.rerank(
            question=topic_query,
            candidates=candidates,
            top_n=settings.rerank_top_n,
        )

        # Synthesize answer
        answer = self._synthesize_answer(topic_query, evidence)

        return {
            "answer": answer,
            "claims": self._extract_claims(evidence),
            "evidence": [e.to_dict() for e in evidence],
        }

    def _synthesize_answer(self, query: str, evidence: list[EvidenceItem]) -> str:
        """Synthesize answer from evidence."""
        if not evidence:
            return "No relevant evidence found."

        # Format evidence for synthesis
        evidence_text = self._format_evidence(evidence)

        system = (
            "You are a research assistant synthesizing answers from evidence.\n"
            "Rules:\n"
            "- Only use provided evidence.\n"
            "- Cite evidence IDs like [e1], [e2] after relevant claims.\n"
            "- Flag uncertainty with phrases like 'appears to', 'suggests', 'may indicate'.\n"
            "- Be precise and concise.\n"
        )

        user = (
            f"Query: {query}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Synthesize a clear answer citing the evidence."
        )

        try:
            answer = call_chat_completion(
                client=self.client,
                model=settings.generation_model,
                system=system,
                user=user,
                max_tokens=min(settings.max_tokens_per_call, 1200),
                temperature=0.1,
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return "Failed to synthesize answer from evidence."

    @staticmethod
    def _format_evidence(evidence: list[EvidenceItem]) -> str:
        """Format evidence for display."""
        lines = []
        for e in evidence:
            lines.append(
                f"[{e.id} | {e.source} | score={e.score:.3f} | page={e.page_start}]\n"
                f"{_box_untrusted(e.text)}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _extract_claims(evidence: list[EvidenceItem]) -> list[str]:
        """Extract key claims from evidence."""
        # Simple implementation: first sentence from each evidence
        claims = []
        for e in evidence:
            sentences = e.text.split(".")
            if sentences:
                claim = sentences[0].strip()
                if claim and len(claim) > 10:
                    claims.append(f"{claim} [{e.id}]")
        return claims[:5]  # Top 5 claims
