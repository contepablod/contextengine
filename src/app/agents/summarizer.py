"""Summarizer agent - condenses long texts."""

from __future__ import annotations

import logging
from typing import Any, Dict

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import call_chat_completion, _box_untrusted, clamp_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """Summarizes long texts into concise form."""

    def __init__(self, client: OpenAI):
        """Initialize Summarizer agent."""
        super().__init__(client)

    def execute(
        self,
        text_to_summarize: str,
        max_words: int = 300,
    ) -> Dict[str, Any]:
        """
        Summarize text.

        Args:
            text_to_summarize: Text to condense
            max_words: Maximum words in summary

        Returns:
            Dictionary with summary
        """
        summary = self._summarize(text_to_summarize, max_words)

        return {
            "summary": summary,
            "original_length": len(text_to_summarize.split()),
            "summary_length": len(summary.split()),
        }

    def _summarize(self, text: str, max_words: int) -> str:
        """Summarize using LLM."""
        text = clamp_str(text, settings.max_context_chars)

        system = (
            f"You are a text summarizer. Condense the provided text to at most {max_words} words.\n"
            "Keep key facts and remove redundancy. Be precise.\n"
        )

        user = f"Text to summarize:\n{_box_untrusted(text)}\n\nProvide summary now."

        try:
            summary = call_chat_completion(
                client=self.client,
                model=settings.generation_model,
                system=system,
                user=user,
                max_tokens=int(max_words * 1.5),
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize: {e}")
            return text[:500] + "..."
