"""Writer agent - generates final output following blueprint."""

from __future__ import annotations

import logging
from typing import Any, Dict

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import call_chat_completion, _box_untrusted, clamp_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Generates final output following a blueprint."""

    def __init__(self, client: OpenAI):
        """Initialize Writer agent."""
        super().__init__(client)

    def execute(
        self,
        blueprint_json: Dict[str, Any],
        facts: Dict[str, Any],
        style_notes: str | None = None,
    ) -> Dict[str, Any]:
        """
        Generate final output following blueprint.

        Args:
            blueprint_json: Writing blueprint (purpose, tone, format, constraints)
            facts: Facts and evidence to include
            style_notes: Optional additional style guidance

        Returns:
            Dictionary with final output
        """
        final_output = self._write_output(blueprint_json, facts, style_notes)

        return {
            "final": final_output,
            "blueprint_applied": True,
        }

    def _write_output(
        self,
        blueprint: Dict[str, Any],
        facts: Dict[str, Any],
        style_notes: str | None = None,
    ) -> str:
        """Generate output using LLM."""
        purpose = blueprint.get("purpose", "Generate a clear response")
        tone = blueprint.get("tone", "professional")
        format_items = blueprint.get("format", ["summary"])
        constraints = blueprint.get("constraints", [])

        # Build system prompt from blueprint
        system = self._build_system_prompt(purpose, tone, format_items, constraints)

        # Build user prompt with facts
        user = self._build_user_prompt(facts, style_notes)

        try:
            output = call_chat_completion(
                client=self.client,
                model=settings.generation_model,
                system=system,
                user=user,
                max_tokens=min(settings.max_tokens_per_call, 2000),
                temperature=0.2,
            )
            return output.strip()
        except Exception as e:
            logger.error(f"Failed to generate output: {e}")
            return "Failed to generate output."

    @staticmethod
    def _build_system_prompt(
        purpose: str, tone: str, format_items: list[str], constraints: list[str]
    ) -> str:
        """Build system prompt from blueprint."""
        prompt = f"Purpose: {purpose}\nTone: {tone}\n"

        if format_items:
            prompt += f"Format: Include {', '.join(format_items)}\n"

        if constraints:
            prompt += "Constraints:\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"

        return prompt

    @staticmethod
    def _build_user_prompt(facts: Dict[str, Any], style_notes: str | None) -> str:
        """Build user prompt with facts."""
        prompt = "Facts and evidence:\n"

        if isinstance(facts, dict):
            prompt += str(facts)
        else:
            prompt += _box_untrusted(clamp_str(str(facts), 5000))

        if style_notes:
            prompt += f"\n\nAdditional style guidance:\n{style_notes}"

        prompt += "\n\nGenerate the output now."
        return prompt
