"""Verifier agent - quality assurance and verification."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import call_chat_completion, _box_untrusted, clamp_str
from .base import BaseAgent

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    """Verifies output quality and consistency."""

    def __init__(self, client: OpenAI):
        """Initialize Verifier agent."""
        super().__init__(client)

    def execute(
        self,
        draft: str,
        reference: str,
        verification_objective: str | None = None,
    ) -> Dict[str, Any]:
        """
        Verify draft against reference.

        Args:
            draft: Output to verify
            reference: Reference material or evidence
            verification_objective: What to specifically check

        Returns:
            Dictionary with verification result
        """
        is_valid, issues, suggestions = self._verify(
            draft, reference, verification_objective
        )

        return {
            "is_valid": is_valid,
            "issues": issues,
            "suggestions": suggestions,
            "revision": self._suggest_revision(draft, issues) if issues else None,
        }

    def _verify(
        self, draft: str, reference: str, objective: str | None = None
    ) -> tuple[bool, list[str], list[str]]:
        """Verify using LLM."""
        draft = clamp_str(draft, 2000)
        reference = clamp_str(reference, 5000)

        objective_text = objective or "Verify accuracy, consistency, and evidence alignment."

        system = (
            "You are a content verifier. Check if a draft aligns with reference material.\n"
            "Return a JSON object with:\n"
            '- "is_valid": boolean\n'
            '- "issues": list of problems found\n'
            '- "suggestions": list of improvements\n'
        )

        user = (
            f"Objective: {objective_text}\n\n"
            f"Draft:\n{_box_untrusted(draft)}\n\n"
            f"Reference:\n{_box_untrusted(reference)}\n\n"
            "Verify and return JSON."
        )

        try:
            output = call_chat_completion(
                client=self.client,
                model=settings.generation_model,
                system=system,
                user=user,
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            obj = json.loads(output)
            is_valid = obj.get("is_valid", True)
            issues = obj.get("issues", [])
            suggestions = obj.get("suggestions", [])

            return is_valid, issues, suggestions

        except Exception as e:
            logger.error(f"Failed to verify: {e}")
            return True, [], []

    def _suggest_revision(self, draft: str, issues: list[str]) -> str:
        """Suggest revision based on issues."""
        if not issues:
            return draft

        draft = clamp_str(draft, 2000)
        issues_text = "\n".join([f"- {issue}" for issue in issues[:5]])

        system = (
            "You are an editor. Revise a draft to address the specified issues.\n"
            "Keep the core message but improve accuracy and clarity.\n"
        )

        user = (
            f"Original draft:\n{_box_untrusted(draft)}\n\n"
            f"Issues to fix:\n{issues_text}\n\n"
            "Provide revised version."
        )

        try:
            revision = call_chat_completion(
                client=self.client,
                model=settings.generation_model,
                system=system,
                user=user,
                max_tokens=1500,
                temperature=0.2,
            )
            return revision.strip()
        except Exception as e:
            logger.error(f"Failed to suggest revision: {e}")
            return draft
