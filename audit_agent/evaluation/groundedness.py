"""Groundedness checking: does the output only assert what the evidence supports?

A second model call critiquing the first is a cheap, model-agnostic guardrail.
It is not a substitute for a full evaluation harness, but it catches the failure
that matters most here - a confident claim with no evidence behind it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from audit_agent.llm.base import LLMProvider
from audit_agent.workflow import prompts

logger = logging.getLogger(__name__)

SUPPORTED_MARKER = "all statements supported"


@dataclass(frozen=True, slots=True)
class GroundednessResult:
    """Verdict on whether every claim in a draft traces to the evidence."""

    is_grounded: bool
    critique: str

    def __bool__(self) -> bool:
        return self.is_grounded


class GroundednessCheck:
    """Asks a model to name any claim the evidence does not support."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def evaluate(self, draft: str, evidence: str) -> GroundednessResult:
        critique = self._llm.complete(
            prompts.FAITHFULNESS_CRITIC,
            f"Evidence:\n{evidence}\n\nDraft to check:\n{draft}",
        )
        is_grounded = SUPPORTED_MARKER in critique.strip().lower()
        logger.info("Groundedness check: %s", "passed" if is_grounded else "flagged claims")
        return GroundednessResult(is_grounded=is_grounded, critique=critique)
