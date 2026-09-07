"""The retrieval contract consumers depend on.

Callers depend on this protocol rather than the Chroma implementation, so the
index can be swapped (or stubbed in tests) without touching the workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from audit_agent.models import RetrievedChunk


@runtime_checkable
class SearchIndex(Protocol):
    """Anything that can return scored, citable chunks for a query."""

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        ...

    def __len__(self) -> int:
        ...


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks as a citation-tagged prompt context block."""
    return "\n\n---\n\n".join(f"[Source: {c.citation}]\n{c.text}" for c in chunks)
