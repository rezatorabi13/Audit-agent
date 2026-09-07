"""Shared state threaded through the workflow graph."""

from __future__ import annotations

from typing import Any, TypedDict


class AuditState(TypedDict, total=False):
    """State object each node reads from and partially updates.

    Nodes return only the keys they change; LangGraph merges the updates. Every
    downstream field traces back to ``evidence``, which is what makes the final
    memo reviewable.
    """

    # Inputs
    vendor: str
    question: str

    # Evidence gathering
    evidence_context: str
    related_party: dict[str, Any]
    anomalies: list[str]
    transaction_labels: list[str]

    # Analysis
    risk_findings: str
    standards_findings: str

    # Review and output
    review_decision: str
    reviewer_comment: str
    revision_count: int
    memo: str
