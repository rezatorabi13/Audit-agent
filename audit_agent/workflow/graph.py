"""Graph assembly and the run/resume facade around it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from audit_agent.workflow.nodes import AuditNodes, WorkflowDependencies
from audit_agent.workflow.state import AuditState

logger = logging.getLogger(__name__)

__all__ = ["AuditWorkflow", "PendingReview", "ReviewDecision", "WorkflowDependencies"]


class ReviewDecision(str, Enum):
    """Outcome a human reviewer returns at the approval gate."""

    APPROVED = "approved"
    REVISE = "revise"


@dataclass(frozen=True, slots=True)
class PendingReview:
    """The graph paused at the approval gate, awaiting a decision."""

    thread_id: str
    vendor: str
    risk_findings: str
    standards_findings: str
    revision_count: int


def build_graph(deps: WorkflowDependencies) -> StateGraph:
    """Wire the nodes into the audit workflow.

    Flow: evidence -> classification -> risk -> standards -> human review,
    then either drafting or back to risk assessment with the reviewer's comment.
    """
    nodes = AuditNodes(deps)

    builder = StateGraph(AuditState)
    builder.add_node("gather_evidence", nodes.gather_evidence)
    builder.add_node("classify_transactions", nodes.classify_transactions)
    builder.add_node("assess_risk", nodes.assess_risk)
    builder.add_node("check_standards", nodes.check_standards)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("draft_memo", nodes.draft_memo)

    builder.add_edge(START, "gather_evidence")
    builder.add_edge("gather_evidence", "classify_transactions")
    builder.add_edge("classify_transactions", "assess_risk")
    builder.add_edge("assess_risk", "check_standards")
    builder.add_edge("check_standards", "human_review")
    builder.add_conditional_edges(
        "human_review",
        nodes.route_after_review,
        {"draft_memo": "draft_memo", "assess_risk": "assess_risk"},
    )
    builder.add_edge("draft_memo", END)

    return builder


class AuditWorkflow:
    """Runs a vendor review, pausing for human approval before drafting."""

    def __init__(self, deps: WorkflowDependencies, checkpointer: Any | None = None) -> None:
        # Checkpointing is what lets a paused review resume in a later process;
        # swap InMemorySaver for a persistent saver to survive restarts.
        self._graph = build_graph(deps).compile(checkpointer=checkpointer or InMemorySaver())

    @property
    def node_names(self) -> list[str]:
        return list(self._graph.get_graph().nodes)

    def mermaid(self) -> str:
        """Mermaid source for the compiled graph, for docs and diagrams."""
        return self._graph.get_graph().draw_mermaid()

    def start(self, vendor: str, question: str, thread_id: str | None = None) -> PendingReview:
        """Run up to the approval gate and return what awaits the reviewer."""
        thread_id = thread_id or f"review-{uuid4().hex[:8]}"
        result = self._graph.invoke(
            {"vendor": vendor, "question": question, "revision_count": 0},
            config=self._config(thread_id),
        )
        return self._to_pending(result, thread_id)

    def resume(
        self,
        thread_id: str,
        decision: ReviewDecision,
        comment: str = "",
    ) -> AuditState | PendingReview:
        """Resume a paused review.

        Returns the completed state when approved, or a fresh
        :class:`PendingReview` when the graph loops back for another pass.
        """
        result = self._graph.invoke(
            Command(resume={"decision": decision.value, "comment": comment}),
            config=self._config(thread_id),
        )
        if "__interrupt__" in result:
            return self._to_pending(result, thread_id)
        return result

    def state(self, thread_id: str) -> AuditState:
        """Current persisted state for a thread, for inspection or audit trails."""
        return self._graph.get_state(self._config(thread_id)).values

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _to_pending(result: dict[str, Any], thread_id: str) -> PendingReview:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            raise RuntimeError(
                "Workflow finished without reaching the review gate; expected an interrupt."
            )
        payload = interrupts[0].value
        return PendingReview(
            thread_id=thread_id,
            vendor=payload["vendor"],
            risk_findings=payload["risk_findings"],
            standards_findings=payload["standards_findings"],
            revision_count=payload.get("revision_count", 0),
        )
