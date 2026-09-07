"""Node implementations for the audit workflow.

Each node is a method on :class:`AuditNodes` so its collaborators arrive by
constructor injection rather than module-level globals, which keeps the nodes
independently testable with stub providers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langgraph.types import interrupt

from audit_agent.knowledge.ledger_rules import detect_anomalies
from audit_agent.llm.base import LLMProvider
from audit_agent.retrieval.hybrid import HybridRetriever
from audit_agent.workflow import prompts
from audit_agent.workflow.classifier import RuleBasedClassifier, TransactionClassifier
from audit_agent.workflow.state import AuditState

logger = logging.getLogger(__name__)

APPROVED = "approved"
REVISE = "revise"

# Bounds the review loop so a reviewer who keeps requesting changes cannot
# leave the graph cycling indefinitely.
MAX_REVISIONS = 2


@dataclass
class WorkflowDependencies:
    """Collaborators the workflow needs, assembled once by the caller."""

    retriever: HybridRetriever
    llm: LLMProvider
    ledger: pd.DataFrame
    client_name: str
    classifier: TransactionClassifier = field(default_factory=RuleBasedClassifier)


class AuditNodes:
    """Graph node bodies, bound to a set of dependencies."""

    def __init__(self, deps: WorkflowDependencies) -> None:
        self._deps = deps

    # --- Evidence --------------------------------------------------------

    def gather_evidence(self, state: AuditState) -> dict[str, Any]:
        """Hybrid retrieval: graph traversal, ledger rules, and vector search."""
        bundle = self._deps.retriever.gather(state["vendor"], state["question"])
        return {
            "evidence_context": bundle.as_context(),
            "related_party": {
                "vendor": bundle.related_party.vendor,
                "is_related_party": bundle.related_party.is_related_party,
                "shared_board_members": list(bundle.related_party.shared_board_members),
            },
            "anomalies": [str(flag) for flag in bundle.anomalies],
        }

    def classify_transactions(self, state: AuditState) -> dict[str, Any]:
        """Attach a fixed-vocabulary label to each of the vendor's ledger entries.

        Classification is separated from narration so the labels a downstream
        system parses are produced deterministically, not phrased by a chat model.
        """
        vendor = state["vendor"]
        vendor_ledger = self._deps.ledger[self._deps.ledger["vendor"] == vendor]
        if vendor_ledger.empty:
            return {"transaction_labels": []}

        rules_by_transaction: dict[str, list[str]] = defaultdict(list)
        for flag in detect_anomalies(self._deps.ledger, vendor=vendor):
            rules_by_transaction[flag.transaction_id].append(flag.rule)

        is_related = bool(state.get("related_party", {}).get("is_related_party"))
        classifier = self._deps.classifier

        labels: list[str] = []
        for row in vendor_ledger.itertuples(index=False):
            transaction_id = str(row.transaction_id)
            rules = rules_by_transaction.get(transaction_id, [])
            if isinstance(classifier, RuleBasedClassifier):
                label = classifier.classify_from_rules(rules, is_related)
            else:
                description = (
                    f"{row.description} for ${row.debit:,.2f} from {row.vendor} on {row.date}"
                )
                label = classifier.classify(description)
            labels.append(f"{transaction_id}: {label}")

        logger.info("Classified %d transaction(s) with %s", len(labels), classifier.name)
        return {"transaction_labels": labels}

    # --- Analysis --------------------------------------------------------

    def assess_risk(self, state: AuditState) -> dict[str, Any]:
        """Narrate the deterministic findings; the model adds prose, not judgements."""
        anomalies = state.get("anomalies") or ["No rule-based anomalies detected."]
        labels = state.get("transaction_labels") or ["No ledger entries for this vendor."]
        reviewer_comment = state.get("reviewer_comment", "")

        user_prompt = (
            f"Vendor: {state['vendor']}\n\n"
            f"Classifier labels:\n" + "\n".join(labels) + "\n\n"
            f"Ledger rule findings:\n" + "\n".join(anomalies) + "\n\n"
            f"Contextual evidence:\n{state['evidence_context']}"
        )
        if reviewer_comment:
            user_prompt += f"\n\nReviewer feedback to address in this revision:\n{reviewer_comment}"

        findings = self._deps.llm.complete(prompts.RISK_ANALYST, user_prompt)
        return {"risk_findings": findings}

    def check_standards(self, state: AuditState) -> dict[str, Any]:
        """Compare the relationship against the client's stated policy."""
        findings = self._deps.llm.complete(prompts.STANDARDS_REVIEWER, state["evidence_context"])
        return {"standards_findings": findings}

    # --- Review gate -----------------------------------------------------

    def human_review(self, state: AuditState) -> dict[str, Any]:
        """Pause the graph until a reviewer approves or requests changes.

        Nothing reaches a memo without this decision being recorded in state.
        """
        response = interrupt(
            {
                "vendor": state["vendor"],
                "risk_findings": state.get("risk_findings", ""),
                "standards_findings": state.get("standards_findings", ""),
                "revision_count": state.get("revision_count", 0),
                "prompt": (
                    "Reply 'approved' to proceed to drafting, or 'revise' "
                    "(optionally with a comment) to send it back."
                ),
            }
        )

        decision, comment = _parse_review_response(response)
        revision_count = state.get("revision_count", 0)
        if decision == REVISE:
            revision_count += 1

        logger.info("Reviewer decision: %s (revision %d)", decision, revision_count)
        return {
            "review_decision": decision,
            "reviewer_comment": comment,
            "revision_count": revision_count,
        }

    def route_after_review(self, state: AuditState) -> str:
        """Send approved work to drafting; loop back on revision, up to the cap."""
        if state.get("review_decision") == APPROVED:
            return "draft_memo"
        if state.get("revision_count", 0) >= MAX_REVISIONS:
            logger.warning("Revision limit (%d) reached; proceeding to drafting", MAX_REVISIONS)
            return "draft_memo"
        return "assess_risk"

    # --- Output ----------------------------------------------------------

    def draft_memo(self, state: AuditState) -> dict[str, Any]:
        """Compose the memo from findings that a reviewer has already signed off."""
        memo = self._deps.llm.complete(
            prompts.MEMO_DRAFTER,
            (
                f"Engagement: {self._deps.client_name} - vendor review: {state['vendor']}\n\n"
                f"Risk findings:\n{state.get('risk_findings', '')}\n\n"
                f"Standards findings:\n{state.get('standards_findings', '')}\n\n"
                f"Reviewer decision: {state.get('review_decision', 'n/a')}"
            ),
        )
        return {"memo": memo}


def _parse_review_response(response: Any) -> tuple[str, str]:
    """Accept either a bare decision string or a ``{decision, comment}`` mapping."""
    if isinstance(response, dict):
        decision = str(response.get("decision", REVISE)).strip().lower()
        comment = str(response.get("comment", ""))
    else:
        decision = str(response).strip().lower()
        comment = ""
    return (APPROVED if decision == APPROVED else REVISE), comment
