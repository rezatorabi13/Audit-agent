"""The workflow must not reach a memo without a recorded human decision."""

from __future__ import annotations

import pandas as pd
import pytest

from audit_agent.knowledge.entity_graph import EntityGraph
from audit_agent.retrieval.hybrid import HybridRetriever
from audit_agent.workflow.graph import AuditWorkflow, PendingReview, ReviewDecision
from audit_agent.workflow.nodes import MAX_REVISIONS, WorkflowDependencies

from .conftest import ARMS_LENGTH_VENDOR, CLIENT, RELATED_VENDOR, StubLLM, StubVectorStore


@pytest.fixture
def workflow(
    ledger: pd.DataFrame,
    related_parties: pd.DataFrame,
    stub_llm: StubLLM,
    stub_vector_store: StubVectorStore,
) -> AuditWorkflow:
    retriever = HybridRetriever(
        vector_store=stub_vector_store,  # type: ignore[arg-type]
        entity_graph=EntityGraph.from_dataframes(CLIENT, related_parties, ledger),
        ledger=ledger,
    )
    deps = WorkflowDependencies(
        retriever=retriever,
        llm=stub_llm,
        ledger=ledger,
        client_name=CLIENT,
    )
    return AuditWorkflow(deps)


def test_graph_pauses_at_the_review_gate(workflow: AuditWorkflow) -> None:
    pending = workflow.start(RELATED_VENDOR, "Assess risk.")

    assert isinstance(pending, PendingReview)
    assert pending.vendor == RELATED_VENDOR
    assert pending.risk_findings
    assert workflow.state(pending.thread_id).get("memo") is None


def test_approval_produces_a_memo(workflow: AuditWorkflow) -> None:
    pending = workflow.start(RELATED_VENDOR, "Assess risk.")
    result = workflow.resume(pending.thread_id, ReviewDecision.APPROVED)

    assert isinstance(result, dict)
    assert result["memo"]
    assert result["review_decision"] == "approved"


def test_revision_loops_back_and_pauses_again(workflow: AuditWorkflow) -> None:
    pending = workflow.start(RELATED_VENDOR, "Assess risk.")
    revised = workflow.resume(pending.thread_id, ReviewDecision.REVISE, "Tighten the wording.")

    assert isinstance(revised, PendingReview)
    assert revised.revision_count == 1


def test_reviewer_comment_reaches_the_risk_prompt(
    workflow: AuditWorkflow, stub_llm: StubLLM
) -> None:
    pending = workflow.start(RELATED_VENDOR, "Assess risk.")
    workflow.resume(pending.thread_id, ReviewDecision.REVISE, "Mention the duplicate explicitly.")

    prompts = [user for _, user in stub_llm.calls]
    assert any("Mention the duplicate explicitly." in prompt for prompt in prompts)


def test_revision_limit_forces_progress_to_drafting(workflow: AuditWorkflow) -> None:
    result = workflow.start(RELATED_VENDOR, "Assess risk.")
    for _ in range(MAX_REVISIONS):
        result = workflow.resume(result.thread_id, ReviewDecision.REVISE)
        if isinstance(result, dict):
            break
    else:
        result = workflow.resume(result.thread_id, ReviewDecision.REVISE)

    assert isinstance(result, dict), "graph should draft once the revision cap is reached"
    assert result["memo"]


def test_related_party_vendor_is_labelled_as_such(workflow: AuditWorkflow) -> None:
    pending = workflow.start(RELATED_VENDOR, "Assess risk.")
    state = workflow.state(pending.thread_id)

    assert state["related_party"]["is_related_party"] is True
    assert all("Related Party" in label for label in state["transaction_labels"])


def test_clean_vendor_yields_no_anomalies(workflow: AuditWorkflow) -> None:
    pending = workflow.start(ARMS_LENGTH_VENDOR, "Assess risk.")
    state = workflow.state(pending.thread_id)

    assert state["anomalies"] == []
    assert state["related_party"]["is_related_party"] is False


def test_workflow_exposes_its_node_topology(workflow: AuditWorkflow) -> None:
    assert {"gather_evidence", "human_review", "draft_memo"} <= set(workflow.node_names)
