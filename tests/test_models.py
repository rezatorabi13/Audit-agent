"""The evidence bundle is what makes a drafted memo traceable."""

from __future__ import annotations

from audit_agent.models import (
    NO_PAGE,
    AnomalyFlag,
    EvidenceBundle,
    RelatedPartyFinding,
    RetrievedChunk,
)


def test_citation_includes_page_when_present() -> None:
    chunk = RetrievedChunk(text="t", source="contract.pdf", page=4, score=0.9)
    assert chunk.citation == "contract.pdf, p.4"


def test_citation_omits_page_for_pageless_sources() -> None:
    assert RetrievedChunk(text="t", source="policy.docx", page=None, score=0.5).citation == (
        "policy.docx"
    )
    assert RetrievedChunk(text="t", source="policy.docx", page=NO_PAGE, score=0.5).citation == (
        "policy.docx"
    )


def test_context_includes_graph_rules_and_citations() -> None:
    bundle = EvidenceBundle(
        vendor="Acme Industrial Supplies Ltd.",
        question="payment terms",
        related_party=RelatedPartyFinding(
            vendor="Acme Industrial Supplies Ltd.",
            is_related_party=True,
            shared_board_members=("Jane Whitfield",),
        ),
        chunks=(RetrievedChunk(text="Net 45.", source="contract.pdf", page=1, score=0.9),),
        anomalies=(AnomalyFlag(rule="duplicate_payment", transaction_id="JE-0002", detail="x"),),
    )

    context = bundle.as_context()
    assert "RELATED PARTY DETECTED" in context
    assert "Jane Whitfield" in context
    assert "duplicate_payment" in context
    assert "[Source: contract.pdf, p.1]" in context


def test_context_states_absence_of_findings_explicitly() -> None:
    bundle = EvidenceBundle(
        vendor="Globex Logistics Partners Inc.",
        question="terms",
        related_party=RelatedPartyFinding(
            vendor="Globex Logistics Partners Inc.", is_related_party=False
        ),
    )
    context = bundle.as_context()
    assert "no related-party link found" in context
    assert "none for this vendor" in context
