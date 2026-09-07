"""Grounded answering and the tool-use routing decision."""

from __future__ import annotations

import pytest

from audit_agent.evaluation.groundedness import GroundednessCheck
from audit_agent.llm.base import ToolRequest, ToolUseNotSupportedError
from audit_agent.qa import GroundedQA

from .conftest import StubLLM, StubVectorStore


def test_answer_carries_its_citations(
    stub_vector_store: StubVectorStore, stub_llm: StubLLM
) -> None:
    answer = GroundedQA(stub_vector_store, stub_llm).ask("What are the payment terms?")  # type: ignore[arg-type]

    assert answer.citations == ["vendor_contract_acme_supplies.pdf, p.1"]
    assert answer.used_retrieval


def test_retrieved_context_is_passed_to_the_model(
    stub_vector_store: StubVectorStore, stub_llm: StubLLM
) -> None:
    GroundedQA(stub_vector_store, stub_llm).ask("Payment terms?")  # type: ignore[arg-type]

    _, user_prompt = stub_llm.calls[0]
    assert "forty-five (45) days" in user_prompt


def test_empty_index_does_not_reach_the_model(stub_llm: StubLLM) -> None:
    answer = GroundedQA(StubVectorStore(chunks=[]), stub_llm).ask("Anything?")  # type: ignore[arg-type]

    assert stub_llm.calls == []
    assert "No indexed documents" in answer.text


def test_tool_use_skips_retrieval_when_the_model_declines(
    stub_vector_store: StubVectorStore,
) -> None:
    llm = StubLLM(tool_request=ToolRequest(tool_name=None, tool_input={}, text="15% of 200 is 30."))
    answer = GroundedQA(stub_vector_store, llm).ask_with_tool_use("What is 15% of 200?")  # type: ignore[arg-type]

    assert not answer.used_retrieval
    assert answer.sources == ()
    assert stub_vector_store.queries == []


def test_tool_use_retrieves_with_the_model_supplied_query(
    stub_vector_store: StubVectorStore,
) -> None:
    llm = StubLLM(
        tool_request=ToolRequest(
            tool_name="search_audit_documents",
            tool_input={"query": "Globex termination clause"},
            text="",
        )
    )
    answer = GroundedQA(stub_vector_store, llm).ask_with_tool_use("Termination terms?")  # type: ignore[arg-type]

    assert stub_vector_store.queries == ["Globex termination clause"]
    assert answer.used_retrieval


def test_tool_use_rejected_by_providers_without_support(
    stub_vector_store: StubVectorStore, stub_llm: StubLLM
) -> None:
    with pytest.raises(ToolUseNotSupportedError):
        GroundedQA(stub_vector_store, stub_llm).ask_with_tool_use("Anything?")  # type: ignore[arg-type]


def test_groundedness_passes_on_the_exact_marker() -> None:
    result = GroundednessCheck(StubLLM("All statements supported.")).evaluate("draft", "evidence")
    assert result.is_grounded
    assert bool(result) is True


def test_groundedness_fails_when_claims_are_listed() -> None:
    critique = "The claim that terms are Net 30 is not supported."
    result = GroundednessCheck(StubLLM(critique)).evaluate("draft", "evidence")

    assert not result.is_grounded
    assert result.critique == critique
