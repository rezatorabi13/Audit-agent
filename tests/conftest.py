"""Shared fixtures. Nothing here requires an API key or network access."""

from __future__ import annotations

import pandas as pd
import pytest

from audit_agent.llm.base import ToolDefinition, ToolRequest
from audit_agent.models import RetrievedChunk

CLIENT = "Northwind Retail Group Inc."
RELATED_VENDOR = "Acme Industrial Supplies Ltd."
ARMS_LENGTH_VENDOR = "Globex Logistics Partners Inc."


class StubLLM:
    """Records prompts and returns canned responses, so nodes stay testable."""

    name = "stub"

    def __init__(self, response: str = "stub response", tool_request: ToolRequest | None = None):
        self._response = response
        self._tool_request = tool_request
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response

    def supports_tools(self) -> bool:
        return self._tool_request is not None

    def decide_tool_use(self, user: str, tools: list[ToolDefinition]) -> ToolRequest:
        if self._tool_request is None:
            raise NotImplementedError
        return self._tool_request


class StubVectorStore:
    """In-memory stand-in for the Chroma-backed store."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        # An explicitly empty list means "empty index", so it must not fall
        # through to the default the way `chunks or [...]` would.
        self._chunks = chunks if chunks is not None else [
            RetrievedChunk(
                text="Client shall pay all undisputed invoices within forty-five (45) days.",
                source="vendor_contract_acme_supplies.pdf",
                page=1,
                score=0.82,
            )
        ]
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._chunks[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def stub_vector_store() -> StubVectorStore:
    return StubVectorStore()


@pytest.fixture
def ledger() -> pd.DataFrame:
    """Ledger containing one duplicate pair and one round-dollar weekend entry."""
    return pd.DataFrame(
        [
            ("JE-0001", "2025-02-10", "Packaging film", RELATED_VENDOR, 13140.00, 0.00),
            ("JE-0002", "2025-03-14", "Packaging film", RELATED_VENDOR, 13110.00, 0.00),
            ("JE-0003", "2025-03-14", "Packaging film (dup)", RELATED_VENDOR, 13110.00, 0.00),
            ("JE-0004", "2025-04-02", "March warehousing", ARMS_LENGTH_VENDOR, 8200.00, 0.00),
            ("JE-0006", "2025-04-19", "Furniture refresh", "Initech Office Solutions",
             50000.00, 0.00),
            # 2025-06-01 is a Sunday: a weekend posting on the credit side.
            ("JE-0010", "2025-06-01", "Customer receipt", "Various Customers", 0.00, 154200.00),
        ],
        columns=["transaction_id", "date", "description", "vendor", "debit", "credit"],
    )


@pytest.fixture
def related_parties() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (CLIENT, "transacts_with", RELATED_VENDOR, "Packaging vendor."),
            (CLIENT, "transacts_with", ARMS_LENGTH_VENDOR, "Logistics vendor."),
            ("Jane Whitfield", "board_member_of", CLIENT, "Independent director."),
            ("Jane Whitfield", "board_member_of", RELATED_VENDOR, "Also on vendor board."),
        ],
        columns=["entity_a", "relationship", "entity_b", "notes"],
    )
