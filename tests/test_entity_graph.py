"""Related-party traversal is the check a vector search cannot make reliably."""

from __future__ import annotations

import pandas as pd
import pytest

from audit_agent.knowledge.entity_graph import LEDGER_ACTIVITY, EntityGraph

from .conftest import ARMS_LENGTH_VENDOR, CLIENT, RELATED_VENDOR


@pytest.fixture
def graph(related_parties: pd.DataFrame, ledger: pd.DataFrame) -> EntityGraph:
    return EntityGraph.from_dataframes(CLIENT, related_parties, ledger)


def test_detects_shared_board_member(graph: EntityGraph) -> None:
    finding = graph.check_related_party(RELATED_VENDOR)
    assert finding.is_related_party
    assert finding.shared_board_members == ("Jane Whitfield",)


def test_arms_length_vendor_is_not_flagged(graph: EntityGraph) -> None:
    finding = graph.check_related_party(ARMS_LENGTH_VENDOR)
    assert not finding.is_related_party
    assert finding.shared_board_members == ()


def test_unknown_vendor_is_not_flagged(graph: EntityGraph) -> None:
    assert not graph.check_related_party("Nonexistent Ltd.").is_related_party


def test_ledger_activity_is_a_distinct_edge_type(graph: EntityGraph) -> None:
    edge = graph.graph.get_edge_data(CLIENT, "Initech Office Solutions")
    assert edge["relationship"] == LEDGER_ACTIVITY
    assert edge["total_debit"] == 50000.00
    assert edge["transaction_count"] == 1


def test_aggregate_placeholders_are_excluded(graph: EntityGraph) -> None:
    assert "Various Customers" not in graph.graph.nodes


def test_vendors_lists_transacting_counterparties(graph: EntityGraph) -> None:
    assert RELATED_VENDOR in graph.vendors()
    assert "Jane Whitfield" not in graph.vendors()


def test_finding_summary_names_the_shared_director(graph: EntityGraph) -> None:
    summary = graph.check_related_party(RELATED_VENDOR).summary()
    assert "RELATED PARTY DETECTED" in summary
    assert "Jane Whitfield" in summary


def test_rejects_table_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        EntityGraph.from_dataframes(CLIENT, pd.DataFrame({"entity_a": ["X"]}))
