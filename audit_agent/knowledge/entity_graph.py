"""Directed graph of engagement entities, used for related-party traversal.

Relationship questions ("does this vendor share a director with the client?")
are traversals, not similarity matches. Answering them on a graph gives a
deterministic, explainable result where vector search would only guess.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from audit_agent.models import RelatedPartyFinding

logger = logging.getLogger(__name__)

BOARD_MEMBERSHIP = "board_member_of"
LEDGER_ACTIVITY = "gl_transacts_with"

# Ledger placeholders that are aggregates, not counterparties.
NON_ENTITY_VENDORS = frozenset({"Various", "Various Customers"})

REQUIRED_COLUMNS = frozenset({"entity_a", "relationship", "entity_b"})


class EntityGraph:
    """Disclosed relationships plus observed ledger activity, in one graph."""

    def __init__(self, client_name: str, graph: nx.DiGraph | None = None) -> None:
        self.client_name = client_name
        self.graph = graph if graph is not None else nx.DiGraph()

    @classmethod
    def from_dataframes(
        cls,
        client_name: str,
        related_parties: pd.DataFrame,
        ledger: pd.DataFrame | None = None,
    ) -> EntityGraph:
        """Build from the related-party register, enriched with ledger totals."""
        missing = REQUIRED_COLUMNS - set(related_parties.columns)
        if missing:
            raise ValueError(f"Related-party table missing columns: {sorted(missing)}")

        instance = cls(client_name)
        instance._add_disclosed_relationships(related_parties)
        if ledger is not None:
            instance._add_ledger_activity(ledger)

        logger.info(
            "Entity graph built: %d node(s), %d edge(s)",
            instance.graph.number_of_nodes(),
            instance.graph.number_of_edges(),
        )
        return instance

    @classmethod
    def from_csv(
        cls,
        client_name: str,
        related_party_path: Path,
        ledger_path: Path | None = None,
    ) -> EntityGraph:
        related = pd.read_csv(related_party_path)
        ledger = pd.read_csv(ledger_path) if ledger_path and ledger_path.exists() else None
        return cls.from_dataframes(client_name, related, ledger)

    def _add_disclosed_relationships(self, table: pd.DataFrame) -> None:
        for row in table.itertuples(index=False):
            self.graph.add_edge(
                row.entity_a,
                row.entity_b,
                relationship=row.relationship,
                notes=getattr(row, "notes", ""),
            )

    def _add_ledger_activity(self, ledger: pd.DataFrame) -> None:
        """Attach observed spend per vendor as its own edge type.

        Disclosed relationships and observed activity are kept as separate
        edges so a reviewer can tell what the client declared from what the
        ledger actually shows.
        """
        spend = ledger[ledger["debit"] > 0]
        totals = spend.groupby("vendor")["debit"].agg(total="sum", count="count")

        for vendor, row in totals.iterrows():
            if vendor in NON_ENTITY_VENDORS:
                continue
            self.graph.add_edge(
                self.client_name,
                str(vendor),
                relationship=LEDGER_ACTIVITY,
                total_debit=float(row["total"]),
                transaction_count=int(row["count"]),
            )

    def check_related_party(self, vendor: str) -> RelatedPartyFinding:
        """Find people sitting on the board of both the client and ``vendor``."""
        shared = [
            person
            for person in self.graph.nodes
            if self._is_board_member(person, self.client_name)
            and self._is_board_member(person, vendor)
        ]
        return RelatedPartyFinding(
            vendor=vendor,
            is_related_party=bool(shared),
            shared_board_members=tuple(sorted(shared)),
        )

    def _is_board_member(self, person: str, organisation: str) -> bool:
        edge = self.graph.get_edge_data(person, organisation)
        return bool(edge) and edge.get("relationship") == BOARD_MEMBERSHIP

    def vendors(self) -> list[str]:
        """Counterparties the client transacts with, per graph or ledger."""
        transacting = {"transacts_with", LEDGER_ACTIVITY}
        return sorted(
            {
                target
                for _, target, data in self.graph.out_edges(self.client_name, data=True)
                if data.get("relationship") in transacting
            }
        )

    def describe(self) -> list[str]:
        """Human-readable edge list, for CLI output and audit trails."""
        return [
            f"{source} --[{data['relationship']}]--> {target}"
            for source, target, data in self.graph.edges(data=True)
        ]
