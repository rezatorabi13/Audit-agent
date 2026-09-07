"""Hybrid retrieval: a structured graph lookup fused with vector search.

"Hybrid" here means combining two kinds of reasoning, not running a fancier
vector search. The graph answers relationship questions exactly; the vector
index supplies the supporting clause text.
"""

from __future__ import annotations

import logging

import pandas as pd

from audit_agent.knowledge.entity_graph import EntityGraph
from audit_agent.knowledge.ledger_rules import detect_anomalies
from audit_agent.models import EvidenceBundle
from audit_agent.retrieval.base import SearchIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Assembles the full evidence bundle for one vendor question."""

    def __init__(
        self,
        vector_store: SearchIndex,
        entity_graph: EntityGraph,
        ledger: pd.DataFrame,
        top_k: int = 4,
    ) -> None:
        self._vector_store = vector_store
        self._entity_graph = entity_graph
        self._ledger = ledger
        self._top_k = top_k

    def gather(self, vendor: str, question: str) -> EvidenceBundle:
        related_party = self._entity_graph.check_related_party(vendor)
        anomalies = detect_anomalies(self._ledger, vendor=vendor)
        chunks = self._vector_store.search(f"{vendor} {question}", top_k=self._top_k)

        logger.info(
            "Evidence for %s: related_party=%s, %d anomaly flag(s), %d chunk(s)",
            vendor,
            related_party.is_related_party,
            len(anomalies),
            len(chunks),
        )
        return EvidenceBundle(
            vendor=vendor,
            question=question,
            related_party=related_party,
            chunks=tuple(chunks),
            anomalies=tuple(anomalies),
        )
