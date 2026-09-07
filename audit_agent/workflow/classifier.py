"""Transaction risk classification, behind a swappable interface.

The default implementation is rule-derived and needs no model. The optional
fine-tuned adapter in ``finetuning/`` implements the same protocol, so it can
be dropped into the workflow without touching any node.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

CATEGORIES = (
    "Normal",
    "Duplicate Payment Risk",
    "Related Party",
    "Unusual Timing or Amount",
)


@runtime_checkable
class TransactionClassifier(Protocol):
    """Assigns one fixed category label to a transaction description."""

    name: str

    def classify(self, description: str) -> str:
        ...


class RuleBasedClassifier:
    """Maps ledger rule hits to categories.

    Deterministic and free; used as the default so the workflow runs without a
    GPU or a second model.
    """

    name = "rule_based"

    RULE_TO_CATEGORY = {
        "duplicate_payment": "Duplicate Payment Risk",
        "round_dollar_amount": "Unusual Timing or Amount",
        "weekend_posting": "Unusual Timing or Amount",
    }

    def __init__(self, related_party_vendors: frozenset[str] | None = None) -> None:
        self._related_party_vendors = related_party_vendors or frozenset()

    def classify(self, description: str) -> str:
        lowered = description.lower()
        if any(vendor.lower() in lowered for vendor in self._related_party_vendors):
            return "Related Party"
        for rule, category in self.RULE_TO_CATEGORY.items():
            if rule.replace("_", " ") in lowered:
                return category
        return "Normal"

    def classify_from_rules(self, rules: list[str], vendor_is_related: bool) -> str:
        """Label a transaction directly from the rules that fired against it."""
        if vendor_is_related:
            return "Related Party"
        for rule in rules:
            if rule in self.RULE_TO_CATEGORY:
                return self.RULE_TO_CATEGORY[rule]
        return "Normal"
