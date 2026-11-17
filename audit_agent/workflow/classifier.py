"""Transaction risk classification, behind a swappable interface.

Two implementations share one protocol:

* :class:`RuleBasedClassifier` (default) - deterministic, needs no model.
* ``finetuning.classifier.LoRAClassifier`` - Qwen2.5-3B-Instruct with a QLoRA
  adapter trained by ``python -m finetuning.train_qlora``.

Select one with ``AUDIT_CLASSIFIER=rule_based|qlora``; no node changes.
"""

from __future__ import annotations

from datetime import date as Date
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


def describe_transaction(
    *,
    vendor: str,
    amount: float,
    posted_on: str,
    memo: str = "",
    shared_directors: tuple[str, ...] | list[str] = (),
    appears_twice: bool = False,
) -> str:
    """Render one ledger entry as text for a model-based classifier.

    A classifier only sees the string it is given. A single ledger row cannot
    reveal a related party (that lives in the entity graph) or a duplicate
    (that needs the rest of the ledger), so those facts are added explicitly,
    phrased the way the fine-tuning examples phrase them.
    """
    try:
        weekday = Date.fromisoformat(str(posted_on)[:10]).strftime("%A")
        when = f"issued on a {weekday} ({posted_on})"
    except ValueError:
        when = f"posted on {posted_on}"

    text = f"Invoice from {vendor} for ${amount:,.2f}, {when}"
    if memo:
        text += f", described as '{memo}'"
    text += "."

    if appears_twice:
        text += " Another invoice from the same vendor for the same amount appears in the ledger."
    if shared_directors:
        text += (
            f" A shared board member ({', '.join(shared_directors)}) links this vendor "
            "to the client's leadership."
        )
    return text
