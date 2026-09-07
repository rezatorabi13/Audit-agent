"""Labelled transaction descriptions for training and evaluating the classifier.

Examples are generated rather than drawn from client data, and the generator is
seeded so a training run is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from audit_agent.workflow.classifier import CATEGORIES

VENDORS = (
    "Acme Industrial Supplies Ltd.",
    "Globex Logistics Partners Inc.",
    "Initech Office Solutions",
    "Wayne Distribution Co.",
    "Stark Materials Inc.",
    "Umbrella Packaging Ltd.",
    "Soylent Freight Corp.",
)

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
WEEKEND_DAYS = ("Saturday", "Sunday")
RELATIONSHIPS = (
    "a shared board member",
    "the CFO's spouse",
    "a director who also owns the vendor",
)
ROUND_AMOUNTS = (10_000, 25_000, 50_000, 100_000)

SYSTEM_PROMPT = (
    "You are an audit risk classifier. Classify the transaction into exactly one of: "
    f"{', '.join(CATEGORIES)}. Respond with only the category name."
)


@dataclass(frozen=True, slots=True)
class LabelledTransaction:
    description: str
    label: str


def generate_examples(per_class: int = 20, seed: int = 0) -> list[LabelledTransaction]:
    """Build a balanced, shuffled set of labelled examples."""
    rng = random.Random(seed)
    examples: list[LabelledTransaction] = []

    for _ in range(per_class):
        amount = rng.randint(500, 9500) + rng.random()
        examples.append(
            LabelledTransaction(
                f"Invoice from {rng.choice(VENDORS)} for ${amount:,.2f}, issued on a "
                f"{rng.choice(WEEKDAYS)}, standard recurring purchase.",
                "Normal",
            )
        )

    for _ in range(per_class):
        vendor, amount = rng.choice(VENDORS), rng.randint(1000, 20000)
        number = rng.randint(1000, 9999)
        examples.append(
            LabelledTransaction(
                f"Two invoices from {vendor}, both numbered {number}, both for "
                f"${amount:,.2f}, same date - appears twice in the ledger.",
                "Duplicate Payment Risk",
            )
        )

    for _ in range(per_class):
        examples.append(
            LabelledTransaction(
                f"Transaction with {rng.choice(VENDORS)}, where {rng.choice(RELATIONSHIPS)} "
                f"links this vendor to the client's leadership.",
                "Related Party",
            )
        )

    for _ in range(per_class):
        examples.append(
            LabelledTransaction(
                f"Invoice from {rng.choice(VENDORS)} for a round amount of "
                f"${rng.choice(ROUND_AMOUNTS):,.2f}, issued on a {rng.choice(WEEKEND_DAYS)}.",
                "Unusual Timing or Amount",
            )
        )

    rng.shuffle(examples)
    return examples


def train_eval_split(
    examples: list[LabelledTransaction], train_fraction: float = 0.8
) -> tuple[list[LabelledTransaction], list[LabelledTransaction]]:
    split = int(len(examples) * train_fraction)
    return examples[:split], examples[split:]
