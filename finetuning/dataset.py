"""Labelled transaction descriptions for training and evaluating the classifier.

Examples are generated rather than drawn from client data, and the generator is
seeded so a training run is reproducible.

Every example is rendered with :func:`audit_agent.workflow.classifier.describe_transaction`,
the same function the workflow uses at inference time. Training and serving
therefore see identically formatted text, which is what lets a small adapter
generalise from a few hundred examples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from audit_agent.workflow.classifier import CATEGORIES, describe_transaction

VENDORS = (
    "Acme Industrial Supplies Ltd.",
    "Globex Logistics Partners Inc.",
    "Initech Office Solutions",
    "Wayne Distribution Co.",
    "Stark Materials Inc.",
    "Umbrella Packaging Ltd.",
    "Soylent Freight Corp.",
)

MEMOS = (
    "Packaging film",
    "Warehousing services",
    "Office supplies",
    "Freight charges",
    "Maintenance contract",
    "Raw materials",
    "IT support retainer",
)

DIRECTORS = ("Jane Whitfield", "Marcus Hale", "Priya Raman", "Tomas Ostrowski")
ROUND_AMOUNTS = (10_000, 25_000, 50_000, 100_000)
YEAR_START = date(2025, 1, 1)

SYSTEM_PROMPT = (
    "You are an audit risk classifier. Classify the transaction into exactly one of: "
    f"{', '.join(CATEGORIES)}. Respond with only the category name."
)


@dataclass(frozen=True, slots=True)
class LabelledTransaction:
    description: str
    label: str


def _random_date(rng: random.Random, weekend: bool) -> str:
    while True:
        day = YEAR_START + timedelta(days=rng.randint(0, 364))
        if (day.weekday() >= 5) == weekend:
            return day.isoformat()


def _irregular_amount(rng: random.Random) -> float:
    # Non-zero cents keep the amount from ever looking round.
    return rng.randint(500, 19_500) + rng.randint(1, 99) / 100


def generate_examples(per_class: int = 40, seed: int = 0) -> list[LabelledTransaction]:
    """Build a balanced, shuffled set of labelled examples.

    Label precedence matches the rule-based classifier: a related party
    outranks a duplicate, which outranks unusual timing or amount.
    """
    rng = random.Random(seed)
    examples: list[LabelledTransaction] = []

    def add(label: str, **facts: object) -> None:
        text = describe_transaction(
            vendor=rng.choice(VENDORS),
            memo=rng.choice(MEMOS),
            **facts,  # type: ignore[arg-type]
        )
        examples.append(LabelledTransaction(text, label))

    for _ in range(per_class):
        add("Normal", amount=_irregular_amount(rng), posted_on=_random_date(rng, weekend=False))

    for _ in range(per_class):
        add(
            "Duplicate Payment Risk",
            amount=_irregular_amount(rng),
            posted_on=_random_date(rng, weekend=rng.random() < 0.2),
            appears_twice=True,
        )

    for _ in range(per_class):
        add(
            "Related Party",
            amount=rng.choice((_irregular_amount(rng), float(rng.choice(ROUND_AMOUNTS)))),
            posted_on=_random_date(rng, weekend=rng.random() < 0.2),
            shared_directors=(rng.choice(DIRECTORS),),
            appears_twice=rng.random() < 0.2,
        )

    for _ in range(per_class):
        weekend, round_amount = rng.choice(((True, False), (False, True), (True, True)))
        add(
            "Unusual Timing or Amount",
            amount=float(rng.choice(ROUND_AMOUNTS)) if round_amount else _irregular_amount(rng),
            posted_on=_random_date(rng, weekend=weekend),
        )

    rng.shuffle(examples)
    return examples


def train_eval_split(
    examples: list[LabelledTransaction], train_fraction: float = 0.8
) -> tuple[list[LabelledTransaction], list[LabelledTransaction]]:
    split = int(len(examples) * train_fraction)
    return examples[:split], examples[split:]
