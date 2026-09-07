"""Deterministic anomaly rules over the general ledger.

Duplicate payments, round-dollar amounts and weekend postings are exactly
computable. Running them as rules rather than asking a model to eyeball a
table keeps the findings reproducible, and gives the drafting step grounded
facts instead of impressions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from audit_agent.models import AnomalyFlag

LedgerRule = Callable[[pd.DataFrame], list[AnomalyFlag]]

ROUND_AMOUNT_THRESHOLD = 10_000
ROUND_AMOUNT_MODULUS = 1_000
SATURDAY = 5

REQUIRED_COLUMNS = frozenset({"transaction_id", "date", "vendor", "debit"})


def duplicate_payments(ledger: pd.DataFrame) -> list[AnomalyFlag]:
    """Same vendor and same amount appearing more than once."""
    debits = ledger[ledger["debit"] > 0]
    duplicated = debits.duplicated(subset=["vendor", "debit"], keep=False)
    return [
        AnomalyFlag(
            rule="duplicate_payment",
            transaction_id=str(row.transaction_id),
            detail=(
                f"{row.vendor} charged ${row.debit:,.2f} on {row.date}; "
                f"an identical vendor/amount pair also appears in the ledger"
            ),
        )
        for row in debits[duplicated].itertuples(index=False)
    ]


def round_dollar_amounts(ledger: pd.DataFrame) -> list[AnomalyFlag]:
    """Large amounts landing on an exact thousand, a common estimate marker."""
    mask = (ledger["debit"] >= ROUND_AMOUNT_THRESHOLD) & (
        ledger["debit"] % ROUND_AMOUNT_MODULUS == 0
    )
    return [
        AnomalyFlag(
            rule="round_dollar_amount",
            transaction_id=str(row.transaction_id),
            detail=f"${row.debit:,.2f} posted on {row.date} is an exact round amount",
        )
        for row in ledger[mask].itertuples(index=False)
    ]


def weekend_postings(ledger: pd.DataFrame) -> list[AnomalyFlag]:
    """Entries dated on a Saturday or Sunday, outside normal posting cycles."""
    weekday = pd.to_datetime(ledger["date"]).dt.weekday
    return [
        AnomalyFlag(
            rule="weekend_posting",
            transaction_id=str(row.transaction_id),
            detail=f"entry dated {row.date}, which falls on a weekend",
        )
        for row in ledger[weekday >= SATURDAY].itertuples(index=False)
    ]


DEFAULT_RULES: tuple[LedgerRule, ...] = (
    duplicate_payments,
    round_dollar_amounts,
    weekend_postings,
)


def detect_anomalies(
    ledger: pd.DataFrame,
    vendor: str | None = None,
    rules: Sequence[LedgerRule] = DEFAULT_RULES,
) -> list[AnomalyFlag]:
    """Run every rule, optionally scoped to a single vendor."""
    missing = REQUIRED_COLUMNS - set(ledger.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {sorted(missing)}")

    scoped = ledger if vendor is None else ledger[ledger["vendor"] == vendor]
    if scoped.empty:
        return []

    flags: list[AnomalyFlag] = []
    for rule in rules:
        flags.extend(rule(scoped))
    return flags
