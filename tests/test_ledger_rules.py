"""Ledger rules must be exact: these findings go into a memo unmodified."""

from __future__ import annotations

import pandas as pd
import pytest

from audit_agent.knowledge.ledger_rules import (
    detect_anomalies,
    duplicate_payments,
    round_dollar_amounts,
    weekend_postings,
)

from .conftest import ARMS_LENGTH_VENDOR, RELATED_VENDOR


def test_duplicate_payments_flags_both_sides_of_the_pair(ledger: pd.DataFrame) -> None:
    flags = duplicate_payments(ledger)
    assert {flag.transaction_id for flag in flags} == {"JE-0002", "JE-0003"}


def test_duplicate_payments_ignores_credits(ledger: pd.DataFrame) -> None:
    assert all(flag.transaction_id != "JE-0010" for flag in duplicate_payments(ledger))


def test_round_dollar_only_flags_large_exact_thousands(ledger: pd.DataFrame) -> None:
    flags = round_dollar_amounts(ledger)
    assert [flag.transaction_id for flag in flags] == ["JE-0006"]


def test_round_dollar_ignores_small_round_amounts() -> None:
    small = pd.DataFrame(
        [("JE-1", "2025-01-06", "Supplies", "V", 5000.00, 0.00)],
        columns=["transaction_id", "date", "description", "vendor", "debit", "credit"],
    )
    assert round_dollar_amounts(small) == []


def test_weekend_posting_detects_both_weekend_days(ledger: pd.DataFrame) -> None:
    """JE-0006 is a Saturday, JE-0010 a Sunday; credits are in scope too."""
    flags = weekend_postings(ledger)
    assert {flag.transaction_id for flag in flags} == {"JE-0006", "JE-0010"}


def test_weekday_entries_are_not_flagged(ledger: pd.DataFrame) -> None:
    flagged = {flag.transaction_id for flag in weekend_postings(ledger)}
    assert "JE-0001" not in flagged  # 2025-02-10 is a Monday


def test_detect_anomalies_scopes_to_vendor(ledger: pd.DataFrame) -> None:
    flags = detect_anomalies(ledger, vendor=RELATED_VENDOR)
    assert {flag.rule for flag in flags} == {"duplicate_payment"}


def test_detect_anomalies_returns_empty_for_clean_vendor(ledger: pd.DataFrame) -> None:
    assert detect_anomalies(ledger, vendor=ARMS_LENGTH_VENDOR) == []


def test_detect_anomalies_returns_empty_for_unknown_vendor(ledger: pd.DataFrame) -> None:
    assert detect_anomalies(ledger, vendor="Nonexistent Ltd.") == []


def test_detect_anomalies_rejects_malformed_ledger() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        detect_anomalies(pd.DataFrame({"transaction_id": ["JE-1"]}))
