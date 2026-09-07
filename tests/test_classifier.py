"""Classification precedence: a related party outranks a rule hit."""

from __future__ import annotations

from audit_agent.workflow.classifier import CATEGORIES, RuleBasedClassifier, TransactionClassifier


def test_satisfies_the_classifier_protocol() -> None:
    assert isinstance(RuleBasedClassifier(), TransactionClassifier)


def test_related_party_takes_precedence_over_rule_hits() -> None:
    classifier = RuleBasedClassifier()
    assert classifier.classify_from_rules(["duplicate_payment"], True) == "Related Party"


def test_rules_map_to_their_categories() -> None:
    classifier = RuleBasedClassifier()
    assert classifier.classify_from_rules(["duplicate_payment"], False) == "Duplicate Payment Risk"
    assert (
        classifier.classify_from_rules(["weekend_posting"], False) == "Unusual Timing or Amount"
    )


def test_unflagged_transaction_is_normal() -> None:
    assert RuleBasedClassifier().classify_from_rules([], False) == "Normal"


def test_labels_stay_within_the_fixed_vocabulary() -> None:
    classifier = RuleBasedClassifier()
    labels = {
        classifier.classify_from_rules(rules, related)
        for rules in ([], ["duplicate_payment"], ["round_dollar_amount"])
        for related in (True, False)
    }
    assert labels <= set(CATEGORIES)


def test_description_matching_uses_the_related_party_roster() -> None:
    classifier = RuleBasedClassifier(related_party_vendors=frozenset({"Acme Industrial"}))
    assert classifier.classify("Invoice from Acme Industrial for $1,000") == "Related Party"
