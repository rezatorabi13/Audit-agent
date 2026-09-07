"""Grounded question answering, with and without model-driven retrieval.

Prerequisites:
    python examples/00_setup_corpus.py

Run:
    python examples/01_grounded_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script puts examples/ on sys.path, not the repository root, so the
# package would not resolve without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_agent.evaluation.groundedness import GroundednessCheck  # noqa: E402
from audit_agent.llm.factory import get_provider  # noqa: E402
from audit_agent.logging_config import configure_logging  # noqa: E402
from audit_agent.pipeline import build_qa  # noqa: E402
from audit_agent.retrieval.base import build_context  # noqa: E402

QUESTIONS = [
    "What are the payment terms in the Acme Industrial Supplies contract?",
    "What does the revenue recognition policy say about related-party transactions?",
    "Does the Globex Logistics contract involve any related parties?",
]

# The first needs client evidence; the second the model can answer alone. The
# point is that it decides, rather than the caller deciding for it.
TOOL_USE_QUESTIONS = [
    "What does the Globex Logistics contract say about termination?",
    "What is 15% of 200?",
]


def main() -> None:
    configure_logging()
    qa = build_qa()
    checker = GroundednessCheck(get_provider())

    print(f"\n{'=' * 70}\nGROUNDED RETRIEVAL (always retrieves)\n{'=' * 70}")
    for question in QUESTIONS:
        answer = qa.ask(question)
        print(f"\nQ: {question}\n")
        print(answer.text)
        print(f"\nSources: {', '.join(answer.citations)}")

        result = checker.evaluate(answer.text, build_context(list(answer.sources)))
        print(f"Grounded: {result.is_grounded}")
        if not result.is_grounded:
            print(f"Critique: {result.critique}")

    print(f"\n{'=' * 70}\nTOOL USE (model decides whether to retrieve)\n{'=' * 70}")
    for question in TOOL_USE_QUESTIONS:
        answer = qa.ask_with_tool_use(question)
        route = "retrieved evidence" if answer.used_retrieval else "answered directly"
        print(f"\nQ: {question}\n   -> {route}\n")
        print(answer.text)
        if answer.citations:
            print(f"\nSources: {', '.join(answer.citations)}")


if __name__ == "__main__":
    main()