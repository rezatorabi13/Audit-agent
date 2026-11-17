"""End-to-end vendor review, including the revision loop.

Demonstrates the approval gate: the graph pauses, a reviewer sends the work back
with a comment, then approves the second pass.

Uses the classifier selected by AUDIT_CLASSIFIER (rule-based by default). To
label transactions with the fine-tuned Qwen adapter instead, train it once and
set AUDIT_CLASSIFIER=qlora, or run examples/03_vendor_review_qlora.py.

Prerequisites:
    python examples/00_setup_corpus.py
    ollama pull llama3.2:3b               # default LLM; or set AUDIT_LLM_PROVIDER=anthropic

Run:
    python examples/02_vendor_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script puts examples/ on sys.path, not the repository root, so the
# package would not resolve without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_agent.config import get_settings  # noqa: E402
from audit_agent.logging_config import configure_logging  # noqa: E402
from audit_agent.pipeline import build_workflow  # noqa: E402
from audit_agent.reporting.memo import write_memo_docx  # noqa: E402
from audit_agent.workflow import ReviewDecision  # noqa: E402

VENDOR = "Acme Industrial Supplies Ltd."


def main() -> None:
    configure_logging()
    settings = get_settings()
    print(f"LLM provider: {settings.llm_provider} | classifier: {settings.classifier}")
    workflow = build_workflow(settings)

    pending = workflow.start(VENDOR, "Assess payment and related-party risk for this vendor.")
    print(f"\n{'=' * 70}\nPAUSED FOR REVIEW: {pending.vendor}\n")
    print(f"Risk findings:\n{pending.risk_findings}\n")
    print(f"Standards findings:\n{pending.standards_findings}")

    revised = workflow.resume(
        pending.thread_id,
        ReviewDecision.REVISE,
        "State the duplicate payment explicitly and name the shared director.",
    )
    print(f"\n{'=' * 70}\nREVISION {revised.revision_count}\n")
    print(revised.risk_findings)

    result = workflow.resume(revised.thread_id, ReviewDecision.APPROVED)
    print(f"\n{'=' * 70}\nAPPROVED MEMO\n")
    print(result["memo"])

    path = write_memo_docx(
        memo_text=result["memo"],
        output_path=settings.output_dir / "risk_memo_acme.docx",
        client_name=settings.client_name,
        vendor=VENDOR,
        reviewer_decision=result["review_decision"],
        evidence_citations=result.get("anomalies", []),
    )
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
