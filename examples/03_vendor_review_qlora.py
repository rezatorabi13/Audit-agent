"""Vendor review with the QLoRA-fine-tuned Qwen classifier labelling transactions.

Same workflow as 02, with the classify_transactions node served by
Qwen2.5-3B-Instruct + the LoRA adapter you trained locally, and Llama 3.2
(via Ollama, by default) writing the narrative. Finishes with a groundedness
check of the approved memo against the evidence gathered.

Prerequisites (once):
    python examples/00_setup_corpus.py
    python -m finetuning.train_qlora      # writes data/models/qwen_risk_classifier/adapter
    ollama pull llama3.2:3b

Run:
    python examples/03_vendor_review_qlora.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script puts examples/ on sys.path, not the repository root, so the
# package would not resolve without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_agent.config import get_settings  # noqa: E402
from audit_agent.evaluation.groundedness import GroundednessCheck  # noqa: E402
from audit_agent.llm.factory import get_provider  # noqa: E402
from audit_agent.logging_config import configure_logging  # noqa: E402
from audit_agent.pipeline import build_workflow  # noqa: E402
from audit_agent.reporting.memo import write_memo_docx  # noqa: E402
from audit_agent.workflow import ReviewDecision  # noqa: E402

VENDOR = "Acme Industrial Supplies Ltd."


def main() -> None:
    configure_logging()
    settings = get_settings().model_copy(update={"classifier": "qlora"})

    print(f"LLM provider:   {settings.llm_provider}")
    print(f"Classifier:     QLoRA adapter at {settings.qlora_adapter_dir}")
    print(f"Base model:     {settings.qlora_base_model}\n")

    workflow = build_workflow(settings)

    pending = workflow.start(VENDOR, "Assess payment and related-party risk for this vendor.")
    state = workflow.state(pending.thread_id)

    print(f"{'=' * 70}\nQLoRA CLASSIFIER LABELS: {VENDOR}\n")
    for label in state.get("transaction_labels", []):
        print(f"  {label}")

    print(f"\n{'=' * 70}\nPAUSED FOR REVIEW\n")
    print(f"Risk findings:\n{pending.risk_findings}\n")
    print(f"Standards findings:\n{pending.standards_findings}")

    result = workflow.resume(pending.thread_id, ReviewDecision.APPROVED)
    print(f"\n{'=' * 70}\nAPPROVED MEMO\n")
    print(result["memo"])

    evidence = (
        f"Classifier labels:\n{chr(10).join(result.get('transaction_labels', []))}\n\n"
        f"Ledger rule findings:\n{chr(10).join(result.get('anomalies', []))}\n\n"
        f"Related-party check: {result.get('related_party')}\n\n"
        f"Risk findings:\n{result.get('risk_findings', '')}\n\n"
        f"Standards findings:\n{result.get('standards_findings', '')}"
    )
    check = GroundednessCheck(get_provider(settings)).evaluate(result["memo"], evidence)
    print(f"\n{'=' * 70}\nGROUNDEDNESS: {'pass' if check.is_grounded else 'review needed'}")
    if not check.is_grounded:
        print(check.critique)

    path = write_memo_docx(
        memo_text=result["memo"],
        output_path=settings.output_dir / "risk_memo_acme_qlora.docx",
        client_name=settings.client_name,
        vendor=VENDOR,
        reviewer_decision=result["review_decision"],
        evidence_citations=result.get("anomalies", []),
    )
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
