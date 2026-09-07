"""Generates the synthetic engagement corpus the pipeline runs against.

Every entity, document and figure here is fictional. Using generated evidence
keeps the repository runnable end to end without exposing any client data, and
plants known anomalies (a duplicate invoice, a round-dollar weekend posting, an
undisclosed shared director) so retrieval and rule coverage can be verified
against a documented ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from docx import Document

logger = logging.getLogger(__name__)

CLIENT = "Northwind Retail Group Inc."

# --- Ground truth -----------------------------------------------------------
# The anomalies deliberately planted below, for use in tests and demos.
PLANTED_ANOMALIES = {
    "duplicate_payment": "JE-0002 and JE-0003 repeat the same Acme charge of $13,110.00.",
    "round_dollar_amount": "JE-0006 posts exactly $50,000.00 to office expense.",
    "weekend_posting": "JE-0006 is dated 2025-04-19, a Saturday.",
    "related_party": "Jane Whitfield sits on the board of both the client and Acme.",
}

INVOICES = [
    ("acme_inv_1001.pdf", "Acme Industrial Supplies Ltd.", "AC-1001", "2025-02-10",
     [("Packaging film, 500 rolls", 12500.00), ("Freight", 640.00)], 13140.00, None),
    ("acme_inv_1002.pdf", "Acme Industrial Supplies Ltd.", "AC-1002", "2025-03-14",
     [("Packaging film, 500 rolls", 12500.00), ("Freight", 610.00)], 13110.00, None),
    ("acme_inv_1002_dup.pdf", "Acme Industrial Supplies Ltd.", "AC-1002", "2025-03-14",
     [("Packaging film, 500 rolls", 12500.00), ("Freight", 610.00)], 13110.00,
     "Duplicate invoice number and amount of AC-1002."),
    ("globex_inv_2201.pdf", "Globex Logistics Partners Inc.", "GX-2201", "2025-04-02",
     [("Warehousing, March", 8200.00), ("Freight forwarding", 3450.00)], 11650.00, None),
    ("globex_inv_2202.pdf", "Globex Logistics Partners Inc.", "GX-2202", "2025-05-03",
     [("Warehousing, April", 8200.00), ("Freight forwarding", 3900.00)], 12100.00, None),
    ("initech_inv_0087.pdf", "Initech Office Solutions", "IN-0087", "2025-04-19",
     [("Office furniture refresh", 50000.00)], 50000.00,
     "Round-dollar amount issued on a Saturday."),
]

LEDGER_ROWS = [
    ("JE-0001", "2025-02-10", "5010-Supplies Expense", "Packaging film purchase",
     "Acme Industrial Supplies Ltd.", 13140.00, 0.00, "mchen"),
    ("JE-0002", "2025-03-14", "5010-Supplies Expense", "Packaging film purchase",
     "Acme Industrial Supplies Ltd.", 13110.00, 0.00, "mchen"),
    ("JE-0003", "2025-03-14", "5010-Supplies Expense", "Packaging film purchase (dup)",
     "Acme Industrial Supplies Ltd.", 13110.00, 0.00, "mchen"),
    ("JE-0004", "2025-04-02", "5020-Warehousing Expense", "March warehousing",
     "Globex Logistics Partners Inc.", 8200.00, 0.00, "jkumar"),
    ("JE-0005", "2025-04-05", "5030-Freight Expense", "March freight forwarding",
     "Globex Logistics Partners Inc.", 3450.00, 0.00, "jkumar"),
    ("JE-0006", "2025-04-19", "6010-Office Expense", "Office furniture refresh",
     "Initech Office Solutions", 50000.00, 0.00, "mchen"),
    ("JE-0007", "2025-05-03", "5020-Warehousing Expense", "April warehousing",
     "Globex Logistics Partners Inc.", 8200.00, 0.00, "jkumar"),
    ("JE-0008", "2025-05-05", "5030-Freight Expense", "April freight forwarding",
     "Globex Logistics Partners Inc.", 3900.00, 0.00, "jkumar"),
    ("JE-0009", "2025-05-20", "6020-Travel Expense", "Client entertainment",
     "Various", 9980.00, 0.00, "aross"),
    ("JE-0010", "2025-06-01", "1000-Cash", "Customer receipt",
     "Various Customers", 0.00, 154200.00, "system"),
    ("JE-0011", "2025-06-12", "5010-Supplies Expense", "Packaging film purchase",
     "Acme Industrial Supplies Ltd.", 12980.00, 0.00, "mchen"),
    ("JE-0012", "2025-06-14", "6010-Office Expense", "Office supplies",
     "Initech Office Solutions", 875.50, 0.00, "aross"),
]

RELATED_PARTY_ROWS = [
    (CLIENT, "parent_of", "Northwind Properties Ltd.", "Wholly owned real-estate subsidiary."),
    (CLIENT, "transacts_with", "Acme Industrial Supplies Ltd.", "Packaging vendor; Net 45 terms."),
    (CLIENT, "transacts_with", "Globex Logistics Partners Inc.",
     "Logistics vendor; arm's length, Net 30 terms."),
    (CLIENT, "transacts_with", "Initech Office Solutions", "Office supplies vendor."),
    ("Jane Whitfield", "board_member_of", CLIENT, "Independent director."),
    ("Jane Whitfield", "board_member_of", "Acme Industrial Supplies Ltd.",
     "Also a vendor board member - related party."),
]


def _write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "PDF generation requires fpdf2. Install it with: "
            "pip install -r requirements.txt"
        ) from exc

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    for paragraph in paragraphs:
        pdf.multi_cell(0, 6, paragraph)
        pdf.ln(2)
    pdf.output(str(path))
    logger.debug("Wrote %s", path.name)


def _acme_contract(docs_dir: Path) -> None:
    """Vendor contract carrying the disclosed shared-director clause."""
    _write_pdf(
        docs_dir / "contracts" / "vendor_contract_acme_supplies.pdf",
        "MASTER SUPPLY AGREEMENT",
        [
            f'This Master Supply Agreement ("Agreement") is entered into as of January 15, '
            f'2025, between {CLIENT} ("Client") and Acme Industrial Supplies Ltd. ("Vendor").',
            "1. TERM. This Agreement commences on the Effective Date and continues for an "
            "initial period of 24 months, renewing automatically for successive 12-month "
            "periods unless either party provides 60 days' written notice of non-renewal.",
            "2. PAYMENT TERMS. Client shall pay all undisputed invoices within forty-five "
            "(45) days of the invoice date (Net 45). Late payments accrue interest at 1.5% "
            "per month.",
            "3. PRICING. Vendor shall supply industrial packaging materials at the unit "
            "prices set out in Schedule A. Pricing is fixed for the first 12 months and "
            "subject to renegotiation thereafter.",
            "4. RELATED PARTY DISCLOSURE. The parties acknowledge that Ms. Jane Whitfield "
            "serves as a member of the Board of Directors of both Client and Vendor. This "
            "relationship has been disclosed to Client's Audit Committee in accordance with "
            "Client's related-party transaction policy.",
            "5. TERMINATION. Either party may terminate this Agreement for material breach "
            "upon 30 days' written notice if the breach is not cured within that period.",
            "6. GOVERNING LAW. This Agreement is governed by the laws of the Province of "
            "Ontario.",
        ],
    )


def _globex_contract(docs_dir: Path) -> None:
    """Arm's-length comparator contract, so retrieval must discriminate."""
    document = Document()
    document.add_heading("LOGISTICS SERVICES AGREEMENT", level=1)
    document.add_paragraph(
        f'This Logistics Services Agreement ("Agreement") is entered into as of March 1, '
        f'2025, between {CLIENT} ("Client") and Globex Logistics Partners Inc. ("Vendor"), '
        f"an arm's-length third party."
    )
    sections = [
        ("1. Term", "This Agreement is effective for 12 months from the Effective Date and "
                    "may be renewed by mutual written consent."),
        ("2. Payment Terms", "Client shall pay all undisputed invoices within thirty (30) "
                             "days of the invoice date (Net 30). Invoices disputed in good "
                             "faith may be withheld pending resolution."),
        ("3. Services and Fees", "Vendor shall provide warehousing and freight forwarding "
                                 "services at the rates set out in Schedule B, billed "
                                 "monthly in arrears."),
        ("4. Related Parties", "The parties confirm that no director, officer, or major "
                               "shareholder of Client holds any ownership interest in or "
                               "board position with Vendor."),
        ("5. Termination", "Either party may terminate this Agreement for convenience upon "
                           "90 days' written notice."),
    ]
    for heading, body in sections:
        document.add_heading(heading, level=2)
        document.add_paragraph(body)
    document.save(str(docs_dir / "contracts" / "vendor_contract_globex_logistics.docx"))


def _invoices(docs_dir: Path) -> None:
    for filename, vendor, number, invoice_date, items, total, note in INVOICES:
        lines = [
            f"Bill To: {CLIENT}",
            f"Vendor: {vendor}",
            f"Invoice Number: {number}",
            f"Invoice Date: {invoice_date}",
            "",
            "Line Items:",
        ]
        lines += [f"  - {description}: ${amount:,.2f}" for description, amount in items]
        lines += ["", f"Total Due: ${total:,.2f}"]
        if note:
            lines += ["", f"Internal Note: {note}"]
        _write_pdf(docs_dir / "invoices" / filename, "INVOICE", lines)


def _revenue_policy(docs_dir: Path) -> None:
    """Policy excerpt the standards-check step retrieves against."""
    document = Document()
    document.add_heading(f"{CLIENT} - Revenue Recognition Policy (Excerpt)", level=1)
    document.add_paragraph(
        "This policy summarises how the Company recognises revenue from contracts with "
        "customers in accordance with the five-step model."
    )
    steps = [
        ("Step 1: Identify the Contract",
         "A contract exists when both parties have approved the arrangement, each party's "
         "rights regarding goods or services are identifiable, payment terms are "
         "identifiable, the contract has commercial substance, and collection of "
         "consideration is probable."),
        ("Step 2: Identify Performance Obligations",
         "The Company identifies each promise to transfer a distinct good or service as a "
         "separate performance obligation."),
        ("Step 3: Determine the Transaction Price",
         "The transaction price is the amount of consideration the Company expects to be "
         "entitled to in exchange for transferring goods or services, adjusted for variable "
         "consideration where applicable."),
        ("Step 4: Allocate the Transaction Price",
         "Where a contract has multiple performance obligations, the transaction price is "
         "allocated based on relative standalone selling prices."),
        ("Step 5: Recognise Revenue",
         "Revenue is recognised when (or as) the Company satisfies a performance obligation "
         "by transferring control of a good or service to the customer."),
        ("Related-Party Transactions",
         "Revenue and purchase transactions with related parties, as disclosed in Note 18 to "
         "the financial statements, are recorded on the same basis as arm's-length "
         "transactions and are subject to Audit Committee review."),
    ]
    for heading, body in steps:
        document.add_heading(heading, level=2)
        document.add_paragraph(body)
    document.save(str(docs_dir / "policies" / "revenue_recognition_policy.docx"))


def _memo_template(docs_dir: Path) -> None:
    document = Document()
    document.add_heading("Audit Risk Assessment Memo", level=1)
    for label in [
        "Engagement:", "Prepared By:", "Date:", "Risk Identified:",
        "Evidence Reviewed:", "Analysis:", "Conclusion:", "Reviewer Sign-off:",
    ]:
        document.add_paragraph(f"{label} [TO BE COMPLETED]")
    document.save(str(docs_dir / "templates" / "risk_assessment_memo_template.docx"))


def _tables(tables_dir: Path) -> None:
    pd.DataFrame(
        LEDGER_ROWS,
        columns=["transaction_id", "date", "account", "description", "vendor",
                 "debit", "credit", "entered_by"],
    ).to_csv(tables_dir / "general_ledger_extract.csv", index=False)

    pd.DataFrame(
        RELATED_PARTY_ROWS,
        columns=["entity_a", "relationship", "entity_b", "notes"],
    ).to_csv(tables_dir / "related_party_list.csv", index=False)


def generate_corpus(docs_dir: Path, tables_dir: Path) -> None:
    """Write the full synthetic corpus to disk, creating directories as needed."""
    for subdirectory in ("contracts", "invoices", "policies", "templates"):
        (docs_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    _acme_contract(docs_dir)
    _globex_contract(docs_dir)
    _invoices(docs_dir)
    _revenue_policy(docs_dir)
    _memo_template(docs_dir)
    _tables(tables_dir)

    logger.info("Synthetic corpus written to %s and %s", docs_dir, tables_dir)
