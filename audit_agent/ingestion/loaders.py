"""Text extraction from the file formats engagement evidence actually arrives in."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument

from audit_agent.models import DocumentRecord

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[DocumentRecord]:
    """Extract one record per non-empty PDF page, preserving page numbers.

    Page numbers are retained because a citation without one is not traceable
    back to the source document during review.
    """
    records: list[DocumentRecord] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append(DocumentRecord(source=path.name, page=page_number, text=text))
    return records


def load_docx(path: Path) -> list[DocumentRecord]:
    """Extract a Word document as a single record.

    Word has no fixed pagination, so these are cited by filename alone.
    """
    document = DocxDocument(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [DocumentRecord(source=path.name, page=None, text=text)]


LOADERS: dict[str, Callable[[Path], list[DocumentRecord]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_directory(folder: Path) -> list[DocumentRecord]:
    """Recursively load every supported document under ``folder``.

    Unreadable files are logged and skipped rather than aborting the run: a
    single corrupt client file should not block ingestion of the rest.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Document directory not found: {folder}. "
            "Run `python examples/00_setup_corpus.py` first."
        )

    records: list[DocumentRecord] = []
    for path in sorted(folder.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            records.extend(loader(path))
        except Exception:
            logger.exception("Failed to load %s; skipping", path.name)

    logger.info("Loaded %d record(s) from %s", len(records), folder)
    return records
