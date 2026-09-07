"""Chunking must preserve the citation metadata retrieval depends on."""

from __future__ import annotations

from audit_agent.ingestion.chunking import chunk_records
from audit_agent.models import NO_PAGE, DocumentChunk, DocumentRecord

LONG_TEXT = "\n\n".join(f"Clause {i}. " + "word " * 60 for i in range(6))


def test_chunks_carry_source_and_page() -> None:
    records = [DocumentRecord(source="contract.pdf", page=3, text=LONG_TEXT)]
    chunks = chunk_records(records, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.source == "contract.pdf" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)


def test_chunk_ids_are_unique_and_stable() -> None:
    records = [DocumentRecord(source="contract.pdf", page=1, text=LONG_TEXT)]
    first = chunk_records(records, chunk_size=200, chunk_overlap=20)
    second = chunk_records(records, chunk_size=200, chunk_overlap=20)

    ids = [chunk.chunk_id for chunk in first]
    assert len(ids) == len(set(ids))
    assert ids == [chunk.chunk_id for chunk in second]


def test_pageless_documents_use_the_metadata_sentinel() -> None:
    chunk = DocumentChunk(chunk_id="policy.docx::pNone::c0", source="policy.docx", text="text")
    assert chunk.page_metadata == NO_PAGE


def test_short_record_produces_a_single_chunk() -> None:
    records = [DocumentRecord(source="note.docx", page=None, text="Net 30 payment terms.")]
    chunks = chunk_records(records, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].page is None
