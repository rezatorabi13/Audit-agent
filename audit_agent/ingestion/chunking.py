"""Splitting documents into retrievable, individually citable chunks."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from audit_agent.models import DocumentChunk, DocumentRecord

# Prefer paragraph, then sentence, then word boundaries so contract clauses
# survive splitting intact; a clause cut mid-sentence retrieves poorly.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_records(
    records: Iterable[DocumentRecord],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Split records into overlapping chunks with stable, unique ids.

    The id encodes source, page and ordinal so re-ingesting the same corpus
    overwrites rather than duplicates entries in the vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )

    chunks: list[DocumentChunk] = []
    for record in records:
        for ordinal, piece in enumerate(splitter.split_text(record.text)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{record.source}::p{record.page}::c{ordinal}",
                    source=record.source,
                    page=record.page,
                    text=piece,
                )
            )
    return chunks
