"""Domain objects passed between ingestion, retrieval, analysis and drafting."""

from __future__ import annotations

from dataclasses import dataclass, field

# Chroma metadata cannot hold None, so page-less sources (Word documents) are
# stored with this sentinel and rendered as an un-paginated citation.
NO_PAGE = -1


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One extracted unit of a source file: a PDF page, or a whole Word document."""

    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable slice of a :class:`DocumentRecord`, carrying its citation."""

    chunk_id: str
    source: str
    text: str
    page: int | None = None

    @property
    def page_metadata(self) -> int:
        return NO_PAGE if self.page is None else self.page


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by vector search, with its similarity score."""

    text: str
    source: str
    page: int | None
    score: float

    @property
    def citation(self) -> str:
        if self.page in (None, NO_PAGE):
            return self.source
        return f"{self.source}, p.{self.page}"


@dataclass(frozen=True, slots=True)
class AnomalyFlag:
    """A deterministic finding raised by a ledger rule, not by a model."""

    rule: str
    transaction_id: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.transaction_id}: {self.detail}"


@dataclass(frozen=True, slots=True)
class RelatedPartyFinding:
    """Result of traversing the entity graph for undisclosed vendor relationships."""

    vendor: str
    is_related_party: bool
    shared_board_members: tuple[str, ...] = ()

    def summary(self) -> str:
        if not self.is_related_party:
            return f"Knowledge graph check: no related-party link found for '{self.vendor}'."
        members = ", ".join(self.shared_board_members)
        return (
            f"Knowledge graph check: RELATED PARTY DETECTED for '{self.vendor}' "
            f"via shared board member(s): {members}."
        )


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything gathered for one vendor before any generation happens.

    Keeping retrieval output in a single object is what makes the drafted memo
    auditable: each claim can be traced to a chunk citation, a graph edge, or a
    ledger rule.
    """

    vendor: str
    question: str
    related_party: RelatedPartyFinding
    chunks: tuple[RetrievedChunk, ...] = ()
    anomalies: tuple[AnomalyFlag, ...] = field(default=())

    def as_context(self) -> str:
        """Render the bundle as a prompt context block with inline citations."""
        blocks = [self.related_party.summary()]
        if self.anomalies:
            flags = "\n".join(str(flag) for flag in self.anomalies)
            blocks.append(f"Ledger rule findings:\n{flags}")
        else:
            blocks.append("Ledger rule findings: none for this vendor.")
        for chunk in self.chunks:
            blocks.append(f"[Source: {chunk.citation}]\n{chunk.text}")
        return "\n\n---\n\n".join(blocks)
