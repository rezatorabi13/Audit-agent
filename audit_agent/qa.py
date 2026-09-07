"""Grounded question answering over the indexed corpus.

Two modes: a fixed retrieve-then-answer chain, and a tool-use variant where the
model decides for itself whether the question needs document evidence at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from audit_agent.llm.base import LLMProvider, ToolDefinition, ToolUseNotSupportedError
from audit_agent.models import RetrievedChunk
from audit_agent.retrieval.base import SearchIndex, build_context
from audit_agent.workflow import prompts

logger = logging.getLogger(__name__)

SEARCH_TOOL = ToolDefinition(
    name="search_audit_documents",
    description=(
        "Search the client's engagement documents (contracts, invoices, accounting "
        "policies) for information relevant to a question. Use this whenever the "
        "question could be answered by client-specific evidence."
    ),
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search query"}},
        "required": ["query"],
    },
)


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer plus the chunks it was grounded in."""

    question: str
    text: str
    sources: tuple[RetrievedChunk, ...]
    used_retrieval: bool = True

    @property
    def citations(self) -> list[str]:
        return [chunk.citation for chunk in self.sources]


class GroundedQA:
    """Answers questions strictly from retrieved evidence, with citations."""

    def __init__(self, vector_store: SearchIndex, llm: LLMProvider, top_k: int = 4) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        chunks = self._vector_store.search(question, top_k=top_k or self._top_k)
        if not chunks:
            return Answer(
                question=question,
                text="No indexed documents matched this question.",
                sources=(),
            )

        text = self._llm.complete(
            prompts.GROUNDED_QA,
            f"Context:\n{build_context(chunks)}\n\nQuestion: {question}",
        )
        return Answer(question=question, text=text, sources=tuple(chunks))

    def ask_with_tool_use(self, question: str) -> Answer:
        """Let the model decide whether to retrieve before answering.

        This is the smallest form of agency in the system: one tool, one
        routing decision, made by the model instead of the caller.
        """
        if not self._llm.supports_tools():
            raise ToolUseNotSupportedError(
                f"Provider '{self._llm.name}' does not support tool use; use ask() instead."
            )

        request = self._llm.decide_tool_use(question, [SEARCH_TOOL])
        if not request.wants_tool:
            logger.info("Model answered without retrieval")
            return Answer(
                question=question,
                text=request.text,
                sources=(),
                used_retrieval=False,
            )

        query = str(request.tool_input.get("query", question))
        logger.info("Model requested retrieval: %r", query)
        chunks = self._vector_store.search(query, top_k=self._top_k)

        text = self._llm.complete(
            prompts.GROUNDED_QA,
            f"Context:\n{build_context(chunks)}\n\nQuestion: {question}",
        )
        return Answer(question=question, text=text, sources=tuple(chunks))
