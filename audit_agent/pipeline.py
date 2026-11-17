"""Composition root: builds wired components from settings.

Keeping construction here means the modules stay dependency-injected and
testable, while callers (CLI, examples, notebooks) get a one-line entry point.
"""

from __future__ import annotations

import logging

import pandas as pd

from audit_agent.config import Settings, get_settings
from audit_agent.ingestion.chunking import chunk_records
from audit_agent.ingestion.loaders import load_directory
from audit_agent.knowledge.entity_graph import EntityGraph
from audit_agent.llm.factory import get_provider
from audit_agent.qa import GroundedQA
from audit_agent.retrieval.embeddings import EmbeddingModel
from audit_agent.retrieval.hybrid import HybridRetriever
from audit_agent.retrieval.vector_store import VectorStore
from audit_agent.workflow.classifier import RuleBasedClassifier, TransactionClassifier
from audit_agent.workflow.graph import AuditWorkflow
from audit_agent.workflow.nodes import WorkflowDependencies

logger = logging.getLogger(__name__)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedder=EmbeddingModel(settings.embedding_model),
    )


def ingest_corpus(settings: Settings | None = None, reset: bool = False) -> int:
    """Load, chunk and index every document in the corpus. Returns chunk count."""
    settings = settings or get_settings()
    store = build_vector_store(settings)
    if reset:
        store.clear()

    records = load_directory(settings.docs_dir)
    chunks = chunk_records(records, settings.chunk_size, settings.chunk_overlap)
    store.index(chunks, show_progress=True)
    return len(chunks)


def load_ledger(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    if not settings.ledger_path.exists():
        raise FileNotFoundError(
            f"Ledger not found at {settings.ledger_path}. "
            "Run `python examples/00_setup_corpus.py` first."
        )
    return pd.read_csv(settings.ledger_path)


def build_entity_graph(settings: Settings | None = None) -> EntityGraph:
    settings = settings or get_settings()
    if not settings.related_party_path.exists():
        raise FileNotFoundError(
            f"Related-party table not found at {settings.related_party_path}. "
            "Run `python examples/00_setup_corpus.py` first."
        )
    return EntityGraph.from_csv(
        client_name=settings.client_name,
        related_party_path=settings.related_party_path,
        ledger_path=settings.ledger_path,
    )


def build_qa(settings: Settings | None = None) -> GroundedQA:
    settings = settings or get_settings()
    return GroundedQA(
        vector_store=build_vector_store(settings),
        llm=get_provider(settings),
        top_k=settings.top_k,
    )


def build_classifier(settings: Settings | None = None) -> TransactionClassifier:
    """Build the transaction classifier selected by ``AUDIT_CLASSIFIER``.

    ``qlora`` loads the fine-tuned Qwen adapter from ``settings.qlora_adapter_dir``.
    The adapter is checked for before any heavy import, so a missing adapter
    fails immediately with the command that trains it.
    """
    settings = settings or get_settings()
    if settings.classifier == "rule_based":
        return RuleBasedClassifier()

    if settings.classifier == "qlora":
        from finetuning.classifier import LoRAClassifier, ensure_adapter_exists

        ensure_adapter_exists(settings.qlora_adapter_dir)
        return LoRAClassifier(
            adapter_path=settings.qlora_adapter_dir,
            base_model=settings.qlora_base_model,
        )

    raise ValueError(f"Unknown classifier: {settings.classifier}")


def build_workflow(
    settings: Settings | None = None,
    classifier: TransactionClassifier | None = None,
) -> AuditWorkflow:
    """Assemble the full review workflow.

    The transaction classifier comes from ``AUDIT_CLASSIFIER`` (``rule_based``
    by default, or ``qlora`` for the fine-tuned Qwen adapter). Passing
    ``classifier`` explicitly overrides the setting.
    """
    settings = settings or get_settings()
    classifier = classifier or build_classifier(settings)
    ledger = load_ledger(settings)
    entity_graph = build_entity_graph(settings)

    retriever = HybridRetriever(
        vector_store=build_vector_store(settings),
        entity_graph=entity_graph,
        ledger=ledger,
        top_k=settings.top_k,
    )

    deps = WorkflowDependencies(
        retriever=retriever,
        llm=get_provider(settings),
        ledger=ledger,
        client_name=settings.client_name,
        classifier=classifier,
    )
    return AuditWorkflow(deps)
