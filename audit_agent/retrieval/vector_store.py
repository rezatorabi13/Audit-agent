"""Persistent Chroma-backed vector index over document chunks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from audit_agent.models import NO_PAGE, DocumentChunk, RetrievedChunk
from audit_agent.retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# Embeddings are normalised, so cosine distance is the metric that matches.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
UPSERT_BATCH_SIZE = 256


class VectorStore:
    """Chunk index with source/page metadata attached to every vector.

    Metadata travels with the vector so retrieved evidence can always be cited
    back to a file and page.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: EmbeddingModel,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=COLLECTION_METADATA,
        )
        self._embedder = embedder
        self.collection_name = collection_name

    def __len__(self) -> int:
        return self._collection.count()

    def index(self, chunks: Sequence[DocumentChunk], show_progress: bool = False) -> int:
        """Embed and upsert chunks. Stable ids make re-indexing idempotent."""
        if not chunks:
            return 0

        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            embeddings = self._embedder.embed_documents(
                [chunk.text for chunk in batch], show_progress=show_progress
            )
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {"source": chunk.source, "page": chunk.page_metadata} for chunk in batch
                ],
            )

        logger.info("Indexed %d chunk(s) into '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def clear(self) -> None:
        """Drop and recreate the collection, for a clean re-ingest."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=COLLECTION_METADATA,
        )

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, scored 0-1 (higher is closer)."""
        if len(self) == 0:
            logger.warning("Vector store '%s' is empty; run ingestion first", self.collection_name)
            return []

        response = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(top_k, len(self)),
        )

        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        results: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            page = metadata.get("page", NO_PAGE)
            results.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    page=None if page == NO_PAGE else int(page),
                    score=1.0 - float(distance),
                )
            )
        return results
