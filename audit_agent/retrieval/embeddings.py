"""Sentence-transformer embeddings, loaded once per model name."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper that keeps embeddings L2-normalised for cosine similarity.

    Local inference keeps per-query cost at zero and avoids sending client
    evidence to a third-party embedding API.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name)
