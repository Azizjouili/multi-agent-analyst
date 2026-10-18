"""Local text embeddings with sentence-transformers (free, runs on CPU).

all-MiniLM-L6-v2 maps text to a 384-dimensional vector. Texts with similar
meanings get vectors that point in similar directions. It is small (~90 MB),
fast on a laptop CPU, and good enough for a few hundred chunks.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import lru_cache

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        # Lazy load: importing torch and loading weights takes seconds. Code
        # paths that never embed anything (tests, the API's health check)
        # don't pay that cost.
        #
        # The lock matters because LangGraph runs parallel Send branches in
        # threads. Two RAG sub-tasks starting together would otherwise both
        # see `None` and load the weights twice.
        if self._model is None:
            with self._lock:
                if self._model is None:  # re-check: another thread may have loaded it
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        # Why separate document/query methods: some embedding models (e5, bge)
        # expect a different prefix for queries and for passages. MiniLM uses
        # none, but going through encode_document/encode_query means swapping
        # in such a model needs no code change.
        #
        # normalize_embeddings=True scales every vector to length 1, so the
        # cosine similarity of two vectors is just their dot product. The
        # vector store relies on that.
        vectors = self.model.encode_document(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode_query([text], normalize_embeddings=True, convert_to_numpy=True)
        return vector[0].astype(np.float32)


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    """One Embedder per model per process.

    Why: the indexer, the retriever and every RAG agent call would otherwise
    each load their own copy of the weights.
    """
    return Embedder(model_name)
