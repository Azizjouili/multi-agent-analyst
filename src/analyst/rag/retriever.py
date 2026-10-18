"""Query -> top-k chunks with hybrid search. The only RAG entry point agents use.

Hybrid = vector (cosine) ranking + BM25 keyword ranking, fused with
Reciprocal Rank Fusion:

    rrf(doc) = sum over rankings r of:  1 / (RRF_K + rank_r(doc))

Why RRF instead of a weighted sum of scores: BM25 scores have no upper bound
and depend on the query and corpus, while cosine scores sit in [-1, 1].
Blending them linearly needs score normalization and a tuned weight, and both
drift as the corpus changes. RRF uses only *ranks*, so there's nothing to
normalize or tune. A document near the top of both lists beats one that is
first in a single list. Elasticsearch and OpenSearch use it for hybrid search
for the same reason.

RRF_K = 60 is the constant from the original paper (Cormack et al., 2009). It
flattens the curve so ranks 1 and 2 differ only slightly. That prevents one
ranking from dominating just because it ranked something first.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from analyst.config import Settings, get_settings
from analyst.rag.bm25 import BM25
from analyst.rag.chunker import contextual_text
from analyst.rag.embedder import Embedder, get_embedder
from analyst.rag.vector_store import Hit, VectorStore

RRF_K = 60
# How deep each ranking goes before fusion. Fusing only the top candidates
# keeps a long tail of weak matches from adding up to a high fused score.
CANDIDATES = 20

Mode = Literal["hybrid", "vector", "keyword"]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Fuse ranked lists of row ids into one {row id: score} map (higher is better)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            fused[row] += 1.0 / (k + rank)
    return dict(fused)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        # BM25 is built from the stored records when the retriever is created
        # rather than saved to disk. For ~10,000 chunks that takes a second or
        # two, once per process, and it can never go stale relative to the vectors.
        self.bm25 = BM25(
            [contextual_text(r.get("title", ""), r["section"], r["text"]) for r in store.records]
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Retriever:
        settings = settings or get_settings()
        store = VectorStore.load(settings.index_dir, expected_model=settings.embedding_model)
        return cls(store, get_embedder(settings.embedding_model))

    @property
    def years(self) -> list[int]:
        """Fiscal or report years in the index. The agent only filters on these."""
        return sorted({r["year"] for r in self.store.records if r.get("year")})

    def retrieve(
        self,
        queries: str | Sequence[str],
        k: int = 5,
        filters: dict[str, Any] | None = None,
        mode: Mode = "hybrid",
    ) -> list[Hit]:
        """Top-k chunks for one or more queries. Hit.score is the fused RRF score.

        Several queries (from query rewriting) are fused the same way as the
        two search methods: each query contributes its own rankings. A chunk
        that several phrasings agree on rises to the top.
        """
        if isinstance(queries, str):
            queries = [queries]
        candidates = np.flatnonzero(self.store.mask(filters))
        if candidates.size == 0:
            return []

        rankings: list[list[int]] = []
        for query in queries:
            if mode in ("hybrid", "vector"):
                cos = self.store.scores(self.embedder.embed_query(query))[candidates]
                rankings.append(self._rank(candidates, cos))
            if mode in ("hybrid", "keyword"):
                bm25 = self.bm25.scores(query)[candidates]
                # A chunk with BM25 score 0 shares no terms with the query.
                # Leaving it out of the keyword ranking, instead of giving it a
                # low rank, stops it from collecting RRF credit for nothing.
                keep = bm25 > 0
                rankings.append(self._rank(candidates[keep], bm25[keep]))

        fused = reciprocal_rank_fusion(rankings)
        # Ties are broken by row order, so results are deterministic.
        top = sorted(fused, key=lambda row: (-fused[row], row))[:k]
        return [Hit(record=self.store.records[row], score=fused[row]) for row in top]

    @staticmethod
    def _rank(rows: np.ndarray, scores: np.ndarray) -> list[int]:
        order = np.argsort(-scores, kind="stable")[:CANDIDATES]
        return [int(r) for r in rows[order]]


# One retriever per process, built on first use. Why lazy: building the graph,
# or running offline tests, shouldn't load the index or the embedding model.
# Why a lock: parallel Send branches run in threads and could otherwise build
# two copies.
_RETRIEVER: Retriever | None = None
_LOCK = threading.Lock()


def get_retriever(settings: Settings | None = None) -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        with _LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = Retriever.from_settings(settings)
    return _RETRIEVER
