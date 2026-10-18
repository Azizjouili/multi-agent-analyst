"""A vector store in about 100 lines of NumPy.

What a vector database does at its core:
  - keep an (n x d) matrix of embeddings plus metadata for each row;
  - on a query: score every row by cosine similarity and return the top k.

With unit-length vectors, cosine similarity = dot product, so scoring every
row is one matrix-vector product, `E @ q`. That's a brute-force (exact) search
in O(n*d). For thousands of chunks it takes well under a millisecond. Real
vector DBs add approximate indexes (HNSW, IVF) to stay fast at millions of
rows, plus filtering, updates and replication. That's what you'd move to at
scale, and why this project doesn't need it.

On disk (a directory):
  embeddings.npy  - float32 matrix, loads instantly with np.load
  chunks.jsonl    - one JSON metadata record per row, in the same order
  manifest.json   - model name, dimension, count, and build settings
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
RECORDS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"
# Bump when the record format changes. Version 2 added the `year` and `title`
# fields that filtering and hybrid search depend on. Version 3 replaced the PDF
# `page` with a `section` (10-K Item or page), its `anchor`, and the `ticker`.
INDEX_VERSION = 3


class VectorIndexError(RuntimeError):
    """The index is missing, corrupt, outdated, or was built with a different model."""


def matches(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Metadata filter: every key must match. A list/tuple/set value means "any of".

    {"year": 2024} -> only 2024 records; {"year": [2023, 2024]} -> either year.
    """
    for key, wanted in (filters or {}).items():
        value = record.get(key)
        if isinstance(wanted, list | tuple | set | frozenset):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


@dataclass(frozen=True)
class Hit:
    record: dict[str, Any]
    score: float  # higher = more relevant (cosine here; a fused RRF score from the retriever)


def _normalize(m: np.ndarray) -> np.ndarray:
    # Unit length per row. Zero rows stay zero instead of becoming NaN.
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self, dim: int, model_name: str | None = None):
        self.dim = dim
        self.model_name = model_name
        self._embeddings = np.zeros((0, dim), dtype=np.float32)
        self._records: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}  # filled by load(): build settings, source files

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def mask(self, filters: dict[str, Any] | None) -> np.ndarray:
        """Boolean array: which rows pass the metadata filter."""
        return np.array([matches(r, filters) for r in self._records], dtype=bool)

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query with every row. The retriever ranks these."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dim {q.shape[0]}, index has dim {self.dim}")
        return self._embeddings @ _normalize(q)

    def add(self, embeddings: np.ndarray, records: list[dict[str, Any]]) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {emb.shape}")
        if len(emb) != len(records):
            raise ValueError(f"{len(emb)} embeddings but {len(records)} records")
        # Normalize again in case a caller passes raw vectors. The dot
        # product in search() is only a cosine similarity for unit vectors.
        self._embeddings = np.vstack([self._embeddings, _normalize(emb)])
        self._records.extend(records)

    def search(
        self, query: np.ndarray, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Hit]:
        scores = self.scores(query)  # cosine similarity with every chunk
        # Filter *before* taking the top k. Taking the top k and filtering
        # afterwards could leave fewer than k results, or none.
        candidates = np.flatnonzero(self.mask(filters))
        if candidates.size == 0:
            return []
        k = min(k, candidates.size)
        cand_scores = scores[candidates]
        # argpartition finds the k best in O(n) without sorting all n scores.
        # Then we sort only those k.
        top = np.argpartition(-cand_scores, k - 1)[:k]
        top = top[np.argsort(-cand_scores[top])]
        return [Hit(record=self._records[i], score=float(scores[i])) for i in candidates[top]]

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path, extra: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self._embeddings)
        with (directory / RECORDS_FILE).open("w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest = {
            "version": INDEX_VERSION,
            "model": self.model_name,
            "dim": self.dim,
            "count": len(self),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **(extra or {}),
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> VectorStore:
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise VectorIndexError(
                f"No index at {directory}. Build it with: uv run python -m analyst.rag.index"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # An outdated index lacks fields newer code relies on. For example, a
        # year filter on records with no `year` would silently match nothing.
        if manifest.get("version", 1) != INDEX_VERSION:
            raise VectorIndexError(
                f"Index at {directory} uses format v{manifest.get('version', 1)}, but this code "
                f"needs v{INDEX_VERSION}. Rebuild it: uv run python -m analyst.rag.index"
            )

        # Why check the model: vectors from different models live in unrelated
        # spaces. Searching an index with a query embedded by another model
        # doesn't crash; it returns plausible-looking but meaningless results.
        # That's the worst kind of bug, so we fail loudly instead.
        if expected_model and manifest["model"] != expected_model:
            raise VectorIndexError(
                f"Index was built with {manifest['model']!r} but the configured model is "
                f"{expected_model!r}. Rebuild it: uv run python -m analyst.rag.index"
            )

        embeddings = np.load(directory / EMBEDDINGS_FILE)
        with (directory / RECORDS_FILE).open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if (
            embeddings.shape != (manifest["count"], manifest["dim"])
            or len(records) != manifest["count"]
        ):
            raise VectorIndexError(f"Index at {directory} is inconsistent; rebuild it.")

        store = cls(dim=manifest["dim"], model_name=manifest["model"])
        store._embeddings = embeddings.astype(np.float32)
        store._records = records
        store.manifest = manifest
        return store
