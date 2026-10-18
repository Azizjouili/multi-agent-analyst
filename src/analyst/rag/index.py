"""Build the vector index from the PDFs in data/docs.

    uv run python -m analyst.rag.index [--docs-dir DIR] [--index-dir DIR]
                                       [--chunk-size N] [--chunk-overlap N]

Indexing is a separate offline step, not something done at query time: it's
the slow part (loading the model, embedding every chunk), and documents change
far less often than questions are asked. Rerun it whenever the PDFs, the
embedding model or the chunking settings change.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from analyst.config import get_settings
from analyst.rag.chunker import chunk_pages
from analyst.rag.embedder import Embedder, get_embedder
from analyst.rag.loader import load_dir
from analyst.rag.vector_store import VectorStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(
    docs_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    pages = load_dir(docs_dir)
    if not pages:
        raise SystemExit(
            f"No PDF text found in {docs_dir}. Generate the samples first: "
            "uv run python scripts/make_sample_data.py"
        )
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)

    t0 = time.perf_counter()
    vectors = embedder.embed_documents([c.embedding_text for c in chunks])
    embed_seconds = time.perf_counter() - t0

    store = VectorStore(dim=vectors.shape[1], model_name=embedder.model_name)
    store.add(vectors, [c.to_record() for c in chunks])
    # The manifest records how the index was built. File hashes let us
    # detect later that the documents changed and the index is stale.
    store.save(
        index_dir,
        extra={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "files": {p.name: _sha256(p) for p in sorted(docs_dir.glob("*.pdf"))},
        },
    )
    return {
        "documents": len({p.doc for p in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks)),
        "dim": store.dim,
        "embed_seconds": round(embed_seconds, 2),
    }


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--docs-dir", type=Path, default=s.docs_dir)
    parser.add_argument("--index-dir", type=Path, default=s.index_dir)
    parser.add_argument("--chunk-size", type=int, default=s.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    args = parser.parse_args(argv)

    print(f"Indexing {args.docs_dir} with {s.embedding_model} ...")
    stats = build_index(
        args.docs_dir,
        args.index_dir,
        get_embedder(s.embedding_model),
        args.chunk_size,
        args.chunk_overlap,
    )
    print(f"Saved index to {args.index_dir}")
    for key, value in stats.items():
        print(f"  {key:>16}: {value}")


if __name__ == "__main__":
    main()
