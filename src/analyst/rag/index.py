"""Build the vector index from the documents in data/docs (10-K filings and PDFs).

    uv run python -m analyst.rag.index [--docs-dir DIR] [--index-dir DIR]
                                       [--chunk-size N] [--chunk-overlap N]

Indexing is a separate offline step, not something done at query time: it's
the slow part (loading the model, embedding every chunk), and documents change
far less often than questions are asked. Rerun it whenever the PDFs, the
embedding model, the chunking settings or EDGAR_ITEMS change.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from analyst.config import get_settings
from analyst.rag.chunker import chunk_sections
from analyst.rag.embedder import Embedder, get_embedder
from analyst.rag.loader import DEFAULT_ITEMS, doc_files, load_dir
from analyst.rag.vector_store import VectorStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(
    docs_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    items: tuple[str, ...] = DEFAULT_ITEMS,
) -> dict:
    sections = load_dir(docs_dir, items)
    if not sections:
        raise SystemExit(
            f"No document text found in {docs_dir}. Fetch the SEC filings first: "
            "uv run python scripts/fetch_edgar.py"
        )
    chunks = chunk_sections(sections, chunk_size, chunk_overlap)

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
            "items": list(items),
            # Only documents that produced text: an HTML file with none of the
            # selected Items was not really indexed.
            "files": {
                p.name: _sha256(p)
                for p in doc_files(docs_dir)
                if p.name in {s.doc for s in sections}
            },
        },
    )
    return {
        "documents": len({s.doc for s in sections}),
        "companies": len({s.ticker for s in sections if s.ticker}),
        "sections": len(sections),
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

    print(f"Indexing {args.docs_dir} (10-K Items {', '.join(s.edgar_items)}) "
          f"with {s.embedding_model} ...")  # fmt: skip
    stats = build_index(
        args.docs_dir,
        args.index_dir,
        get_embedder(s.embedding_model),
        args.chunk_size,
        args.chunk_overlap,
        s.edgar_items,
    )
    print(f"Saved index to {args.index_dir}")
    for key, value in stats.items():
        print(f"  {key:>16}: {value}")


if __name__ == "__main__":
    main()
