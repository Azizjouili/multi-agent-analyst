"""Manually check what retrieval returns for a query.

    uv run python -m analyst.rag.search "key risks in 2024" [-k 5] [--mode hybrid|vector|keyword]
                                                            [--year 2024]

Why have this: when the RAG agent gives a bad answer, the first question is
"did retrieval find the right chunks?" This answers it without an LLM.
`--mode` lets you compare vector-only, keyword-only and hybrid on the same
query, which shows directly what each one contributes.
"""

from __future__ import annotations

import argparse
import textwrap

from analyst.config import get_settings
from analyst.rag.retriever import Retriever


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search the RAG index.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=get_settings().retrieval_k)
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    parser.add_argument("--year", type=int, action="append", help="metadata filter (repeatable)")
    args = parser.parse_args(argv)

    filters = {"year": args.year} if args.year else None
    hits = Retriever.from_settings().retrieve(args.query, args.k, filters=filters, mode=args.mode)
    for rank, hit in enumerate(hits, start=1):
        rec = hit.record
        print(f"{rank}. score={hit.score:.4f}  {rec['doc']} p.{rec['page']}  ({rec['id']})")
        print(textwrap.indent(textwrap.shorten(rec["text"], 300), "   "))


if __name__ == "__main__":
    main()
