"""Retrieval-augmented generation, built from scratch:

loader (PDF -> pages) -> chunker (pages -> chunks) -> embedder (chunks -> vectors)
-> vector_store (cosine top-k) -> retriever (query -> cited chunks)
"""
