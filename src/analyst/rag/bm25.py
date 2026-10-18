"""Okapi BM25 keyword scoring, written from scratch.

Why have keyword search next to embeddings: dense embeddings capture *meaning*
("sales increased" ~ "revenue grew") but blur *exact tokens*. To a small model,
"2023" and "2024" look almost the same, and so do "31%" and "42%". BM25 is the
opposite: it can't do synonyms, but it rewards the exact terms in a query,
especially rare ones. Combining both (hybrid search) covers each one's blind
spot.

The score of document d for query q:

    sum over terms t in q of:  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

- idf(t): rare terms count more ("tariff" is informative, "company" isn't).
- tf saturation (k1): the 5th occurrence of a term adds much less than the 1st.
- length normalization (b): long chunks don't win just by containing more words.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

# Words too common to help with ranking. idf already down-weights them, but
# on a corpus this small they still add noise, so we drop them.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or our that the this "
    "to was we were what which with how did does do about".split()
)


def _stem(token: str) -> str:
    """Strip plural endings so "risks" matches "Risk Factors".

    A full stemmer (Porter, Snowball) would also merge "growing"/"grew"/
    "growth", but plurals give most of the benefit here at no dependency cost.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"  # "categories" -> "category"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]  # "risks" -> "risk", but keep "business", "analysis"
    return token


def tokenize(text: str) -> list[str]:
    # Numbers are kept as tokens: years and figures are exactly what BM25 is for.
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in documents]
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        # Inverted index: term -> [(doc index, term frequency)]. A query then
        # touches only the documents that contain its terms, not every document.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))

        # The "+1" inside the log keeps idf positive even for a term that
        # appears in most documents (the classic formula can go negative).
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every document (0 = shares no terms with the query)."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue  # term appears in no document
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
