"""Test doubles shared across test modules."""

import hashlib
import re

import httpx
import numpy as np

from analyst.tools.web_fetch import PageFetcher
from analyst.tools.web_search import SearchResult

PUBLIC_IP = "93.184.216.34"


class FakeSite:
    """An in-memory internet for the fetcher, built on httpx.MockTransport.

    `routes` maps a full URL to (status, headers, body). Unknown URLs return 404,
    which for /robots.txt means "no rules". Every requested URL is recorded in
    `requests`. The fetcher runs its real code paths: streaming, redirects,
    robots parsing. Only the transport is fake.
    """

    def __init__(self, routes: dict[str, tuple[int, dict, str | bytes]] | None = None):
        self.routes = routes or {}
        self.requests: list[str] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404, request=request)
        if callable(route):
            return route(request)
        status, headers, body = route
        return httpx.Response(status, headers=headers, content=body, request=request)

    def fetcher(self, resolver=lambda host: [PUBLIC_IP], **kwargs) -> PageFetcher:
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        return PageFetcher(client, resolver=resolver, **kwargs)


def html_page(title: str, body: str) -> tuple[int, dict, str]:
    return (
        200,
        {"content-type": "text/html; charset=utf-8"},
        f"<html><head><title>{title}</title></head><body>{body}</body></html>",
    )


class FakeSearch:
    """A search function that returns canned results per query and records the queries."""

    def __init__(self, results: dict[str, list[SearchResult]] | list[SearchResult]):
        self.results = results
        self.queries: list[str] = []

    def __call__(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        if isinstance(self.results, dict):
            return self.results.get(query, [])[:max_results]
        return self.results[:max_results]


class BagOfWordsEmbedder:
    """Offline stand-in for the real model: hashed word counts, unit length.

    It captures word overlap but not meaning, which is enough to test that the
    pipeline is wired correctly without downloading model weights.
    """

    model_name = "test/bag-of-words"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)
