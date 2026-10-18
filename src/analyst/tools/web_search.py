"""Web search through DuckDuckGo (the `ddgs` package). No API key, no cost.

Why behind a tiny wrapper: the agent depends only on
`search(query, max_results) -> list[SearchResult]`. Tests pass a fake
function, and switching to Tavily or Brave Search later means writing one more
function with the same signature.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(RuntimeError):
    """Search failed (rate limit, timeout, network). The agent can try another query."""


SearchFn = Callable[[str, int], list[SearchResult]]


def ddg_search(query: str, max_results: int = 5) -> list[SearchResult]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        raw = DDGS(timeout=10).text(query, max_results=max_results)
    except DDGSException as exc:
        # DuckDuckGo rate-limits aggressive clients. Report it as a normal
        # failure the agent can handle, not a crash.
        raise SearchError(f"DuckDuckGo search failed: {exc}") from exc
    return [
        SearchResult(title=r.get("title", ""), url=r["href"], snippet=r.get("body", ""))
        for r in raw
        if r.get("href")
    ]
