"""Web-research agent: answers one sub-task from live web pages, citing URLs.

Inside the node:
  1. plan    - the LLM writes 1-2 search-engine queries (structured output)
  2. search  - DuckDuckGo for each query; results deduplicated by URL
  3. select  - the LLM picks at most `web_max_pages` results worth reading
  4. fetch   - those pages are fetched in parallel through the guarded fetcher
               (SSRF check, robots.txt, content type, size cap, timeout)
  5. answer  - the LLM answers ONLY from the most relevant passages of each
               page, citing [W1], [W2], ...; labels are then resolved to URL sources

Each LLM step has a deterministic fallback (raw sub-task as the query, top
results as the selection, search snippets as evidence when no page could be
fetched), so one flaky step degrades the answer instead of failing it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.agents.citations import resolve_citations
from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import WEB_ANSWER_SYSTEM, WEB_QUERY_SYSTEM, WEB_SELECT_SYSTEM
from analyst.rag.bm25 import BM25
from analyst.rag.chunker import split_text
from analyst.state import Finding, Source, SpecialistInput, make_source_id
from analyst.tools.web_fetch import FetchedPage, FetchError, PageFetcher
from analyst.tools.web_search import SearchError, SearchFn, SearchResult, ddg_search

log = logging.getLogger("analyst.agents.web")

SNIPPET_CHARS = 1500


class WebQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=2, description="1-2 web search queries")


class PageSelection(BaseModel):
    selected: list[str] = Field(description='Labels of the results to read, e.g. ["R1", "R3"]')


@dataclass(frozen=True)
class Excerpt:
    label: str  # "W1"
    source: Source
    text: str  # what the LLM reads: selected passages (or the search snippet)


class WebResearchAgent:
    """A LangGraph node: `WebResearchAgent()(SpecialistInput) -> {"findings", "sources"}`."""

    def __init__(
        self,
        search: SearchFn | None = None,
        fetcher: PageFetcher | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        fast_llm: BaseChatModel | None = None,
    ):
        self.settings = settings or get_settings()
        self.search = search or ddg_search
        self._fetcher = fetcher
        self._llm = llm
        # An injected `llm` serves both tiers unless `fast_llm` is given (see RagAgent).
        self._fast_llm = fast_llm or llm

    @property
    def fetcher(self) -> PageFetcher:
        if self._fetcher is None:
            s = self.settings
            self._fetcher = PageFetcher(
                user_agent=s.web_user_agent,
                timeout_seconds=s.web_timeout_seconds,
                max_bytes=s.web_max_bytes,
                max_chars=s.web_max_page_chars,
                respect_robots=s.web_respect_robots,
            )
        return self._fetcher

    @property
    def llm(self) -> BaseChatModel:
        """Strong tier: writes the grounded answer from page excerpts."""
        if self._llm is None:
            self._llm = get_llm(self.settings, tier="strong", temperature=0)
        return self._llm

    @property
    def fast_llm(self) -> BaseChatModel:
        """Fast tier: search queries and page selection are light, frequent steps."""
        if self._fast_llm is None:
            self._fast_llm = get_llm(self.settings, tier="fast", temperature=0)
        return self._fast_llm

    # -- the node ------------------------------------------------------------
    def __call__(self, inp: SpecialistInput) -> dict:
        task = inp["subtask"]
        question, instruction = inp["question"], task["instruction"]

        queries = self._plan(question, instruction)
        results = self._search(queries)
        if not results:
            return _result(task["id"], "NOT FOUND: the web search returned no results.", [])

        chosen = self._select(question, instruction, results)
        backups = [r for r in results if r not in chosen]
        excerpts = self._read(chosen, backups, focus=" ".join([instruction, *queries]))
        log.info(
            "web[%s]: queries=%s -> %d results, read %d page(s)",
            task["id"], queries, len(results), len(excerpts),
        )  # fmt: skip

        try:
            answer = self._answer(question, instruction, excerpts)
        except Exception as exc:  # noqa: BLE001 - don't let one LLM failure sink the run
            log.exception("web[%s]: answer generation failed", task["id"])
            return _result(task["id"], f"ERROR: web answer failed ({short_error(exc)}).", [])

        labels = {e.label: e.source["id"] for e in excerpts}
        summary, used_ids = resolve_citations(answer, labels)
        sources = [e.source for e in excerpts if e.source["id"] in used_ids]
        return _result(task["id"], summary, sources)

    # -- 1. plan ---------------------------------------------------------------
    def _plan(self, question: str, instruction: str) -> list[str]:
        try:
            plan = self.fast_llm.with_structured_output(WebQueries).invoke(
                [
                    SystemMessage(WEB_QUERY_SYSTEM),
                    HumanMessage(f"Overall research question: {question}\nSub-task: {instruction}"),
                ]
            )
            queries = [q.strip() for q in plan.queries if q.strip()] if plan else []
        except Exception as exc:  # noqa: BLE001
            log.warning("web: query planning failed (%s); searching with the sub-task text", exc)
            queries = []
        return queries or [instruction]

    # -- 2. search -------------------------------------------------------------
    def _search(self, queries: list[str]) -> list[SearchResult]:
        seen: set[str] = set()
        results: list[SearchResult] = []
        for query in queries:
            try:
                hits = self.search(query, self.settings.web_max_results)
            except SearchError as exc:
                log.warning("web: search failed for %r: %s", query, exc)
                continue
            for hit in hits:
                key = _url_key(hit.url)
                if key not in seen:  # the same page often appears for both queries
                    seen.add(key)
                    results.append(hit)
        return results

    # -- 3. select -------------------------------------------------------------
    def _select(
        self, question: str, instruction: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        max_pages = self.settings.web_max_pages
        if len(results) <= max_pages:
            return results  # nothing to choose between: skip an LLM call
        listing = "\n".join(
            f"R{i}. {r.title} - {r.url}\n    {r.snippet}" for i, r in enumerate(results, start=1)
        )
        try:
            selection = self.fast_llm.with_structured_output(PageSelection).invoke(
                [
                    SystemMessage(WEB_SELECT_SYSTEM.format(max_pages=max_pages)),
                    HumanMessage(
                        f"Overall research question: {question}\nSub-task: {instruction}\n\n"
                        f"Search results:\n{listing}"
                    ),
                ]
            )
            labels = selection.selected if selection else []
        except Exception as exc:  # noqa: BLE001
            log.warning("web: page selection failed (%s); reading the top results", exc)
            labels = []

        # Keep only labels that exist, in the model's order, without duplicates.
        by_label = {f"R{i}": r for i, r in enumerate(results, start=1)}
        chosen = list(dict.fromkeys(by_label[lbl] for lbl in labels if lbl in by_label))
        return chosen[:max_pages] or results[:max_pages]

    # -- 4. fetch --------------------------------------------------------------
    def _read(
        self, chosen: list[SearchResult], backups: list[SearchResult], focus: str
    ) -> list[Excerpt]:
        fetched = self._fetch_with_backfill(chosen, backups)
        budget = self.settings.web_prompt_chars_per_page
        if fetched:
            items = [
                (page.url, page.title or r.title, select_passages(page.text, focus, budget))
                for r, page in fetched
            ]
        else:
            # Every fetch failed (robots.txt, paywall, timeouts). Search snippets
            # are thin evidence but still real and citable, which beats nothing.
            log.warning("web: no page could be fetched; falling back to search snippets")
            items = [(r.url, f"{r.title} (search snippet)", r.snippet) for r in chosen]

        excerpts = []
        for i, (url, title, text) in enumerate(items, start=1):
            source: Source = {
                "id": make_source_id("web", url),
                "kind": "web",
                "title": title,
                "locator": url,
                "snippet": text[:SNIPPET_CHARS],
            }
            excerpts.append(Excerpt(label=f"W{i}", source=source, text=text))
        return excerpts

    def _fetch_with_backfill(
        self, chosen: list[SearchResult], backups: list[SearchResult]
    ) -> list[tuple[SearchResult, FetchedPage]]:
        """Fetch the chosen pages, and replace failures with the next-best results.

        Why backfill: in practice many sites answer 403 to non-browser clients,
        or block crawlers in robots.txt. In a live test, 2 of 3 chosen pages
        failed this way. Topping up from the remaining results recovers most
        of that, while the attempt cap keeps the time and number of requests
        bounded. (Faking a browser User-Agent would "fix" the 403s too, but it
        is impersonation and ignores the site's wishes.)
        """
        target = self.settings.web_max_pages
        max_attempts = 2 * target
        queue = [*chosen, *backups]
        fetched: list[tuple[SearchResult, FetchedPage]] = []
        blocked_hosts: set[str] = set()
        attempts = 0
        while queue and len(fetched) < target and attempts < max_attempts:
            # A host that already refused us (403, robots.txt) will almost
            # certainly refuse its other URLs too. Skip them without spending an
            # attempt, so the budget goes to different sites.
            queue = [r for r in queue if _host(r.url) not in blocked_hosts]
            if not queue:
                break
            batch_size = min(target - len(fetched), max_attempts - attempts)
            batch, queue = queue[:batch_size], queue[batch_size:]
            attempts += len(batch)
            # Fetch each batch in parallel: the work is network-bound, so three
            # pages at up to 10 s each take ~10 s, not ~30 s. httpx.Client is thread-safe.
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                outcomes = list(pool.map(self._try_fetch, batch))
            for result, outcome in zip(batch, outcomes, strict=True):
                if isinstance(outcome, FetchError):
                    if outcome.blocked:
                        blocked_hosts.add(_host(result.url))
                else:
                    fetched.append((result, outcome))
        return fetched

    def _try_fetch(self, result: SearchResult) -> FetchedPage | FetchError:
        try:
            return self.fetcher.fetch(result.url)
        except FetchError as exc:
            log.info("web: skipped %s: %s", result.url, exc)
            return exc

    # -- 5. answer -------------------------------------------------------------
    def _answer(self, question: str, instruction: str, excerpts: list[Excerpt]) -> str:
        blocks = "\n\n".join(
            f"[{e.label}] {e.source['title']}\nURL: {e.source['locator']}\n<<<\n{e.text}\n>>>"
            for e in excerpts
        )
        messages = [
            SystemMessage(WEB_ANSWER_SYSTEM),
            HumanMessage(
                f"Overall research question: {question}\nSub-task: {instruction}\n\n"
                f"Web page excerpts:\n\n{blocks}"
            ),
        ]
        return self.llm.invoke(messages).text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_passages(text: str, focus: str, budget: int, passage_chars: int = 600) -> str:
    """The parts of a long page most relevant to `focus`, within `budget` characters.

    Why not just the first `budget` characters: articles often get to the
    useful figure in paragraph eight. We reuse our own chunker and BM25: split
    the page into passages, score them against the sub-task, keep the best
    ones, and print them in page order so the text still reads naturally.
    If nothing matches, BM25 scores are all zero and the tie-break on position
    keeps the start of the page, a sensible default.
    """
    if len(text) <= budget:
        return text
    # A passage must fit within the budget, or the most relevant one could
    # never be selected (a small budget would then keep only scraps).
    passages = split_text(text, size=min(passage_chars, budget), overlap=0)
    scores = BM25(passages).scores(focus)
    ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    chosen, used = [], 0
    for i in ranked:
        if used + len(passages[i]) <= budget:
            chosen.append(i)
            used += len(passages[i]) + 5  # + separator
    return "\n[...]\n".join(passages[i] for i in sorted(chosen))


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _url_key(url: str) -> str:
    """Normalize a URL for deduplication: case-insensitive host, no fragment or trailing '/'."""
    p = urlsplit(url)
    return f"{p.netloc.lower()}{p.path.rstrip('/')}?{p.query}"


def _result(subtask_id: str, summary: str, sources: list[Source]) -> dict:
    finding: Finding = {
        "subtask_id": subtask_id,
        "agent": "web",
        "summary": summary,
        "source_ids": [s["id"] for s in sources],
    }
    return {"findings": [finding], "sources": {s["id"]: s for s in sources}}
