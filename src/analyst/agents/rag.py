"""RAG agent: answers one sub-task from the private PDFs, with page-level citations.

Inside the node:
  1. plan     - the LLM rewrites the sub-task into 1-3 search queries and an optional
                year filter (structured output)
  2. retrieve - hybrid search (vector + BM25, fused with RRF), filtered by year
  3. answer   - the LLM answers ONLY from the retrieved excerpts, citing [D1], [D2], ...
  4. resolve  - local labels -> stable source ids; citations to excerpts that
                don't exist are dropped
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.agents.citations import resolve_citations
from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import RAG_ANSWER_SYSTEM, RAG_QUERY_SYSTEM
from analyst.rag.retriever import Retriever, get_retriever
from analyst.rag.vector_store import Hit
from analyst.state import Finding, Source, SpecialistInput, make_source_id

log = logging.getLogger("analyst.agents.rag")

SNIPPET_CHARS = 1500  # evidence text kept per source (the writer and critic read it later)


class RetrievalPlan(BaseModel):
    """What the LLM returns in step 1. The field descriptions are sent to the model."""

    queries: list[str] = Field(
        min_length=1, max_length=3, description="1-3 short keyword-style search queries"
    )
    years: list[int] = Field(
        default_factory=list,
        description="Report years to restrict the search to; empty unless the sub-task "
        "explicitly names specific years",
    )


@dataclass(frozen=True)
class Excerpt:
    """One retrieved *page*, shown to the LLM as [label]."""

    label: str  # "D1"
    source: Source


class RagAgent:
    """A LangGraph node: `RagAgent()(SpecialistInput) -> {"findings", "sources"}`.

    It's a class rather than a bare function so the retriever and the LLM can
    be injected. Tests pass a fake embedder and a scripted LLM; in production
    both are created lazily on the first call, so building the graph stays
    instant.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        fast_llm: BaseChatModel | None = None,
    ):
        self.settings = settings or get_settings()
        self._retriever = retriever
        self._llm = llm
        # An injected `llm` serves both tiers unless `fast_llm` is given, so a
        # test can script one fake model and see every call in order.
        self._fast_llm = fast_llm or llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever(self.settings)
        return self._retriever

    @property
    def llm(self) -> BaseChatModel:
        """Strong tier: writes the grounded answer."""
        if self._llm is None:
            # Temperature 0: retrieval planning and grounded answering should be
            # as repeatable as possible. There's no use for creativity here.
            self._llm = get_llm(self.settings, tier="strong", temperature=0)
        return self._llm

    @property
    def fast_llm(self) -> BaseChatModel:
        """Fast tier: query rewriting is a light task a small model handles well."""
        if self._fast_llm is None:
            self._fast_llm = get_llm(self.settings, tier="fast", temperature=0)
        return self._fast_llm

    # -- the node ------------------------------------------------------------
    def __call__(self, inp: SpecialistInput) -> dict:
        task = inp["subtask"]
        queries, years = self.plan(inp["question"], task["instruction"])

        filters = {"year": years} if years else None
        hits = self.retriever.retrieve(queries, k=self.settings.retrieval_k, filters=filters)
        if not hits and filters:
            # The filter was too strict (e.g. no chunk passed it). Searching
            # everything beats answering "not found" for a sub-task the
            # documents may still cover.
            log.info("rag[%s]: no hits for years=%s; retrying without filter", task["id"], years)
            hits = self.retriever.retrieve(queries, k=self.settings.retrieval_k)

        excerpts = group_by_page(hits)
        log.info(
            "rag[%s]: queries=%s years=%s -> %d chunks on %d pages",
            task["id"], queries, years or "all", len(hits), len(excerpts),
        )  # fmt: skip
        if not excerpts:
            return _result(task["id"], "NOT FOUND: the internal documents index is empty.", [])

        try:
            answer = self.answer(inp["question"], task["instruction"], excerpts)
        except Exception as exc:  # noqa: BLE001 - see comment
            # A failed LLM call (e.g. a 503 from the free tier after retries)
            # shouldn't crash the whole multi-agent run. Return an explicit
            # error finding; the writer and critic can see the gap and work
            # around it.
            log.exception("rag[%s]: answer generation failed", task["id"])
            return _result(task["id"], f"ERROR: document answer failed ({short_error(exc)}).", [])

        labels = {e.label: e.source["id"] for e in excerpts}
        summary, used_ids = resolve_citations(answer, labels)
        if not used_ids and not summary.startswith("NOT FOUND"):
            log.warning("rag[%s]: answer has no valid citations", task["id"])
        # Return only the sources the answer actually cites. Retrieved but
        # unused pages would be noise in the bibliography.
        sources = [e.source for e in excerpts if e.source["id"] in used_ids]
        return _result(task["id"], summary, sources)

    # -- step 1: query planning ----------------------------------------------
    def plan(self, question: str, instruction: str) -> tuple[list[str], list[int]]:
        available = self.retriever.years
        messages = [
            SystemMessage(
                RAG_QUERY_SYSTEM.format(years=", ".join(map(str, available)) or "unknown")
            ),
            HumanMessage(f"Overall research question: {question}\nSub-task: {instruction}"),
        ]
        try:
            plan = self.fast_llm.with_structured_output(RetrievalPlan).invoke(messages)
            if plan is None:  # some providers return None when parsing fails
                raise ValueError("empty structured output")
        except Exception as exc:  # noqa: BLE001
            # Degrade gracefully: the raw sub-task is still a decent query.
            log.warning("rag: query planning failed (%s); searching with the sub-task text", exc)
            return [instruction], []

        # Always search with the original wording too. Rewrites can drift from
        # what was asked, and RRF rewards chunks that several queries agree on.
        queries = _dedupe([q.strip() for q in plan.queries if q.strip()] + [instruction])
        # Only filter on years that exist. A made-up year ("2031") would
        # otherwise filter out every chunk.
        years = sorted(set(plan.years) & set(available))
        if set(plan.years) - set(available):
            log.info("rag: ignoring years not in the index: %s", set(plan.years) - set(available))
        return queries, years

    # -- step 3: grounded answer ---------------------------------------------
    def answer(self, question: str, instruction: str, excerpts: Sequence[Excerpt]) -> str:
        # Delimiters make clear where each excerpt starts and ends, which helps
        # the model treat document text as data (see prompt rule 5).
        blocks = "\n\n".join(
            f"[{e.label}] {e.source['locator']} ({e.source['title']})\n"
            f"<<<\n{e.source['snippet']}\n>>>"
            for e in excerpts
        )
        messages = [
            SystemMessage(RAG_ANSWER_SYSTEM),
            HumanMessage(
                f"Overall research question: {question}\nSub-task: {instruction}\n\n"
                f"Excerpts:\n\n{blocks}"
            ),
        ]
        # `.text` flattens content blocks. Newer Gemini models return `content`
        # as a list of parts, not a plain string.
        return self.llm.invoke(messages).text.strip()


# ---------------------------------------------------------------------------
# Helpers (pure functions, unit-tested directly)
# ---------------------------------------------------------------------------
def group_by_page(hits: Sequence[Hit]) -> list[Excerpt]:
    """Merge chunk hits into page-level excerpts, labelled D1, D2, ... in rank order.

    Why per page: citations point to pages ("annual_report_2024.pdf#p4"), so
    two chunks from one page are one source. Merging also saves prompt tokens
    when chunks overlap.
    """
    pages: dict[tuple[str, int], list[dict]] = {}
    for hit in hits:  # hits arrive best-first, so labels follow relevance
        rec = hit.record
        pages.setdefault((rec["doc"], rec["page"]), []).append(rec)

    excerpts = []
    for i, ((doc, page), recs) in enumerate(pages.items(), start=1):
        recs.sort(key=lambda r: r["index"])  # restore reading order within the page
        locator = f"{doc}#p{page}"
        source: Source = {
            "id": make_source_id("rag", locator),
            "kind": "rag",
            "title": f"{recs[0].get('title') or doc}, p. {page}",
            "locator": locator,
            "snippet": "\n".join(r["text"] for r in recs)[:SNIPPET_CHARS],
        }
        excerpts.append(Excerpt(label=f"D{i}", source=source))
    return excerpts


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]


def _result(subtask_id: str, summary: str, sources: list[Source]) -> dict:
    finding: Finding = {
        "subtask_id": subtask_id,
        "agent": "rag",
        "summary": summary,
        "source_ids": [s["id"] for s in sources],
    }
    return {"findings": [finding], "sources": {s["id"]: s for s in sources}}
