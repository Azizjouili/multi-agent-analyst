"""Finalize: turn the approved draft into the reader-facing report.

  1. Renumber citations: stable ids ([rag:ab12cd34, data:9f8e7d6c]) become
     [1], [2], ... in order of first appearance. A source cited again reuses
     its number, and numbers inside one group are sorted: [1, 3].
  2. Drop citations to ids that aren't in state["sources"]. This is the last
     line of defense against invented citations.
  3. Append a "Sources" section listing every cited source with its label,
     type, and a one-line description.

Pure code, no LLM: renumbering must be exact, and a model could easily
misnumber or forget an entry. It's also free and instant.
"""

from __future__ import annotations

import logging

from analyst.agents.citations import SOURCE_ID, rewrite_citations
from analyst.state import AnalystState, Source

log = logging.getLogger("analyst.agents.finalize")

DESCRIPTION_CHARS = 160
TYPE_LABELS = {"rag": "internal document", "data": "SQL query", "web": "web page"}


def finalize(state: AnalystState) -> dict:
    report, cited = render_final_report(state.get("draft", ""), state.get("sources", {}))
    log.info(
        "finalize: %d source(s) cited of %d collected", len(cited), len(state.get("sources", {}))
    )
    return {"final_report": report}


def render_final_report(draft: str, sources: dict[str, Source]) -> tuple[str, list[str]]:
    """Return (report with [n] citations and a Sources section, cited ids in number order)."""
    # Pass 1: which valid ids are cited, in order of first appearance. Unknown
    # ids map to nothing, so they're excluded from the numbering.
    _, cited = rewrite_citations(draft, {sid: sid for sid in sources}, part=SOURCE_ID)
    number = {sid: str(i) for i, sid in enumerate(cited, start=1)}

    # Pass 2: replace ids with numbers (sorted within a group); unknown ids vanish.
    def join(nums: list[str]) -> str:
        return "[" + ", ".join(sorted(nums, key=int)) + "]"

    body, _ = rewrite_citations(draft, number, part=SOURCE_ID, join=join)

    if cited:
        entries = "\n".join(_entry(i, sources[sid]) for i, sid in enumerate(cited, start=1))
    else:
        entries = "_No sources were cited._"
    return f"{body.rstrip()}\n\n## Sources\n\n{entries}\n", cited


def _entry(n: int, source: Source) -> str:
    kind = source["kind"]
    label = source["title"].removeprefix("SQL: ")  # data titles carry a redundant prefix
    return f"{n}. **{label}** ({kind}: {TYPE_LABELS.get(kind, kind)}) - {describe(source)}"


def describe(source: Source) -> str:
    """One line telling a reader where the evidence is and how to check it."""
    kind, locator = source["kind"], source["locator"]
    if kind == "web":
        return f"<{locator}>"
    if kind == "rag":
        doc, _, page = locator.partition("#p")  # "annual_report_2024.pdf#p4"
        return f"{doc}, page {page}" if page else doc
    # data: the query itself is the citation, so a reader can re-run it.
    sql = " ".join(locator.split())
    if len(sql) > DESCRIPTION_CHARS:
        sql = sql[: DESCRIPTION_CHARS - 3] + "..."
    return f"`{sql}`"
