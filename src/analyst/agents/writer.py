"""Writer: synthesizes all findings into one cited Markdown report.

Three situations, chosen from the state:
  - first draft:   no draft yet
  - revision:      a draft exists and there's feedback to address, from
                     - the human reviewer (rejected at the gate): highest priority
                     - the critic ("revise" or "research" verdict)
                   and, after a research round, new findings to work in
  - (both write the complete report, not a patch)

Citations: the LLM sees sources as [S1], [S2], ... and its output is converted
back to stable source ids before it is stored. So the draft in state always
cites ids that exist in state["sources"]: invented labels are dropped here,
and finalize drops anything else that slipped through.

Uses the strong model: the report is the product, and one call per draft is
cheap compared with what feeds it.
"""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from analyst.agents.citations import relabel_source_ids, resolve_citations
from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import WRITER_SYSTEM
from analyst.state import AnalystState, Finding, Source, feedback_points

log = logging.getLogger("analyst.agents.writer")

MAX_WORDS = 600
SOURCE_EXCERPT_CHARS = 400  # findings carry the substance; excerpts add context for the writer
LOCATOR_CHARS = 120

# A bibliography heading the model added despite the instructions. It must be a
# heading made of *only* that word, so "## Sources of revenue growth" stays.
_BIBLIOGRAPHY = re.compile(
    r"\n#{1,6}[ \t]*(sources|references|bibliography|citations)[ \t]*:?[ \t]*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)


class WriterAgent:
    """A LangGraph node: `WriterAgent()(state) -> {"draft", "revision_count"}`."""

    def __init__(self, llm: BaseChatModel | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            # A little temperature for readable prose. Facts are pinned by the
            # citation rules, not by the sampling temperature.
            self._llm = get_llm(self.settings, tier="strong", temperature=0.3)
        return self._llm

    def __call__(self, state: AnalystState) -> dict:
        findings = state.get("findings", [])
        sources = state.get("sources", {})
        previous = state.get("draft")
        # A draft already exists, so this call is a rewrite. Count it so the
        # critic loop's budget can be enforced (see graph.route_after_critic).
        revision_count = state.get("revision_count", 0) + (1 if previous else 0)

        labels = _assign_labels(findings, sources)  # source id -> "S1"
        feedback = collect_feedback(state) if previous else []
        messages = [
            SystemMessage(WRITER_SYSTEM.format(max_words=MAX_WORDS)),
            HumanMessage(_brief(state["question"], findings, sources, labels, previous, feedback)),
        ]

        try:
            text = self.llm.invoke(messages).text.strip()
        except Exception as exc:  # noqa: BLE001 - keep the pipeline moving
            log.exception("writer: LLM call failed")
            if previous:
                # Keep the last good draft. The revision still counts, so the
                # loop budget advances and can't spin on a broken model.
                return {"draft": previous, "revision_count": revision_count}
            return {
                "draft": fallback_report(state["question"], findings, exc),
                "revision_count": revision_count,
            }

        text = _BIBLIOGRAPHY.sub("", text).rstrip()
        draft, cited = resolve_citations(text, {label: sid for sid, label in labels.items()})
        log.info(
            "writer: %s (%d words, cites %d of %d sources, %d feedback point(s))",
            f"revision {revision_count}" if previous else "first draft",
            len(draft.split()), len(cited), len(sources), len(feedback),
        )  # fmt: skip
        return {"draft": draft, "revision_count": revision_count}


# ---------------------------------------------------------------------------
# Building the prompt
# ---------------------------------------------------------------------------
def _assign_labels(findings: list[Finding], sources: dict[str, Source]) -> dict[str, str]:
    """Label sources S1..Sn: first those cited by findings (in finding order), then the rest.

    Ids that aren't in `sources` get no label, so they can't be cited.
    """
    ordered = [sid for f in findings for sid in f["source_ids"] if sid in sources]
    ordered += list(sources)
    return {sid: f"S{i}" for i, sid in enumerate(dict.fromkeys(ordered), start=1)}


def collect_feedback(state: AnalystState) -> list[str]:
    """The points a revision must address, most important first.

    Human feedback comes first and is marked as the priority: a person
    rejected the draft, and their judgment outranks the critic's. Critic
    feedback is included only when the critic asked for changes. An "accept"
    critique (the one that let the draft reach the human) has nothing to fix.
    """
    points: list[str] = []
    if state.get("human_feedback") and not state.get("approved"):
        points.append(f"[Human reviewer - highest priority] {state['human_feedback']}")
    critiques = state.get("critiques", [])
    if critiques and critiques[-1]["verdict"] in ("revise", "research"):
        last = critiques[-1]
        points += [f"[Critic] {point}" for point in feedback_points(last)]
        points += [f"[Critic] Cover: {topic}" for topic in last.get("missing_topics", [])]
    return points


def _brief(
    question: str,
    findings: list[Finding],
    sources: dict[str, Source],
    labels: dict[str, str],
    previous: str | None,
    feedback: list[str],
) -> str:
    finding_lines = [
        f"[F{i}] ({f['agent']}, sub-task {f['subtask_id']}) "
        f"{relabel_source_ids(f['summary'], labels)}"
        for i, f in enumerate(findings, start=1)
    ]
    source_lines = []
    for sid, label in labels.items():
        s = sources[sid]
        locator = " ".join(s["locator"].split())[:LOCATOR_CHARS]
        excerpt = " ".join(s["snippet"].split())[:SOURCE_EXCERPT_CHARS]
        source_lines.append(
            f"[{label}] ({s['kind']}) {s['title']} - {locator}\n    excerpt: {excerpt}"
        )

    parts = [
        f"Research question: {question}",
        "Findings from the specialist agents (summaries ending in NOT FOUND or ERROR are gaps "
        "to mention under Limitations):\n" + ("\n".join(finding_lines) or "(no findings)"),
        "Sources you may cite:\n" + ("\n".join(source_lines) or "(no sources)"),
    ]
    if previous:
        numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(feedback, start=1))
        parts += [
            "Your previous draft:\n" + relabel_source_ids(previous, labels),
            "Revise it. Address EVERY feedback point below, and work in any findings that "
            "are new since the previous draft. Return the complete revised report.\n"
            "Feedback:\n" + (numbered or "(no explicit feedback: improve accuracy and clarity)"),
        ]
    return "\n\n".join(parts)


def fallback_report(question: str, findings: list[Finding], exc: BaseException) -> str:
    """A plain report assembled from the findings when the writer model is unavailable.

    The findings already carry valid citations, so the result is still a
    faithful, cited (if unpolished) report the critic and human can review.
    """
    body = "\n".join(f"- {f['summary']}" for f in findings) or "- No findings were produced."
    return (
        f"# Research report: {question}\n\n## Findings\n{body}\n\n## Limitations\n"
        f"- The writer model was unavailable ({short_error(exc)}); "
        "this report lists the specialists' findings verbatim."
    )
