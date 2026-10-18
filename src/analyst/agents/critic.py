"""Critic: reviews each draft and drives the reflection loop.

    draft -> 1. deterministic citation check  -- problems? --> "revise" (no LLM call)
          -> 2. LLM judge (rubric + verdict)
          -> 3. budget enforcement             --> final verdict

1. The citation check is code: every factual sentence must cite a source, and
   every cited id must exist in state["sources"]. A rule catches a
   hallucinated or missing citation reliably and for free; an LLM judge might
   not notice. If it fails, the judge is skipped: the draft must change
   anyway, so there's no point spending a call on it (like a linter before
   code review).
2. The judge scores groundedness, coverage and clarity against the evidence
   and picks accept / revise / research, with actionable feedback and, for
   research, the missing topics that drive the supervisor's next round.
3. Budgets turn the judge's verdict into a guarantee. With no revisions left,
   "revise" becomes "accept"; with no research rounds left, "research"
   becomes "revise" (so the gaps get stated under Limitations) or "accept".
   The open feedback is kept, so the human at the gate sees what the critic
   still dislikes. graph.route_after_critic checks the same budgets again:
   defense in depth.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.agents.citations import SOURCE_ID, relabel_source_ids
from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import CRITIC_SYSTEM
from analyst.state import AnalystState, Critique, Source, Verdict

log = logging.getLogger("analyst.agents.critic")

MIN_CLAIM_WORDS = 6  # shorter fragments ("Revenue trend") aren't claims that need a source
MAX_FEEDBACK = 6
MAX_TOPICS = 3
QUOTE_CHARS = 120
SOURCE_EXCERPT_CHARS = 600
ACCEPT_MIN_SCORE = 3  # an "accept" with any score below this contradicts itself


# ---------------------------------------------------------------------------
# 1. Deterministic citation check
# ---------------------------------------------------------------------------
@dataclass
class CitationReport:
    claim_sentences: int = 0
    uncited: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.uncited and not self.unknown_ids

    def counts(self) -> dict[str, int]:
        return {
            "claim_sentences": self.claim_sentences,
            "uncited": len(self.uncited),
            "unknown_ids": len(self.unknown_ids),
        }

    def feedback(self) -> list[str]:
        points = [
            f'Add a citation to this claim or remove it: "{_quote(s)}"' for s in self.uncited[:4]
        ]
        if len(self.uncited) > 4:
            points.append(f"{len(self.uncited) - 4} more sentences state facts without a citation.")
        if self.unknown_ids:
            points.append(
                f"{len(self.unknown_ids)} citation(s) pointed to sources that don't exist. "
                "Cite only the listed sources, and only for claims they support."
            )
        return points


_HEADING = re.compile(r"^\s*#{1,6}\s*(.*)$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BOLD_LABEL = re.compile(r"^\*\*[^*]{1,60}\*\*:?\s*")  # "**Supplier concentration:** ..."
# Split after . ! ? when the next sentence starts with a capital, digit or quote.
# "17.98%" and "e.g. revenue" don't split; a rare "U.S. Tariffs" might, harmlessly.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])")


def check_citations(draft: str, sources: dict[str, Source]) -> CitationReport:
    """Find factual sentences without a citation, and cited ids that don't exist.

    Skipped: headings, intro lines ending in ':', table rows, short fragments,
    and the Limitations section. Statements about missing evidence have no
    source to cite.
    """
    report = CitationReport()
    report.unknown_ids = sorted({sid for sid in SOURCE_ID.findall(draft) if sid not in sources})

    in_limitations = False
    for line in draft.splitlines():
        if heading := _HEADING.match(line):
            in_limitations = heading.group(1).strip().lower().startswith("limitation")
            continue
        text = line.strip()
        if in_limitations or not text or text.startswith("|"):
            continue
        text = _BOLD_LABEL.sub("", _LIST_MARKER.sub("", text))
        for sentence in _SENTENCE_END.split(text):
            words = re.findall(r"\w+", SOURCE_ID.sub("", sentence))
            if len(words) < MIN_CLAIM_WORDS or sentence.rstrip().endswith(":"):
                continue
            report.claim_sentences += 1
            if not SOURCE_ID.search(sentence):
                report.uncited.append(sentence.strip())
    return report


def _quote(sentence: str) -> str:
    return sentence if len(sentence) <= QUOTE_CHARS else sentence[: QUOTE_CHARS - 3] + "..."


# ---------------------------------------------------------------------------
# 2. The LLM judge's structured output
# ---------------------------------------------------------------------------
class Scores(BaseModel):
    groundedness: int = Field(description="1-5")
    coverage: int = Field(description="1-5")
    clarity: int = Field(description="1-5")


class Judgment(BaseModel):
    # Lengths and ranges are enforced in code, not in the schema. A judge
    # returning 7 feedback points shouldn't fail validation (and so count as
    # "critic unavailable"): we keep the first 6 instead.
    scores: Scores
    verdict: Literal["accept", "revise", "research"]
    feedback: list[str] = Field(default_factory=list, description="Actionable points, at most 6")
    missing_topics: list[str] = Field(
        default_factory=list, description="Only for research: at most 3 searchable topics"
    )


# ---------------------------------------------------------------------------
# 3. Budgets
# ---------------------------------------------------------------------------
def apply_budgets(
    verdict: Verdict,
    feedback: list[str],
    missing_topics: list[str],
    revisions_left: int,
    research_left: int,
) -> tuple[Verdict, list[str], list[str], bool]:
    """Turn a wanted verdict into one the remaining budgets allow.

    Returns (verdict, feedback, missing_topics, budget_exhausted).
    """
    if verdict == "research" and research_left <= 0:
        if revisions_left > 0:
            # Can't gather more evidence, but the writer can be honest about
            # the gap instead of glossing over it.
            note = "No research rounds left: state these gaps under Limitations: " + "; ".join(
                missing_topics
            )
            return "revise", [*feedback, note], [], True
        return "accept", feedback, [], True
    if verdict == "revise" and revisions_left <= 0:
        return "accept", feedback, [], True
    return verdict, feedback, missing_topics, False


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------
class CriticAgent:
    """A LangGraph node: `CriticAgent()(state) -> {"critiques": [critique]}`."""

    def __init__(self, llm: BaseChatModel | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            # Strong tier: judging needs the best model. get_llm adds the
            # fallback, so a quota error on the strong model is retried on the
            # fast one. Temperature 0: the same draft should get the same verdict.
            self._llm = get_llm(self.settings, tier="strong", temperature=0)
        return self._llm

    def __call__(self, state: AnalystState) -> dict:
        draft = state.get("draft", "")
        sources = state.get("sources", {})
        revisions_left = self.settings.max_revisions - state.get("revision_count", 0)
        research_left = self.settings.max_research_rounds - state.get("research_rounds", 0)

        report = check_citations(draft, sources)
        scores: dict[str, int] = {}
        missing: list[str] = []

        if revisions_left <= 0 and research_left <= 0:
            # Nothing the critic says can change the draft any more: skip the
            # judge call and hand over to the human with the mechanical issues.
            verdict, feedback = "accept", report.feedback()
        elif not report.ok:
            verdict, feedback = "revise", report.feedback()
        else:
            verdict, feedback, missing, scores = self._judge(state, revisions_left, research_left)

        verdict, feedback, missing, exhausted = apply_budgets(
            verdict, feedback, missing, revisions_left, research_left
        )
        exhausted = exhausted or (revisions_left <= 0 and research_left <= 0)

        critique: Critique = {
            "scores": scores,
            "verdict": verdict,
            "feedback": feedback,
            "missing_topics": missing,
            "checks": report.counts(),
            "budget_exhausted": exhausted,
        }
        log.info(
            "critic: verdict=%s scores=%s uncited=%d unknown_ids=%d%s",
            verdict, scores or "-", len(report.uncited), len(report.unknown_ids),
            " (budget exhausted)" if exhausted else "",
        )  # fmt: skip
        return {"critiques": [critique]}

    def _judge(
        self, state: AnalystState, revisions_left: int, research_left: int
    ) -> tuple[Verdict, list[str], list[str], dict[str, int]]:
        allowed = (
            ["accept"] + ["revise"] * (revisions_left > 0) + ["research"] * (research_left > 0)
        )
        try:
            judgment = self.llm.with_structured_output(Judgment).invoke(
                [
                    SystemMessage(CRITIC_SYSTEM.format(allowed=", ".join(allowed))),
                    HumanMessage(_review_brief(state)),
                ]
            )
            if judgment is None:
                raise ValueError("empty structured output")
        except Exception as exc:  # noqa: BLE001
            # Even the fallback model failed. Don't block the run: let the
            # human reviewer, the final backstop, judge the draft.
            log.exception("critic: judge failed; accepting for human review")
            return (
                "accept",
                [f"Automated review unavailable ({type(exc).__name__}); human review needed."],
                [],
                {},
            )

        scores = {k: max(1, min(5, v)) for k, v in judgment.scores.model_dump().items()}
        feedback = [f.strip() for f in judgment.feedback if f.strip()][:MAX_FEEDBACK]
        missing = [t.strip() for t in judgment.missing_topics if t.strip()][:MAX_TOPICS]
        verdict: Verdict = judgment.verdict

        # Keep the verdict consistent with the judge's own output.
        if verdict == "research" and not missing:
            verdict = "revise"  # research without topics gives the supervisor nothing to plan
        if verdict == "accept" and min(scores.values()) < ACCEPT_MIN_SCORE:
            verdict = "revise"
            low = ", ".join(f"{k}={v}" for k, v in scores.items() if v < ACCEPT_MIN_SCORE)
            feedback = feedback or [f"Improve the weakest criteria ({low})."]
        return verdict, feedback, missing if verdict == "research" else [], scores


def _review_brief(state: AnalystState) -> str:
    """The question, the evidence and the draft, with sources relabeled [S1]..[Sn]."""
    sources = state.get("sources", {})
    labels = {sid: f"S{i}" for i, sid in enumerate(sources, start=1)}
    findings = "\n".join(
        f"- ({f['agent']}) {relabel_source_ids(f['summary'], labels)}"
        for f in state.get("findings", [])
    )
    excerpts = "\n".join(
        f"[{labels[sid]}] ({s['kind']}) {s['title']}\n    "
        + " ".join(s["snippet"].split())[:SOURCE_EXCERPT_CHARS]
        for sid, s in sources.items()
    )
    return (
        f"Research question: {state['question']}\n\n"
        f"Findings:\n{findings or '(none)'}\n\n"
        f"Source excerpts:\n{excerpts or '(none)'}\n\n"
        f"Draft report to review:\n{relabel_source_ids(state.get('draft', ''), labels)}"
    )
