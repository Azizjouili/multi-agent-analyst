"""Human gate: a person approves the report, or rejects it with feedback.

    critic --accept / budget spent--> human_gate --interrupt()--> (graph pauses)
    caller: Command(resume="approve")             --> finalize
            Command(resume="Add more on risks")   --> writer (feedback = top priority)

How interrupt() works: the first time the node runs, `interrupt(payload)`
raises a special exception. LangGraph catches it, saves the state with the
checkpointer, and returns to the caller with the payload under
"__interrupt__". The process may then exit. On resume, LangGraph loads the
checkpoint and runs this node again *from the top*. This time interrupt()
returns the resume value instead of raising.

Because the node runs twice, everything before interrupt() must be cheap,
deterministic and free of side effects. Building the payload is pure code here
(no LLM calls, no I/O), so running it twice is harmless.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from analyst.agents.finalize import render_final_report
from analyst.config import Settings, get_settings
from analyst.state import AnalystState

log = logging.getLogger("analyst.agents.human_gate")

APPROVE_WORDS = {"approve", "approved", "yes", "y"}
# A rejection with no comment still needs something the writer can act on.
NO_COMMENT_FEEDBACK = (
    "The reviewer rejected the draft without comments: improve accuracy and clarity."
)


def parse_decision(value: Any) -> tuple[bool, str | None]:
    """Turn a resume value into (approved, feedback).

    Two forms are accepted:
      - a string: "approve" (or yes/y) approves; any other text is the rejection
        feedback. This is what a person types in the CLI.
      - a dict {"approved": bool, "feedback": str | None}: the structured form
        an API client sends (step 13).
    """
    if isinstance(value, dict):
        approved = bool(value.get("approved"))
        feedback = str(value.get("feedback") or "").strip()
    elif isinstance(value, str):
        feedback = value.strip()
        approved = feedback.lower() in APPROVE_WORDS
    else:
        raise TypeError(f"human gate: unsupported resume value {value!r}")
    if approved:
        return True, None
    return False, feedback or NO_COMMENT_FEEDBACK


def review_payload(state: AnalystState) -> dict[str, Any]:
    """What the human needs to decide, sent to the caller with the interrupt.

    `preview` is the draft exactly as the reader will get it (numbered
    citations and a Sources list), so the person approves the real output,
    not the internal form with hash ids.
    """
    critiques = state.get("critiques", [])
    preview, _ = render_final_report(state.get("draft", ""), state.get("sources", {}))
    return {
        "question": state["question"],
        "draft": state.get("draft", ""),
        "preview": preview,
        "critique": critiques[-1] if critiques else None,
        "revision_count": state.get("revision_count", 0),
        "research_rounds": state.get("research_rounds", 0),
    }


class HumanGate:
    """A LangGraph node: `HumanGate()(state) -> {"approved", "human_feedback"}`."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def __call__(self, state: AnalystState) -> dict:
        if not self.settings.human_review:
            # Batch mode (evals, scripted runs): nobody is there to answer, and
            # an interrupt would just leave the run paused forever.
            log.info("human_gate: review disabled, auto-approving")
            return {"approved": True, "human_feedback": None}

        # Pauses here on the first pass; returns the caller's answer on resume.
        approved, feedback = parse_decision(interrupt(review_payload(state)))
        log.info("human_gate: %s", "approved" if approved else f"rejected: {feedback}")
        # The writer reads `human_feedback` only while `approved` is False and
        # puts it first in its prompt, ahead of the critic's points.
        return {"approved": approved, "human_feedback": feedback}
