"""Supervisor: turns the research question into sub-tasks for the specialists.

Two modes, chosen from the state:
  - first round:    no critiques yet -> decompose the whole question (2-N sub-tasks)
  - research round: the critic returned "research" -> plan ONLY sub-tasks that fill
                    the gaps it named, given what earlier findings already cover

Its output is `plan` (a list of Subtask). graph.dispatch() turns each sub-task
into a parallel `Send` to the matching specialist. Earlier findings are never
discarded: `findings` has an append reducer, so a research round adds to them.

Planning stays on the strong model. A bad decomposition makes every later step
worse, and it is only one call per round.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import SUPERVISOR_GAP_SYSTEM, SUPERVISOR_PLAN_SYSTEM
from analyst.state import AgentName, AnalystState, Critique, Subtask, feedback_points
from analyst.tools.sql import table_catalog

log = logging.getLogger("analyst.agents.supervisor")

MAX_GAP_SUBTASKS = 3  # a research round fixes specific gaps; it shouldn't re-plan everything
FINDING_PREVIEW_CHARS = 300


class PlannedTask(BaseModel):
    agent: Literal["web", "rag", "data"] = Field(description="The specialist that should do it")
    instruction: str = Field(
        min_length=10, description="Self-contained, specific instruction for the specialist"
    )


class ResearchPlan(BaseModel):
    subtasks: list[PlannedTask] = Field(min_length=1, max_length=6)


class SupervisorAgent:
    """A LangGraph node: `SupervisorAgent()(state) -> {"plan", "research_rounds"}`."""

    def __init__(self, llm: BaseChatModel | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            # Low temperature: the same question should get the same plan.
            self._llm = get_llm(self.settings, tier="strong", temperature=0)
        return self._llm

    def __call__(self, state: AnalystState) -> dict:
        resources = describe_resources(self.settings)
        critiques = state.get("critiques", [])
        if not critiques:
            plan = self._first_round(state["question"], resources)
            log.info("supervisor: initial plan with %d sub-task(s)", len(plan))
            return {"plan": plan, "research_rounds": 0}

        # This node starts the new loop iteration, so it counts it (see graph.py).
        rounds = state.get("research_rounds", 0) + 1
        plan = self._gap_round(state, critiques[-1], rounds, resources)
        log.info("supervisor: research round %d with %d gap sub-task(s)", rounds, len(plan))
        return {"plan": plan, "research_rounds": rounds}

    # -- first round -----------------------------------------------------------
    def _first_round(self, question: str, resources: Resources) -> list[Subtask]:
        system = SUPERVISOR_PLAN_SYSTEM.format(
            max_subtasks=self.settings.max_subtasks, **resources.prompt_fields()
        )
        try:
            planned = self._ask(system, f"Research question: {question}")
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the whole run
            log.warning("supervisor: planning failed (%s); using one sub-task per specialist", exc)
            planned = [PlannedTask(agent=a, instruction=question) for a in resources.available]
        return self._finalize(planned, resources, prefix="t", limit=self.settings.max_subtasks)

    # -- research round --------------------------------------------------------
    def _gap_round(
        self, state: AnalystState, critique: Critique, rounds: int, resources: Resources
    ) -> list[Subtask]:
        # Gaps are the critic's missing topics; its free-text feedback is the
        # fallback when it didn't list any.
        gaps = critique.get("missing_topics") or feedback_points(critique) or ["unspecified gap"]
        done = "\n".join(
            f"- [{f['agent']}] {f['summary'][:FINDING_PREVIEW_CHARS]}"
            for f in state.get("findings", [])
        )
        system = SUPERVISOR_GAP_SYSTEM.format(
            max_subtasks=MAX_GAP_SUBTASKS, **resources.prompt_fields()
        )
        human = (
            f"Research question: {state['question']}\n\n"
            f"Findings so far (kept):\n{done or '(none)'}\n\n"
            "Gaps found by the critic:\n" + "\n".join(f"- {g}" for g in gaps)
        )
        try:
            planned = self._ask(system, human)
        except Exception as exc:  # noqa: BLE001
            # Without the LLM we can't tell which specialist fits each gap. The
            # web is the broadest default; the critic reviews the result anyway.
            log.warning("supervisor: gap planning failed (%s); sending each gap to web", exc)
            planned = [PlannedTask(agent="web", instruction=g) for g in gaps if len(g) >= 10]
        return self._finalize(planned, resources, prefix=f"r{rounds}-", limit=MAX_GAP_SUBTASKS)

    # -- shared ----------------------------------------------------------------
    def _ask(self, system: str, human: str) -> list[PlannedTask]:
        plan = self.llm.with_structured_output(ResearchPlan).invoke(
            [SystemMessage(system), HumanMessage(human)]
        )
        if plan is None:
            raise ValueError("empty structured output")
        return plan.subtasks

    def _finalize(
        self, planned: list[PlannedTask], resources: Resources, prefix: str, limit: int
    ) -> list[Subtask]:
        """Validate the LLM's plan: drop unavailable specialists and duplicates, cap the size,
        and assign ids. The prompt asks for all of this, but code enforces it."""
        seen: set[tuple[str, str]] = set()
        plan: list[Subtask] = []
        for task in planned:
            # A duplicate is the same instruction for the *same* specialist. The
            # same question sent to web and rag isn't redundant: they search
            # different sources.
            key = (task.agent, " ".join(task.instruction.lower().split()))
            if task.agent not in resources.available:
                log.warning(
                    "supervisor: dropping %s task (specialist unavailable): %s",
                    task.agent,
                    task.instruction,
                )
                continue
            if key in seen:
                continue
            seen.add(key)
            plan.append(
                {
                    "id": f"{prefix}{len(plan) + 1}",
                    "agent": task.agent,
                    "instruction": task.instruction.strip(),
                }
            )
            if len(plan) == limit:
                break
        return plan


# ---------------------------------------------------------------------------
# What the specialists can reach
# ---------------------------------------------------------------------------
class Resources:
    """A catalog of the data each specialist can actually use right now.

    Why tell the planner: "ask rag about risks" only makes sense if documents
    are indexed, and "ask data for revenue by channel" only if such a column
    exists. Listing real file and column names leads to sharper, answerable
    instructions, and marking a missing resource "unavailable" stops the
    planner from sending work to a specialist that would only fail.
    """

    def __init__(self, docs: list[str], tables: list[tuple[str, int, list[str]]]):
        self.docs = docs
        self.tables = tables

    @property
    def available(self) -> list[AgentName]:
        agents: list[AgentName] = ["web"]
        if self.docs:
            agents.append("rag")
        if self.tables:
            agents.append("data")
        return agents

    def prompt_fields(self) -> dict[str, str]:
        docs = ", ".join(self.docs) if self.docs else "(unavailable: no documents indexed)"
        tables = (
            "; ".join(
                f"{name} ({rows} rows: {', '.join(cols)})" for name, rows, cols in self.tables
            )
            if self.tables
            else "(unavailable: no database loaded)"
        )
        return {"docs": docs, "tables": tables}


def describe_resources(settings: Settings) -> Resources:
    # Documents = what the index actually contains, not just files in a folder.
    # A PDF that was never indexed can't be searched.
    docs: list[str] = []
    manifest = settings.index_dir / "manifest.json"
    if manifest.exists():
        try:
            docs = sorted(json.loads(manifest.read_text(encoding="utf-8")).get("files", {}))
        except (OSError, ValueError):
            log.warning("supervisor: unreadable index manifest at %s", manifest)
    tables = table_catalog(settings.db_path) if settings.db_path.exists() else []
    return Resources(docs, tables)
