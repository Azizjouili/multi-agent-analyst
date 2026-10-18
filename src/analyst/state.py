"""Shared state that flows between agents.

In LangGraph every node receives the current state and returns a *partial
update* (a dict containing only the keys it changed). For each key, the graph
decides how to apply that update:

- No annotation: last write wins. Right for values with a single owner, such as
  `draft` (only the writer sets it) or `plan` (only the supervisor sets it).
- `Annotated[T, reducer]`: the graph calls `reducer(old, new)`. This is needed
  for keys that several nodes write *in the same step*. The three specialists
  run in parallel and all write `findings` and `sources`; without a reducer,
  LangGraph would raise `InvalidUpdateError`, because it refuses to pick a
  winner silently.
"""

from __future__ import annotations

import hashlib
import operator
from typing import Annotated, Literal, NotRequired, Required, TypedDict

AgentName = Literal["web", "rag", "data"]
Verdict = Literal["accept", "revise", "research"]


class Subtask(TypedDict):
    """One unit of work the supervisor assigns to a specialist."""

    id: str
    agent: AgentName
    instruction: str


class Source(TypedDict):
    """A piece of evidence a claim can cite."""

    id: str  # stable, collision-free id; see make_source_id()
    kind: AgentName
    title: str
    locator: str  # URL, "file.pdf#p4", or the SQL query
    snippet: str  # the exact evidence text the claim rests on


class Finding(TypedDict):
    """A specialist's answer to one sub-task."""

    subtask_id: str
    agent: AgentName
    summary: str  # answer text with inline [source_id] citations
    source_ids: list[str]


class Critique(TypedDict):
    """The critic's review of one draft."""

    scores: dict[str, int]  # groundedness, coverage, clarity (1-5)
    verdict: Verdict
    feedback: list[str]  # specific, actionable points; read it with feedback_points()
    missing_topics: list[str]  # what the supervisor should research on a "research" loop
    # Deterministic citation check: counts of uncited sentences and unknown ids.
    checks: NotRequired[dict[str, int]]
    # True when a budget forced the verdict (e.g. "revise" turned into "accept").
    budget_exhausted: NotRequired[bool]


def feedback_points(critique: Critique) -> list[str]:
    """The critique's feedback as a list of points.

    Tolerates a plain string, the format before feedback became a list. A run
    saved by an older version (checkpoints are persisted from step 11) can
    then still be resumed.
    """
    feedback = critique.get("feedback") or []
    return [feedback] if isinstance(feedback, str) else list(feedback)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for `sources`: union of two dicts, the newer value wins on equal keys.

    Equal keys mean the *same* source (ids are content hashes), so overwriting
    is harmless. It deduplicates a page that two agents both cited.
    """
    return {**(left or {}), **(right or {})}


def make_source_id(kind: AgentName, locator: str) -> str:
    """Return a deterministic id such as 'web:3f9a1c2b'.

    Why a hash and not a counter ("S1", "S2"): parallel agents can't
    coordinate a shared counter, so two of them would both create "S1". Hashing
    the locator gives every source an id that is unique and the same on every
    run, so citations stay valid across revisions. The finalize step renumbers
    them to [1], [2], ... for readers.
    """
    digest = hashlib.sha1(locator.encode("utf-8")).hexdigest()[:8]
    return f"{kind}:{digest}"


class AnalystState(TypedDict, total=False):
    """The state of one run of the graph.

    `total=False`: a run starts with only `question`; the other keys appear as
    nodes write them. Nodes therefore read with `state.get(key, default)`.
    The reducer-backed lists and dicts start empty automatically.
    """

    question: Required[str]

    # Supervisor
    plan: list[Subtask]  # replaced (not appended) on each research round
    research_rounds: int  # incremented by the supervisor each time it re-plans

    # Specialists (run in parallel, so these keys need reducers)
    findings: Annotated[list[Finding], operator.add]  # accumulate across rounds
    sources: Annotated[dict[str, Source], merge_dicts]

    # Writer
    draft: str
    revision_count: int  # incremented by the writer on every rewrite

    # Critic. The full history, not just the latest critique, so the writer
    # can see what was already asked for and evals can inspect the loop.
    critiques: Annotated[list[Critique], operator.add]

    # Human gate
    human_feedback: str | None
    approved: bool

    # Finalize
    final_report: str


class SpecialistInput(TypedDict):
    """The private payload each specialist receives through `Send`.

    Why not the whole AnalystState: a specialist needs only the question for
    context and its own sub-task. A smaller input keeps prompts short and the
    agent focused. It also lets two sub-tasks for the same agent run as two
    independent parallel invocations of that node.
    """

    question: str
    subtask: Subtask
