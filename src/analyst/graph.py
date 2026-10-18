"""Builds the multi-agent graph.

    START -> supervisor --Send x N--> {web_research | rag | data_analysis} -> writer
    writer -> critic --accept / budget spent--> human_gate --approved--> finalize -> END
                     --revise-------------> writer                 \\--rejected--> writer
                     --research-----------> supervisor

All the control flow lives in this file. Agents only do work and return state
updates. Routing decisions are small pure functions (no LLM calls, no I/O),
so each one can be unit-tested directly.

Run `uv run python -m analyst.graph` to print a Mermaid diagram and run a demo
with human review off. `scripts/review.py` runs it with the human gate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from analyst.agents.critic import CriticAgent
from analyst.agents.data_analyst import DataAnalystAgent
from analyst.agents.finalize import finalize
from analyst.agents.human_gate import HumanGate
from analyst.agents.rag import RagAgent
from analyst.agents.supervisor import SupervisorAgent
from analyst.agents.web_research import WebResearchAgent
from analyst.agents.writer import WriterAgent
from analyst.config import Settings, get_settings
from analyst.state import AnalystState
from analyst.tracing import tracing_config

# Node names in one place, so routing code and tests can't drift apart through typos.
SUPERVISOR = "supervisor"
WEB = "web_research"
RAG = "rag"
DATA = "data_analysis"
WRITER = "writer"
CRITIC = "critic"
HUMAN_GATE = "human_gate"
FINALIZE = "finalize"

# Which node handles each Subtask.agent value.
AGENT_TO_NODE = {"web": WEB, "rag": RAG, "data": DATA}


def default_nodes(settings: Settings) -> dict[str, Callable[..., dict]]:
    """The node implementations used unless overridden. Real agents replace
    placeholders here one roadmap step at a time.

    A function, not a module-level dict, so agents receive the same settings
    the graph was built with.
    """
    return {
        SUPERVISOR: SupervisorAgent(settings=settings),  # step 8: real agent
        WEB: WebResearchAgent(settings=settings),  # step 7: real agent
        RAG: RagAgent(settings=settings),  # step 5: real agent
        DATA: DataAnalystAgent(settings=settings),  # step 6: real agent
        WRITER: WriterAgent(settings=settings),  # step 9: real agent
        CRITIC: CriticAgent(settings=settings),  # step 10: real agent
        HUMAN_GATE: HumanGate(settings=settings),  # step 11: interrupt() for a real person
        FINALIZE: finalize,  # step 9: deterministic, no LLM
    }


# ---------------------------------------------------------------------------
# Routing functions (pure: state in, destination out)
# ---------------------------------------------------------------------------
def dispatch(state: AnalystState) -> list[Send] | str:
    """Fan out: one `Send` per sub-task, all run in the same step (in parallel).

    Why `Send` rather than fixed edges supervisor -> each specialist: the
    plan's size and mix change from run to run (two web tasks and no data task,
    say). `Send(node, payload)` creates one task per item at runtime, each with
    its own private payload, which is the map step of map-reduce. The reduce
    happens through the `findings`/`sources` reducers and the join at `writer`.
    """
    plan = state.get("plan", [])
    if not plan:
        # Nothing to research (e.g. the question needs no evidence): skip to the writer.
        return WRITER
    return [
        Send(AGENT_TO_NODE[task["agent"]], {"question": state["question"], "subtask": task})
        for task in plan
    ]


def make_route_after_critic(settings: Settings) -> Callable[[AnalystState], str]:
    """Build the critic router with the loop budgets fixed at build time."""

    def route_after_critic(state: AnalystState) -> str:
        verdict = state["critiques"][-1]["verdict"]
        if verdict == "revise" and state.get("revision_count", 0) < settings.max_revisions:
            return WRITER
        if verdict == "research" and state.get("research_rounds", 0) < settings.max_research_rounds:
            return SUPERVISOR
        # "accept", or a budget is spent. Either way a human sees the draft
        # next, with the critique attached. Budgets guarantee the loop ends
        # even if the critic is never satisfied.
        return HUMAN_GATE

    return route_after_critic


def route_after_human(state: AnalystState) -> str:
    # Human rejection has no budget: the human, not a counter, decides when to stop.
    return FINALIZE if state.get("approved") else WRITER


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(
    nodes: Mapping[str, Callable[..., dict]] | None = None,
    *,
    settings: Settings | None = None,
    checkpointer: Any = None,
):
    """Compile the graph.

    `nodes` overrides individual node implementations, e.g.
    `build_graph({CRITIC: always_revise})`. Tests use it to script agent
    behaviour, and later steps use it to plug in real agents one at a time.
    `checkpointer` saves the state after every step. The human gate's
    `interrupt()` needs it: a paused run is just its last saved checkpoint.
    """
    settings = settings or get_settings()
    if settings.human_review and checkpointer is None and HUMAN_GATE not in (nodes or {}):
        # Fail at build time, not after the run has spent all its LLM calls
        # and reaches the gate, where interrupt() would find nowhere to save.
        raise ValueError(
            "HUMAN_REVIEW is on, so the graph needs a checkpointer (see sqlite_checkpointer). "
            "Set HUMAN_REVIEW=false to auto-approve instead."
        )
    impl = {**default_nodes(settings), **(nodes or {})}

    g = StateGraph(AnalystState)
    for name, fn in impl.items():
        g.add_node(name, fn)

    g.add_edge(START, SUPERVISOR)

    # The third argument lists every possible destination. LangGraph can't see
    # inside `dispatch`, so this is how it knows the edges (for validation and
    # for drawing the diagram).
    g.add_conditional_edges(SUPERVISOR, dispatch, [WEB, RAG, DATA, WRITER])

    # Join: each specialist has an edge to `writer`. LangGraph runs in
    # supersteps: all specialists sent in one step finish, their updates are
    # merged by the reducers, and only then does `writer` run, once. That
    # barrier is why no explicit "wait for all" code is needed.
    for specialist in (WEB, RAG, DATA):
        g.add_edge(specialist, WRITER)

    g.add_edge(WRITER, CRITIC)
    g.add_conditional_edges(
        CRITIC, make_route_after_critic(settings), [WRITER, SUPERVISOR, HUMAN_GATE]
    )
    g.add_conditional_edges(HUMAN_GATE, route_after_human, [FINALIZE, WRITER])
    g.add_edge(FINALIZE, END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Persistence helpers for paused runs
# ---------------------------------------------------------------------------
@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[Any]:
    """Open a SQLite-backed checkpointer, closing the connection afterwards.

    Why SQLite: a paused run must outlive the process. The human might answer
    hours later, from a new CLI call or another API worker. An in-memory saver
    would lose the run on exit; a single file needs no database server.
    """
    # Imported here so code that never pauses (tests, evals) doesn't need it.
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def run_config(
    thread_id: str, question: str | None = None, settings: Settings | None = None
) -> dict:
    """The RunnableConfig for one graph run. The same thread_id resumes the same run.

    With `question`, it also carries the tracing callbacks (when Langfuse is
    configured). This is the single place they are injected: LangChain hands
    a run's callbacks down to every node and LLM call. Without it (state
    lookups such as `pending_review`), nothing runs, so there's nothing to trace.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}
    if question is not None:
        config.update(tracing_config(question, thread_id, settings))
    return config


def pending_review(graph, config: dict) -> dict | None:
    """The human gate's payload if this run is paused there, else None.

    Reads the saved checkpoint, so it works in a fresh process: that is how a
    caller finds out, after a restart, that a run is waiting for a decision.
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        if task.name == HUMAN_GATE and task.interrupts:
            return task.interrupts[0].value
    return None


if __name__ == "__main__":
    import logging
    import os
    import sys

    from analyst.tracing import flush

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    # Review off: this demo runs straight through. scripts/review.py pauses.
    graph = build_graph(settings=get_settings().model_copy(update={"human_review": False}))

    print("Mermaid diagram (paste into https://mermaid.live):\n")
    print(graph.get_graph().draw_mermaid())

    question = " ".join(sys.argv[1:]) or "Summarize the company's revenue trend and key risks"
    print(f"\nDemo run (human review off): {question!r}\n")
    final: dict = {}
    config = run_config(f"demo-{os.getpid()}", question)  # traced when Langfuse keys are set
    # stream_mode="updates" yields {node_name: update} after every node, which
    # is how a UI or the API can show progress live.
    for step in graph.stream({"question": question}, config, stream_mode="updates"):
        for node, update in step.items():
            print(f"[{node}] wrote: {sorted(update)}")
            final.update(update)
    flush()
    print("\nFinal report:\n" + final["final_report"])
