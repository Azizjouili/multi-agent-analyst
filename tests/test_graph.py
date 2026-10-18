"""Routing tests for the graph wiring, run with placeholder or scripted nodes (no LLM)."""

import operator
import threading
from typing import Annotated, TypedDict

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from analyst.agents import placeholders
from analyst.config import Settings
from analyst.graph import (
    CRITIC,
    DATA,
    FINALIZE,
    HUMAN_GATE,
    RAG,
    SUPERVISOR,
    WEB,
    WRITER,
    dispatch,
    make_route_after_critic,
)
from analyst.graph import build_graph as _build_graph
from analyst.state import make_source_id, merge_dicts


def _settings(**overrides) -> Settings:
    return Settings(llm_provider="fake", _env_file=None, **overrides)


# These tests check wiring, not agents, so real agents are swapped for their
# placeholders. That keeps the tests offline, with no index or model needed.
OFFLINE_NODES = {
    SUPERVISOR: placeholders.supervisor,
    WEB: placeholders.web_research,
    RAG: placeholders.rag,
    DATA: placeholders.data_analysis,
    WRITER: placeholders.writer,
    CRITIC: placeholders.critic,
}


def build_graph(nodes=None, **kwargs):
    return _build_graph({**OFFLINE_NODES, **(nodes or {})}, **kwargs)


def run(graph, question="Q?"):
    """Run the graph; return (node names in execution order, final state)."""
    visited, final = [], {}
    for mode, chunk in graph.stream({"question": question}, stream_mode=["updates", "values"]):
        if mode == "updates":
            visited.extend(chunk.keys())
        else:
            final = chunk
    return visited, final


def critic_with_verdicts(*verdicts, missing=("gap topic",)):
    """A critic that returns the given verdicts in order, repeating the last one."""
    calls = {"n": 0}

    def _critic(state):
        verdict = verdicts[min(calls["n"], len(verdicts) - 1)]
        calls["n"] += 1
        return {
            "critiques": [
                {
                    "scores": {},
                    "verdict": verdict,
                    "feedback": f"scripted {verdict}",
                    "missing_topics": list(missing) if verdict == "research" else [],
                }
            ]
        }

    return _critic


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_source_id_is_stable_and_namespaced():
    a = make_source_id("web", "https://example.com/x")
    assert a == make_source_id("web", "https://example.com/x")
    assert a.startswith("web:")
    assert a != make_source_id("web", "https://example.com/y")


def test_merge_dicts_handles_none_and_overlap():
    assert merge_dicts(None, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_dispatch_creates_one_send_per_subtask():
    plan = [
        {"id": "t1", "agent": "web", "instruction": "x"},
        {"id": "t2", "agent": "web", "instruction": "y"},
        {"id": "t3", "agent": "data", "instruction": "z"},
    ]
    sends = dispatch({"question": "Q?", "plan": plan})
    assert [s.node for s in sends] == [WEB, WEB, DATA]
    assert sends[0].arg == {"question": "Q?", "subtask": plan[0]}  # private payload


def test_dispatch_with_empty_plan_skips_to_writer():
    assert dispatch({"question": "Q?", "plan": []}) == WRITER


@pytest.mark.parametrize(
    ("verdict", "revisions", "rounds", "expected"),
    [
        ("accept", 0, 0, HUMAN_GATE),
        ("revise", 0, 0, WRITER),
        ("revise", 2, 0, HUMAN_GATE),  # revision budget spent
        ("research", 0, 0, SUPERVISOR),
        ("research", 0, 1, HUMAN_GATE),  # research budget spent
    ],
)
def test_route_after_critic_respects_budgets(verdict, revisions, rounds, expected):
    route = make_route_after_critic(_settings(max_revisions=2, max_research_rounds=1))
    state = {
        "question": "Q?",
        "critiques": [{"verdict": verdict}],
        "revision_count": revisions,
        "research_rounds": rounds,
    }
    assert route(state) == expected


# ---------------------------------------------------------------------------
# Whole-graph runs
# ---------------------------------------------------------------------------
def test_happy_path_runs_every_stage_once():
    visited, final = run(build_graph(settings=_settings()))

    assert visited[0] == SUPERVISOR
    assert set(visited[1:4]) == {WEB, RAG, DATA}  # order within the fan-out may vary
    assert visited[4:] == [WRITER, CRITIC, HUMAN_GATE, FINALIZE]

    # The reducers merged all three parallel updates instead of keeping only the last.
    assert {f["agent"] for f in final["findings"]} == {"web", "rag", "data"}
    assert len(final["sources"]) == 3
    assert final["final_report"].startswith("# Report")


def test_specialists_really_run_in_parallel():
    # Each specialist blocks until all three are inside it at the same time.
    # If LangGraph ran them one after another, the barrier would time out.
    barrier = threading.Barrier(3, timeout=5)

    def wrap(fn):
        def _node(inp):
            barrier.wait()
            return fn(inp)

        return _node

    graph = build_graph(
        {
            WEB: wrap(placeholders.web_research),
            RAG: wrap(placeholders.rag),
            DATA: wrap(placeholders.data_analysis),
        },
        settings=_settings(),
    )
    _, final = run(graph)
    assert len(final["findings"]) == 3


def test_writer_waits_for_all_branches_and_runs_once():
    seen = []

    def spy_writer(state):
        seen.append(len(state.get("findings", [])))
        return placeholders.writer(state)

    def plan_two_web_one_rag(state):
        return {
            "plan": [
                {"id": "a", "agent": "web", "instruction": "a"},
                {"id": "b", "agent": "web", "instruction": "b"},
                {"id": "c", "agent": "rag", "instruction": "c"},
            ]
        }

    visited, _ = run(
        build_graph({SUPERVISOR: plan_two_web_one_rag, WRITER: spy_writer}, settings=_settings())
    )
    assert visited.count(WEB) == 2  # same node, two independent Send invocations
    assert seen == [3]  # the writer ran once, after all three findings had arrived


def test_revise_loop_stops_at_budget():
    graph = build_graph(
        {CRITIC: critic_with_verdicts("revise")}, settings=_settings(max_revisions=2)
    )
    visited, final = run(graph)

    assert visited.count(WRITER) == 3  # first draft + 2 revisions
    assert visited.count(CRITIC) == 3
    assert final["revision_count"] == 2
    assert visited[-2:] == [HUMAN_GATE, FINALIZE]  # the budget forced it to the human


def test_research_loop_replans_then_accepts():
    graph = build_graph(
        {CRITIC: critic_with_verdicts("research", "accept")},
        settings=_settings(max_research_rounds=1),
    )
    visited, final = run(graph)

    assert visited.count(SUPERVISOR) == 2
    assert final["research_rounds"] == 1
    # Round 2 planned only the gap; findings from both rounds accumulate.
    assert final["plan"] == [{"id": "r1-1", "agent": "web", "instruction": "gap topic"}]
    assert len(final["findings"]) == 4


def test_research_loop_stops_at_budget():
    graph = build_graph(
        {CRITIC: critic_with_verdicts("research")}, settings=_settings(max_research_rounds=1)
    )
    visited, _ = run(graph)
    assert visited.count(SUPERVISOR) == 2
    assert visited[-1] == FINALIZE


def test_human_rejection_sends_draft_back_to_writer():
    decisions = iter([(False, "Add more on risks"), (True, None)])

    def scripted_human(state):
        approved, feedback = next(decisions)
        return {"approved": approved, "human_feedback": feedback}

    visited, final = run(build_graph({HUMAN_GATE: scripted_human}, settings=_settings()))

    assert visited.count(HUMAN_GATE) == 2
    assert visited.count(WRITER) == 2
    assert final["approved"] is True
    assert final["revision_count"] == 1


# ---------------------------------------------------------------------------
# Why reducers exist: a counter-example
# ---------------------------------------------------------------------------
def _fan_out_graph(findings_annotation):
    class S(TypedDict, total=False):
        findings: findings_annotation

    def worker(_):
        return {"findings": ["x"]}

    g = StateGraph(S)
    g.add_node("worker", worker)
    g.add_conditional_edges(START, lambda s: [Send("worker", {}), Send("worker", {})], ["worker"])
    g.add_edge("worker", END)
    return g.compile()


def test_parallel_writes_without_reducer_are_rejected():
    with pytest.raises(InvalidUpdateError):
        _fan_out_graph(list[str]).invoke({})


def test_parallel_writes_with_reducer_are_merged():
    out = _fan_out_graph(Annotated[list[str], operator.add]).invoke({})
    assert out["findings"] == ["x", "x"]
