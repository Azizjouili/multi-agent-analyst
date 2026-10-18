"""Placeholder nodes: no LLM calls, deterministic output.

Why build the graph with stubs first: the hard part of a multi-agent system is
often the *wiring* (fan-out, joins, loops, budgets), not the prompts. With
stubs we can test every route offline in milliseconds. Each later roadmap
step replaces one placeholder with a real agent that keeps the same signature:

    node(state) -> partial state update (dict)

Every placeholder logs what it does, so a demo run shows the order of
execution.
"""

from __future__ import annotations

import logging

from analyst.state import (
    AnalystState,
    Critique,
    Finding,
    Source,
    SpecialistInput,
    Subtask,
    make_source_id,
)

log = logging.getLogger("analyst.nodes")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
def supervisor(state: AnalystState) -> dict:
    # Replaced by agents/supervisor.py in step 8; kept so graph tests stay deterministic.
    critiques = state.get("critiques", [])
    if not critiques:
        # First round: one sub-task per specialist.
        plan: list[Subtask] = [
            {"id": "t1", "agent": "web", "instruction": f"Public info on: {state['question']}"},
            {"id": "t2", "agent": "rag", "instruction": f"Internal docs on: {state['question']}"},
            {"id": "t3", "agent": "data", "instruction": f"Numbers for: {state['question']}"},
        ]
        log.info("supervisor: initial plan with %d sub-tasks", len(plan))
        return {"plan": plan, "research_rounds": 0}

    # Research round: plan only the gaps the critic named. This node starts
    # the new loop iteration, so it is the one that counts it.
    rounds = state.get("research_rounds", 0) + 1
    gaps = critiques[-1]["missing_topics"] or ["unspecified gap"]
    plan = [
        {"id": f"r{rounds}-{i}", "agent": "web", "instruction": topic}
        for i, topic in enumerate(gaps, start=1)
    ]
    log.info("supervisor: research round %d, %d gap sub-task(s)", rounds, len(plan))
    return {"plan": plan, "research_rounds": rounds}


# ---------------------------------------------------------------------------
# Specialists: each receives a SpecialistInput through Send, not the full state
# ---------------------------------------------------------------------------
def _fake_finding(inp: SpecialistInput, title: str, locator: str) -> dict:
    task = inp["subtask"]
    source: Source = {
        "id": make_source_id(task["agent"], locator),
        "kind": task["agent"],
        "title": title,
        "locator": locator,
        "snippet": f"(placeholder evidence for {task['id']})",
    }
    finding: Finding = {
        "subtask_id": task["id"],
        "agent": task["agent"],
        "summary": f"Placeholder answer to '{task['instruction']}' [{source['id']}]",
        "source_ids": [source["id"]],
    }
    log.info("%s: finished sub-task %s", task["agent"], task["id"])
    # Lists and dicts here are *merged* into the shared state by the reducers,
    # so parallel specialists never overwrite each other.
    return {"findings": [finding], "sources": {source["id"]: source}}


def web_research(inp: SpecialistInput) -> dict:
    # Replaced by agents/web_research.py in step 7; kept so graph tests can run offline.
    return _fake_finding(inp, "Example web page", f"https://example.com/{inp['subtask']['id']}")


def rag(inp: SpecialistInput) -> dict:
    # Replaced by agents/rag.py in step 5; kept so graph tests can run offline.
    return _fake_finding(inp, "Annual report", f"annual_report.pdf#{inp['subtask']['id']}")


def data_analysis(inp: SpecialistInput) -> dict:
    # Replaced by agents/data_analyst.py in step 6; kept so graph tests can run offline.
    return _fake_finding(inp, "Sales query", f"SELECT ... -- {inp['subtask']['id']}")


# ---------------------------------------------------------------------------
# Writer, critic, human gate, finalize
# ---------------------------------------------------------------------------
def writer(state: AnalystState) -> dict:
    # Replaced by agents/writer.py in step 9; kept so graph tests stay deterministic.
    findings = state.get("findings", [])
    # A draft already exists, so this call is a rewrite. Count it so the
    # critic loop's budget can be enforced.
    is_rewrite = bool(state.get("draft"))
    revision_count = state.get("revision_count", 0) + (1 if is_rewrite else 0)
    body = "\n".join(f"- {f['summary']}" for f in findings)
    draft = f"# Report: {state['question']}\n\n{body}"
    log.info("writer: draft from %d findings (revision %d)", len(findings), revision_count)
    return {"draft": draft, "revision_count": revision_count}


def critic(state: AnalystState) -> dict:
    # Replaced by agents/critic.py in step 10; kept so graph tests stay deterministic.
    critique: Critique = {
        "scores": {"groundedness": 5, "coverage": 5, "citations": 5, "clarity": 5},
        "verdict": "accept",
        "feedback": ["Placeholder critic: always accepts."],
        "missing_topics": [],
    }
    log.info("critic: verdict=%s", critique["verdict"])
    return {"critiques": [critique]}


def human_gate(state: AnalystState) -> dict:
    # Replaced by agents/human_gate.py in step 11; kept so tests can run without a checkpointer.
    log.info("human_gate: auto-approving (placeholder)")
    return {"approved": True, "human_feedback": None}


def finalize(state: AnalystState) -> dict:
    # Replaced by agents/finalize.py in step 9.
    log.info("finalize: report ready")
    return {"final_report": state.get("draft", "")}
