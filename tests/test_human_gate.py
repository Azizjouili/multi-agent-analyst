"""Offline tests for the human gate: interrupt/resume, persistence, and the CLI."""

import re

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from analyst.agents import placeholders
from analyst.agents.human_gate import NO_COMMENT_FEEDBACK, parse_decision
from analyst.agents.writer import WriterAgent
from analyst.config import Settings, get_settings
from analyst.graph import (
    CRITIC,
    DATA,
    FINALIZE,
    HUMAN_GATE,
    RAG,
    SUPERVISOR,
    WEB,
    WRITER,
    build_graph,
    pending_review,
    run_config,
    sqlite_checkpointer,
)
from analyst.llm import FakeChatModel
from scripts import review

QUESTION = "Summarize the company's revenue trend and key risks"
CONFIG = run_config("thread-1")
DRAFT = "# Revenue report\n\n- The findings agree on the main points [S1, S2].\n"


def _settings(**kw):
    return Settings(llm_provider="fake", _env_file=None, **{"human_review": True, **kw})


def _nodes():
    """Placeholder agents, except a real writer, so we can see the prompt it gets."""
    writer_llm = FakeChatModel(responses=[DRAFT])
    return {
        SUPERVISOR: placeholders.supervisor,
        WEB: placeholders.web_research,
        RAG: placeholders.rag,
        DATA: placeholders.data_analysis,
        WRITER: WriterAgent(llm=writer_llm, settings=_settings()),
        CRITIC: placeholders.critic,  # always accepts, so every draft reaches the gate
    }, writer_llm


def _graph(checkpointer, **settings_kw):
    nodes, writer_llm = _nodes()
    graph = build_graph(nodes, settings=_settings(**settings_kw), checkpointer=checkpointer)
    return graph, writer_llm


def _resume(graph, value):
    """Resume a paused run; return the nodes that ran (the pause marker excluded)."""
    visited = []
    for step in graph.stream(Command(resume=value), CONFIG, stream_mode="updates"):
        visited.extend(node for node in step if node != "__interrupt__")
    return visited


# ---------------------------------------------------------------------------
# Resume values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("approve", (True, None)),
        ("  Approve ", (True, None)),
        ("y", (True, None)),
        ("Add more on risks", (False, "Add more on risks")),
        ("   ", (False, NO_COMMENT_FEEDBACK)),  # a rejection must give the writer something
        ({"approved": True, "feedback": "ignored"}, (True, None)),  # the API's structured form
        ({"approved": False, "feedback": "Shorter"}, (False, "Shorter")),
        ({"approved": False}, (False, NO_COMMENT_FEEDBACK)),
    ],
)
def test_parse_decision(value, expected):
    assert parse_decision(value) == expected


def test_parse_decision_rejects_unknown_types():
    with pytest.raises(TypeError):
        parse_decision(42)


# ---------------------------------------------------------------------------
# Interrupt and resume
# ---------------------------------------------------------------------------
def test_gate_pauses_and_surfaces_draft_and_critique():
    graph, _ = _graph(InMemorySaver())
    out = graph.invoke({"question": QUESTION}, CONFIG)

    (pause,) = out["__interrupt__"]
    payload = pause.value
    assert payload["critique"]["verdict"] == "accept"
    assert payload["draft"].startswith("# Revenue report")
    # The preview is the reader-facing version: numbered citations + Sources.
    assert "[1, 2]" in payload["preview"] and "## Sources" in payload["preview"]
    assert "final_report" not in out
    assert graph.get_state(CONFIG).next == (HUMAN_GATE,)
    assert pending_review(graph, CONFIG) == payload


def test_approve_goes_to_finalize():
    graph, _ = _graph(InMemorySaver())
    graph.invoke({"question": QUESTION}, CONFIG)

    assert _resume(graph, "approve") == [HUMAN_GATE, FINALIZE]
    state = graph.get_state(CONFIG)
    assert state.next == ()  # finished
    assert state.values["approved"] is True and state.values["human_feedback"] is None
    assert (
        state.values["final_report"].endswith("\n") and "## Sources" in state.values["final_report"]
    )
    assert pending_review(graph, CONFIG) is None


def test_rejection_goes_to_writer_with_feedback_first():
    graph, writer_llm = _graph(InMemorySaver())
    graph.invoke({"question": QUESTION}, CONFIG)

    # Rejected: writer revises, critic reviews, and the gate pauses again.
    assert _resume(graph, "Add more on risks") == [HUMAN_GATE, WRITER, CRITIC]
    values = graph.get_state(CONFIG).values
    assert values["approved"] is False and values["human_feedback"] == "Add more on risks"
    assert values["revision_count"] == 1
    # The writer got the human's feedback as point 1, marked as the top priority.
    prompt = writer_llm.history[-1][-1].content
    assert "1. [Human reviewer - highest priority] Add more on risks" in prompt
    assert pending_review(graph, CONFIG) is not None  # back with the human

    assert _resume(graph, "approve") == [HUMAN_GATE, FINALIZE]


def test_checkpointer_round_trip_survives_a_restart(tmp_path):
    db = tmp_path / "checkpoints.db"
    with sqlite_checkpointer(db) as saver:
        graph, _ = _graph(saver)
        graph.invoke({"question": QUESTION}, CONFIG)
        draft = pending_review(graph, CONFIG)["draft"]
    # The connection is closed: as far as the run knows, the process exited.

    # A fresh connection and a freshly built graph stand in for a new process.
    with sqlite_checkpointer(db) as saver:
        graph, writer_llm = _graph(saver)
        assert pending_review(graph, CONFIG)["draft"] == draft  # state came back from disk
        final = graph.invoke(Command(resume="approve"), CONFIG)

        assert writer_llm.history == []  # nothing before the gate ran again
        assert final["approved"] is True and "## Sources" in final["final_report"]
        assert pending_review(graph, run_config("another-thread")) is None  # runs are isolated


def test_disabled_gate_auto_approves_without_a_checkpointer():
    nodes, _ = _nodes()
    graph = build_graph(nodes, settings=_settings(human_review=False))  # no checkpointer
    out = graph.invoke({"question": QUESTION})

    assert "__interrupt__" not in out
    assert out["approved"] is True and "## Sources" in out["final_report"]


def test_enabled_gate_without_a_checkpointer_fails_at_build_time():
    nodes, _ = _nodes()
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(nodes, settings=_settings())


# ---------------------------------------------------------------------------
# The CLI, end to end: pause, exit, resume, reject, approve
# ---------------------------------------------------------------------------
def _answers(*replies):
    """A scripted human. Running out of replies acts like Ctrl+D (leave paused)."""
    queue = list(replies)

    def ask(_prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return ask


def test_cli_pause_then_resume_with_rejection_and_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    get_settings.cache_clear()

    # First call: run to the gate, show the draft, then the user leaves (EOF).
    assert review.main([QUESTION], nodes=_nodes()[0], ask=_answers()) == 0
    out = capsys.readouterr().out
    assert "DRAFT FOR REVIEW" in out and "Critic verdict: accept" in out
    thread_id = re.search(r"--resume (\w+)", out).group(1)

    # Second call, as a new process would: a blank line is ignored, then a
    # rejection, then approval of the revised draft.
    answers = _answers("", "Add more on risks", "approve")
    assert review.main(["--resume", thread_id], nodes=_nodes()[0], ask=answers) == 0
    out = capsys.readouterr().out
    assert "Sent back to the writer." in out and out.count("DRAFT FOR REVIEW") == 2
    assert "FINAL REPORT" in out and "## Sources" in out


def test_cli_unknown_thread(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    get_settings.cache_clear()
    assert review.main(["--resume", "nope"], nodes=_nodes()[0]) == 1
    assert "No run found" in capsys.readouterr().err
