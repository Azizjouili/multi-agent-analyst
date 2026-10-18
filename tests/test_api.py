"""Offline tests for the HTTP API, with scripted agents and a temporary checkpoint DB."""

import threading
import time
from contextlib import contextmanager

from fastapi.testclient import TestClient

from analyst.agents import placeholders
from analyst.api import create_app
from analyst.config import Settings
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER

QUESTION = "Summarize the company's revenue trend and key risks"

# The real graph wiring, real human gate and real finalize, with placeholder
# agents: no LLM, no network, and each run takes milliseconds.
FAKE_NODES = {
    SUPERVISOR: placeholders.supervisor,
    WEB: placeholders.web_research,
    RAG: placeholders.rag,
    DATA: placeholders.data_analysis,
    WRITER: placeholders.writer,
    CRITIC: placeholders.critic,  # always accepts, so every draft reaches the gate
}


@contextmanager
def api(tmp_path, nodes=None):
    """A running app. The same tmp_path means the same checkpoint DB: a "restart"."""
    settings = Settings(
        llm_provider="fake",
        _env_file=None,
        human_review=True,
        checkpoint_db=tmp_path / "checkpoints.db",
    )
    app = create_app(settings, {**FAKE_NODES, **(nodes or {})})
    with TestClient(app) as client:  # `with` runs the lifespan: opens the checkpointer
        yield client


def poll(client, thread_id, timeout=10.0):
    """GET the run until it stops being 'running'; return that status payload."""
    deadline = time.monotonic() + timeout
    while True:
        info = client.get(f"/runs/{thread_id}").json()
        if info["status"] != "running":
            return info
        assert time.monotonic() < deadline, f"run {thread_id} still running after {timeout}s"
        time.sleep(0.02)


def start(client):
    response = client.post("/runs", json={"question": QUESTION})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running" and body["thread_id"]
    return body["thread_id"]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_health(tmp_path):
    with api(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok", "active_runs": 0}


def test_web_ui_served_at_root(tmp_path):
    with api(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        html = response.text
        # The elements the page's script wires up, and the endpoints it calls.
        for marker in ('id="question"', 'id="ask-btn"', 'id="approve-btn"', 'id="feedback-btn"'):
            assert marker in html
        assert "/runs" in html and "/review" in html
        # The UI is a page, not an API operation: it stays out of the OpenAPI schema.
        assert "/" not in client.get("/openapi.json").json()["paths"]


def test_start_poll_approve_get_report(tmp_path):
    with api(tmp_path) as client:
        thread_id = start(client)

        info = poll(client, thread_id)
        assert info["status"] == "awaiting_review"
        assert info["question"] == QUESTION
        review = info["review"]
        assert review["critique"]["verdict"] == "accept"
        assert review["draft"].startswith("# Report") and "## Sources" in review["preview"]
        assert info["report"] is None

        response = client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert response.status_code == 202 and response.json()["status"] == "running"

        info = poll(client, thread_id)
        assert info["status"] == "done"
        assert info["report"].startswith("# Report") and "## Sources" in info["report"]
        assert info["review"] is None


def test_reject_sends_the_draft_back_then_approve(tmp_path):
    with api(tmp_path) as client:
        thread_id = start(client)
        assert poll(client, thread_id)["status"] == "awaiting_review"

        client.post(
            f"/runs/{thread_id}/review",
            json={"approved": False, "feedback": "Add more on risks"},
        )
        info = poll(client, thread_id)
        # Writer revised, critic reviewed, and the run is back with the human.
        assert info["status"] == "awaiting_review"
        assert info["review"]["revision_count"] == 1

        client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert poll(client, thread_id)["status"] == "done"


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------
def test_api_stays_responsive_while_a_run_works(tmp_path):
    release = threading.Event()

    def slow_supervisor(state):
        assert release.wait(5), "test never released the run"
        return placeholders.supervisor(state)

    with api(tmp_path, {SUPERVISOR: slow_supervisor}) as client:
        thread_id = start(client)

        # The run is blocked inside a node, yet every request answers at once.
        assert client.get(f"/runs/{thread_id}").json()["status"] == "running"
        assert client.get("/health").json()["active_runs"] == 1
        response = client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert response.status_code == 409  # can't review a run that's still working

        release.set()
        assert poll(client, thread_id)["status"] == "awaiting_review"
        assert client.get("/health").json()["active_runs"] == 0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_unknown_run_is_404(tmp_path):
    with api(tmp_path) as client:
        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs/nope/review", json={"approved": True}).status_code == 404


def test_invalid_bodies_are_422(tmp_path):
    with api(tmp_path) as client:
        assert client.post("/runs", json={"question": "?"}).status_code == 422  # too short
        assert client.post("/runs", json={}).status_code == 422


def test_review_of_a_finished_run_is_409(tmp_path):
    with api(tmp_path) as client:
        thread_id = start(client)
        poll(client, thread_id)
        client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert poll(client, thread_id)["status"] == "done"
        response = client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert response.status_code == 409


def test_a_failing_run_reports_failed(tmp_path):
    def broken_writer(state):
        raise RuntimeError("writer exploded")

    with api(tmp_path, {WRITER: broken_writer}) as client:
        info = poll(client, start(client))
        assert info["status"] == "failed"
        assert info["error"] == "RuntimeError: writer exploded"


# ---------------------------------------------------------------------------
# Persistence across API restarts
# ---------------------------------------------------------------------------
def test_paused_run_survives_an_api_restart(tmp_path):
    with api(tmp_path) as client:
        thread_id = start(client)
        draft = poll(client, thread_id)["review"]["draft"]
    # The app shut down: its memory (the run registry) is gone.

    with api(tmp_path) as client:
        info = client.get(f"/runs/{thread_id}").json()  # status read from the checkpoint
        assert info["status"] == "awaiting_review" and info["review"]["draft"] == draft
        client.post(f"/runs/{thread_id}/review", json={"approved": True})
        assert poll(client, thread_id)["status"] == "done"


def test_run_cut_off_mid_way_continues_after_a_restart(tmp_path):
    def broken_writer(state):
        raise RuntimeError("process died here")

    with api(tmp_path, {WRITER: broken_writer}) as client:
        thread_id = start(client)
        assert poll(client, thread_id)["status"] == "failed"

    # The new process has a working writer. Asking for the status continues
    # the run from its last checkpoint: the research already done isn't redone.
    calls = []

    def counting_supervisor(state):
        calls.append(1)
        return placeholders.supervisor(state)

    with api(tmp_path, {SUPERVISOR: counting_supervisor}) as client:
        assert client.get(f"/runs/{thread_id}").json()["status"] == "running"
        info = poll(client, thread_id)
        assert info["status"] == "awaiting_review"
        assert calls == []  # supervisor and specialists didn't run again
