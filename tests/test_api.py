"""Offline tests for the HTTP API, with scripted agents and a temporary checkpoint DB."""

import re
import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from analyst.agents import placeholders
from analyst.api import create_app, default_static_dir
from analyst.config import Settings
from analyst.graph import CRITIC, DATA, FINALIZE, HUMAN_GATE, RAG, SUPERVISOR, WEB, WRITER

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
def api(tmp_path, nodes=None, static_dir=None):
    """A running app. The same tmp_path means the same checkpoint DB: a "restart"."""
    settings = Settings(
        llm_provider="fake",
        _env_file=None,
        human_review=True,
        checkpoint_db=tmp_path / "checkpoints.db",
    )
    app = create_app(settings, {**FAKE_NODES, **(nodes or {})}, static_dir=static_dir)
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


# ---------------------------------------------------------------------------
# Web UI (the built React SPA)
# ---------------------------------------------------------------------------
BUNDLE = "/assets/index-abc123.js"


def fake_build(root):
    """The shape of a Vite build: index.html pointing at a hashed bundle."""
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        f'<!doctype html><div id="root"></div><script type="module" src="{BUNDLE}"></script>'
    )
    (root / "assets" / "index-abc123.js").write_text("console.log('app')")
    (root / "favicon.svg").write_text("<svg/>")
    return root


def test_web_ui_served_at_root(tmp_path):
    with api(tmp_path, static_dir=fake_build(tmp_path / "dist")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert '<div id="root">' in response.text and BUNDLE in response.text
        assert response.headers["cache-control"] == "no-cache"  # always picks up a new build

        bundle = client.get(BUNDLE)
        assert bundle.status_code == 200 and "immutable" in bundle.headers["cache-control"]
        assert client.get("/favicon.svg").status_code == 200

        # The UI is a page, not an API operation: it stays out of the OpenAPI schema.
        assert "/" not in client.get("/openapi.json").json()["paths"]


def test_spa_fallback_does_not_shadow_the_api(tmp_path):
    with api(tmp_path, static_dir=fake_build(tmp_path / "dist")) as client:
        # Client-side routes get the app...
        assert '<div id="root">' in client.get("/runs-history/view").text
        # ...API routes keep answering as before...
        assert client.get("/health").json() == {"status": "ok", "active_runs": 0}
        missing = client.get("/runs/nope")
        assert missing.status_code == 404 and "nope" in missing.json()["detail"]
        assert client.get("/docs").status_code == 200
        assert client.post("/runs", json={"question": "x"}).status_code == 422
        # ...and unknown API paths, missing files and traversal are real 404s, not HTML.
        for path in ("/runs/a/b/c", "/assets/gone.js", "/missing.png", "/..%2F..%2Fpyproject.toml"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert "root" not in response.text, path


def test_web_ui_not_built_explains_how(tmp_path):
    with api(tmp_path, static_dir=tmp_path / "empty") as client:
        response = client.get("/")
        assert response.status_code == 503 and "npm run build" in response.text
        assert client.get("/health").status_code == 200  # the API itself is unaffected


def test_real_frontend_build_is_served(tmp_path):
    """Checks the actual `npm run build` output when there is one (Docker, or a
    local build). CI without Node skips it; the fake-build tests cover routing."""
    static_dir = default_static_dir()
    if not (static_dir / "index.html").is_file():
        pytest.skip("frontend not built (cd frontend && npm run build)")
    with api(tmp_path, static_dir=static_dir) as client:
        html = client.get("/").text
        assert '<div id="root">' in html
        bundle = re.search(r'src="(/assets/[^"]+\.js)"', html)
        assert bundle, "index.html doesn't reference a built JS bundle"
        assert client.get(bundle.group(1)).status_code == 200


# ---------------------------------------------------------------------------
# Live stage progress
# ---------------------------------------------------------------------------
def test_stages_follow_the_graph_live(tmp_path):
    with api(tmp_path) as client:
        thread_id = start(client)
        info = poll(client, thread_id)
        assert info["status"] == "awaiting_review"

        stages = info["stages"]
        nodes = [s["node"] for s in stages]
        assert nodes[0] == SUPERVISOR
        # The specialists ran in parallel: any order, but one shared superstep.
        assert set(nodes[1:4]) == {WEB, RAG, DATA}
        assert len({s["step"] for s in stages[1:4]}) == 1
        assert nodes[4:] == [WRITER, CRITIC, HUMAN_GATE]
        assert all(s["status"] == "done" and s["finished_at"] for s in stages[:-1])
        # Paused at the gate: waiting, not failed and not done.
        assert stages[-1]["status"] == "waiting" and stages[-1]["finished_at"] is None
        assert all(s["started_at"] for s in stages)

        assert client.post(f"/runs/{thread_id}/review", json={"approved": True}).status_code == 202
        done = poll(client, thread_id)
        assert done["status"] == "done"
        nodes = [s["node"] for s in done["stages"]]
        # The gate re-ran on resume, but it is still one entry in the timeline.
        assert nodes.count(HUMAN_GATE) == 1 and nodes[-1] == FINALIZE
        assert all(s["status"] == "done" for s in done["stages"])
        # Purely additive: the existing fields are unchanged.
        assert done["report"] and done["review"] is None and done["error"] is None


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
