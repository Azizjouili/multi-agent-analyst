"""HTTP API: start research runs, poll them, and submit the human review.

    POST /runs                      {"question"}             -> 202 {thread_id, status}
    GET  /runs/{thread_id}                                   -> status + review or report
    POST /runs/{thread_id}/review   {"approved", "feedback"} -> 202, resumes the run
    GET  /health
    GET  /  (and any other non-API path)                     -> the web UI (React SPA)

    status:  running ---> awaiting_review --approve--> running ---> done
                               \\--reject (feedback)--> running ---> awaiting_review ...
             (failed: the run raised an error in this process; see `error`)

Why runs go to a background thread: a run takes minutes (dozens of LLM calls).
Holding the HTTP request open that long would time out at proxies and block
the client. So POST returns a thread_id at once, and the client polls.

Why the API doesn't keep run state in memory: the SQLite checkpointer from
step 11 already holds it. GET reads the status from the checkpoint, so after an
API restart every run is still there: a paused run is still awaiting review,
and a run that was cut off mid-way continues from its last checkpoint the next
time someone asks about it. The in-memory registry only tracks which runs are
executing *in this process* right now.

`stages` (live node progress for the UI) is the one exception: it is recorded
in memory while this process runs the graph, so after a restart it starts
empty. It's a progress display, not state the run depends on.

Start it with `uv run python scripts/serve.py`. Web UI: /. Interactive docs: /docs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel, Field

from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.graph import build_graph, pending_review, run_config, sqlite_checkpointer

log = logging.getLogger("analyst.api")

Status = Literal["running", "awaiting_review", "done", "failed"]
# "waiting": the node paused the run (the human gate's interrupt); it becomes
# "running" again, in the same entry, when the run resumes.
StageStatus = Literal["running", "done", "waiting", "failed"]

# Where the built React app is served from. In the image, the Dockerfile's
# Node stage copies frontend/dist here, inside the package, next to this module.
STATIC_DIR = Path(__file__).parent / "static"
# In a dev checkout, `npm run build` writes to frontend/dist instead; serving it
# from there means no copy step before trying the production build locally.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
# First path segments that belong to the API (or FastAPI's docs). The SPA
# fallback never answers these with index.html: a mistyped API call should get
# a JSON 404, not an HTML page that a client would fail to parse.
API_PREFIXES = frozenset({"runs", "health", "docs", "redoc", "openapi.json", "assets"})


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class StartRun(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="The research question")


class ReviewDecision(BaseModel):
    approved: bool
    feedback: str | None = Field(
        None, max_length=2000, description="Required in spirit when rejecting: what to change"
    )


class Stage(BaseModel):
    node: str = Field(description="Graph node name, e.g. supervisor, rag, writer")
    status: StageStatus
    step: int | None = Field(
        None, description="LangGraph superstep; nodes sharing a step ran in parallel"
    )
    started_at: datetime
    finished_at: datetime | None = None


class RunInfo(BaseModel):
    thread_id: str
    status: Status
    question: str | None = None
    review: dict[str, Any] | None = Field(
        None, description="When awaiting_review: draft, preview and the critic's verdict"
    )
    report: str | None = Field(None, description="When done: the final report")
    error: str | None = Field(None, description="When failed: what went wrong")
    stages: list[Stage] = Field(
        default_factory=list,
        description="Graph nodes in execution order, as run by this process (live progress)",
    )


class Conflict(Exception):
    """The run exists but isn't in a state that allows this action."""


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------
class StageRecorder(BaseCallbackHandler):
    """Records each graph node as it starts and finishes, for the live timeline.

    How progress is captured without touching the graph: LangGraph runs every
    node as a LangChain runnable, so callbacks passed in the run config (the
    same channel the Langfuse handler uses) fire on_chain_start/end for each
    node, in real time, including parallel specialists. A node's run is named
    after the node and carries `langgraph_node` metadata; everything nested
    inside it (LLM calls, routing functions like `dispatch`) inherits that
    metadata but has a different name, so `name == langgraph_node` picks out
    exactly the node runs.

    Why callbacks rather than `graph.stream(stream_mode="updates")`: updates
    arrive only when a node *finishes*, so a running node couldn't be shown.
    And a callback can't break a run: LangChain logs handler errors instead
    of raising them (raise_error is False).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # parallel specialists call back from several threads
        self._stages: list[dict[str, Any]] = []
        self._open: dict[UUID, dict[str, Any]] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        node = metadata.get("langgraph_node")
        if node is None or name != node:
            return
        with self._lock:
            last = next((s for s in reversed(self._stages) if s["node"] == node), None)
            if last is not None and last["status"] == "waiting":
                # The human gate re-runs from the top on resume (see human_gate.py).
                # Reuse its entry so the timeline shows one gate, not two.
                stage = last
                stage.update(status="running", finished_at=None)
            else:
                stage = {
                    "node": node,
                    "status": "running",
                    "step": metadata.get("langgraph_step"),
                    "started_at": datetime.now(UTC),
                    "finished_at": None,
                }
                self._stages.append(stage)
            self._open[run_id] = stage

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, "done")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # interrupt() works by raising GraphInterrupt: a pause, not a failure.
        self._close(run_id, "waiting" if isinstance(error, GraphInterrupt) else "failed")

    def _close(self, run_id: UUID, status: StageStatus) -> None:
        with self._lock:
            stage = self._open.pop(run_id, None)
            if stage is not None:
                stage["status"] = status
                stage["finished_at"] = None if status == "waiting" else datetime.now(UTC)

    def stages(self) -> list[Stage]:
        with self._lock:
            return [Stage(**s) for s in self._stages]


@dataclass
class _Task:
    """A run executing in this process."""

    thread: threading.Thread | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class RunService:
    """Starts runs in background threads and reports their status from the checkpoint."""

    def __init__(self, graph, settings: Settings):
        self.graph = graph
        self.settings = settings
        self._tasks: dict[str, _Task] = {}
        # Per run, not per task: a run is several tasks (start, then one per
        # review decision), and the timeline should span all of them.
        self._recorders: dict[str, StageRecorder] = {}
        # One lock around "check state, then launch". Without it, two review
        # requests arriving together could both see awaiting_review and both
        # resume the run.
        self._lock = threading.Lock()
        # Runs beyond the limit wait here (and show as "running") instead of
        # all hammering the LLM quota at once.
        self._slots = threading.BoundedSemaphore(settings.api_max_concurrent_runs)

    # -- public API ------------------------------------------------------------
    def start(self, question: str) -> RunInfo:
        thread_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._launch(thread_id, {"question": question}, question)
        return RunInfo(thread_id=thread_id, status="running", question=question)

    def status(self, thread_id: str) -> RunInfo:
        info = self._status(thread_id)
        # Additive: every other field is exactly what it was without stages.
        if recorder := self._recorders.get(thread_id):
            info.stages = recorder.stages()
        return info

    def _status(self, thread_id: str) -> RunInfo:
        with self._lock:
            task = self._tasks.get(thread_id)
            if task and not task.done.is_set():
                return RunInfo(thread_id=thread_id, status="running")
            if task and task.error:
                return RunInfo(thread_id=thread_id, status="failed", error=task.error)

            snapshot = self.graph.get_state(run_config(thread_id))
            if not snapshot.values:
                raise KeyError(thread_id)
            question = snapshot.values["question"]
            if (review := pending_review(self.graph, run_config(thread_id))) is not None:
                return RunInfo(
                    thread_id=thread_id, status="awaiting_review", question=question, review=review
                )
            if not snapshot.next:
                report = snapshot.values.get("final_report")
                return RunInfo(thread_id=thread_id, status="done", question=question, report=report)

            # Unfinished, not paused, and nobody here is running it: the process
            # that was running it died (crash, restart, deploy). Continue from
            # the last checkpoint; finished nodes don't run again.
            log.info("api: resuming run %s, cut off mid-way by a restart", thread_id)
            self._launch(thread_id, None, question)
            return RunInfo(thread_id=thread_id, status="running", question=question)

    def review(self, thread_id: str, decision: ReviewDecision) -> RunInfo:
        with self._lock:
            task = self._tasks.get(thread_id)
            if task and not task.done.is_set():
                raise Conflict("The run is still working; wait for status awaiting_review.")
            snapshot = self.graph.get_state(run_config(thread_id))
            if not snapshot.values:
                raise KeyError(thread_id)
            if pending_review(self.graph, run_config(thread_id)) is None:
                raise Conflict("The run is not awaiting review.")
            # The structured resume form; the human gate's parse_decision reads it.
            resume = Command(resume={"approved": decision.approved, "feedback": decision.feedback})
            self._launch(thread_id, resume, snapshot.values["question"])
        return RunInfo(
            thread_id=thread_id,
            status="running",
            question=snapshot.values["question"],
            stages=self._recorders[thread_id].stages(),
        )

    def active_runs(self) -> int:
        with self._lock:
            return sum(not t.done.is_set() for t in self._tasks.values())

    def wait(self, thread_id: str, timeout: float | None = None) -> bool:
        """Block until this process's current task for the run finishes (for tests/tools)."""
        task = self._tasks.get(thread_id)
        return task.done.wait(timeout) if task else True

    # -- internals -------------------------------------------------------------
    def _launch(self, thread_id: str, graph_input: Any, question: str) -> None:
        """Run the graph in a background thread until it pauses or ends. Caller holds the lock."""
        task = _Task()
        recorder = self._recorders.setdefault(thread_id, StageRecorder())
        # The question goes into the config for tracing (step 12). The stage
        # recorder rides along with the tracing callbacks, not instead of them.
        config = run_config(thread_id, question, self.settings)
        config["callbacks"] = [*config.get("callbacks", []), recorder]

        def work() -> None:
            try:
                with self._slots:
                    self.graph.invoke(graph_input, config)
            except Exception as exc:  # noqa: BLE001 - report it through the status endpoint
                log.exception("api: run %s failed", thread_id)
                task.error = short_error(exc)
            finally:
                task.done.set()

        # Daemon thread: stopping the server shouldn't wait minutes for runs to
        # finish. Killing one mid-way is safe; its checkpoints let it continue.
        task.thread = threading.Thread(target=work, name=f"run-{thread_id}", daemon=True)
        self._tasks[thread_id] = task
        task.thread.start()


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
def _runs(request: Request) -> RunService:
    return request.app.state.runs


# Dependency injection: each endpoint declares that it needs the service, and
# FastAPI supplies it, so tests could swap it without touching routes. Defined
# at module level because `from __future__ import annotations` turns
# annotations into strings, which FastAPI resolves from the module's globals.
Runs = Annotated[RunService, Depends(_runs)]


class _HashedAssets(StaticFiles):
    """StaticFiles for Vite's /assets: file names carry a content hash
    (index-3f9a1c.js), so a URL's content never changes and browsers may cache
    it forever. index.html, which points at the current hashes, is never cached."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def default_static_dir() -> Path:
    """The packaged build (Docker) if present, else a local `npm run build`."""
    if (STATIC_DIR / "index.html").is_file() or not (FRONTEND_DIST / "index.html").is_file():
        return STATIC_DIR
    return FRONTEND_DIST


_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Multi-Agent Analyst</title>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>The web UI isn't built</h1>
<p>The API is running (see <a href="/docs">/docs</a>). For the UI, either run
<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code> and restart,
or develop with <code>npm run dev</code> and open the Vite URL.</p></body>"""


def create_app(
    settings: Settings | None = None,
    nodes: Mapping[str, Callable[..., dict]] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the app. A factory (not a module-level `app`) so that importing
    this module opens no database, and tests can pass their own settings,
    scripted nodes and frontend build."""
    settings = settings or get_settings()
    static_dir = (static_dir or default_static_dir()).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One checkpointer connection and one compiled graph for the app's
        # lifetime, shared by all runs; each run is isolated by its thread_id.
        with sqlite_checkpointer(settings.checkpoint_db) as saver:
            graph = build_graph(nodes, settings=settings, checkpointer=saver)
            app.state.runs = RunService(graph, settings)
            yield

    app = FastAPI(
        title="Multi-Agent Analyst",
        description="Research runs with a supervisor, specialists, a critic and human approval.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Plain `def` endpoints, not `async def`: they call blocking code (SQLite
    # reads). FastAPI runs sync endpoints in a thread pool, so a slow read
    # doesn't stall the event loop for other requests.
    @app.post("/runs", response_model=RunInfo, status_code=202)
    def start_run(body: StartRun, service: Runs) -> RunInfo:
        return service.start(body.question)

    @app.get("/runs/{thread_id}", response_model=RunInfo)
    def get_run(thread_id: str, service: Runs) -> RunInfo:
        try:
            return service.status(thread_id)
        except KeyError:
            raise HTTPException(404, f"No run with thread_id {thread_id!r}") from None

    @app.post("/runs/{thread_id}/review", response_model=RunInfo, status_code=202)
    def review_run(thread_id: str, body: ReviewDecision, service: Runs) -> RunInfo:
        try:
            return service.review(thread_id, body)
        except KeyError:
            raise HTTPException(404, f"No run with thread_id {thread_id!r}") from None
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/health")
    def health(service: Runs) -> dict[str, Any]:
        return {"status": "ok", "active_runs": service.active_runs()}

    # -- the web UI (React SPA) -------------------------------------------------
    # Served by this same app rather than a separate frontend server: same
    # origin, so the UI calls /runs with no CORS setup, and one port works the
    # same locally and in Docker. Everything below is registered *after* the
    # API routes, and Starlette matches routes in order, so none of it can
    # shadow /runs, /health, /docs or /openapi.json. Hidden from the schema:
    # it isn't API.
    if (static_dir / "assets").is_dir():
        app.mount("/assets", _HashedAssets(directory=static_dir / "assets"), name="assets")

    def index() -> Response:
        page = static_dir / "index.html"
        if not page.is_file():
            # The API still works without a build; say how to get the UI.
            return HTMLResponse(_NOT_BUILT, status_code=503)
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    def web_ui() -> Response:
        return index()

    # SPA fallback: a client-side route (a bookmarked /some/view, a reload)
    # must get index.html so React can render it; the browser would otherwise
    # show a 404. Real files at the build root (favicon.svg) are served as
    # files. Paths that look like a file, or start with an API prefix, get a
    # real 404 instead: answering a missing script or a mistyped API call with
    # HTML only hides the mistake.
    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> Response:
        if path.split("/", 1)[0] in API_PREFIXES:
            raise HTTPException(404, "Not Found")
        candidate = (static_dir / path).resolve()
        # is_relative_to: "../../etc/passwd" must not escape the build folder.
        if candidate.is_file() and candidate.is_relative_to(static_dir):
            return FileResponse(candidate)
        if "." in path.rsplit("/", 1)[-1]:
            raise HTTPException(404, "Not Found")
        return index()

    return app
