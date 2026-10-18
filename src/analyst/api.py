"""HTTP API: start research runs, poll them, and submit the human review.

    POST /runs                      {"question"}             -> 202 {thread_id, status}
    GET  /runs/{thread_id}                                   -> status + review or report
    POST /runs/{thread_id}/review   {"approved", "feedback"} -> 202, resumes the run
    GET  /health

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

Start it with `uv run python scripts/serve.py`. Interactive docs: /docs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.graph import build_graph, pending_review, run_config, sqlite_checkpointer

log = logging.getLogger("analyst.api")

Status = Literal["running", "awaiting_review", "done", "failed"]


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


class RunInfo(BaseModel):
    thread_id: str
    status: Status
    question: str | None = None
    review: dict[str, Any] | None = Field(
        None, description="When awaiting_review: draft, preview and the critic's verdict"
    )
    report: str | None = Field(None, description="When done: the final report")
    error: str | None = Field(None, description="When failed: what went wrong")


class Conflict(Exception):
    """The run exists but isn't in a state that allows this action."""


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------
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
        return RunInfo(thread_id=thread_id, status="running", question=snapshot.values["question"])

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

        def work() -> None:
            try:
                with self._slots:
                    # The question goes into the config for tracing (step 12).
                    self.graph.invoke(graph_input, run_config(thread_id, question, self.settings))
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


def create_app(
    settings: Settings | None = None,
    nodes: Mapping[str, Callable[..., dict]] | None = None,
) -> FastAPI:
    """Build the app. A factory (not a module-level `app`) so that importing
    this module opens no database, and tests can pass their own settings and
    scripted nodes."""
    settings = settings or get_settings()

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

    return app
