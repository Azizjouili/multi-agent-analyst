"""Run the full pipeline from the terminal and approve or reject the report.

    uv run python scripts/review.py "Summarize the company's revenue trend and key risks"
    uv run python scripts/review.py --resume <thread_id>

The run pauses at the human gate and prints the report. Type `approve` to
finalize it, or type feedback to send it back to the writer; the writer treats
your feedback as the top priority. Press Ctrl+Z then Enter (Windows) or Ctrl+D
(macOS/Linux) to leave the run paused. Its state is in the SQLite checkpoint
file, so `--resume <thread_id>` picks it up later, even after a reboot.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Callable, Mapping

from langgraph.types import Command

from analyst.agents.human_gate import parse_decision
from analyst.config import get_settings
from analyst.graph import build_graph, pending_review, run_config, sqlite_checkpointer
from analyst.state import feedback_points
from analyst.tracing import flush

RULE = "=" * 78


def main(
    argv: list[str] | None = None,
    *,
    nodes: Mapping[str, Callable[..., dict]] | None = None,
    ask: Callable[[str], str] = input,
) -> int:
    """CLI entry point. `nodes` and `ask` let tests script the agents and the human."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question", nargs="*", help="the research question (new run)")
    parser.add_argument("--resume", metavar="THREAD_ID", help="continue a paused run")
    args = parser.parse_args(argv)
    if bool(args.question) == bool(args.resume):
        parser.error("give either a question or --resume THREAD_ID")

    # This script exists to review, so the gate is always on here, whatever .env says.
    settings = get_settings().model_copy(update={"human_review": True})
    thread_id = args.resume or uuid.uuid4().hex[:12]

    with sqlite_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(nodes, settings=settings, checkpointer=saver)

        if args.resume:
            snapshot = graph.get_state(run_config(thread_id))
            if not snapshot.values:
                print(f"No run found with thread id {thread_id!r}.", file=sys.stderr)
                return 1
            # The question comes from the checkpoint, so the resumed part is
            # traced under the same name and session as the first part.
            config = run_config(thread_id, snapshot.values["question"], settings)
            if snapshot.next and pending_review(graph, config) is None:
                # The process died mid-run (crash, Ctrl+C). invoke(None) means
                # "continue from the last checkpoint": finished nodes don't re-run.
                print(f"Run {thread_id} was interrupted mid-way; continuing it.")
                _stream(graph, None, config)
        else:
            question = " ".join(args.question)
            config = run_config(thread_id, question, settings)
            print(f"Run {thread_id} started. Working... (this takes a few LLM calls)")
            _stream(graph, {"question": question}, config)

        while (payload := pending_review(graph, config)) is not None:
            _show(payload)
            answer = _ask_decision(ask)
            if answer is None:
                print("\nLeft paused. Resume with:")
                print(f"  uv run python scripts/review.py --resume {thread_id}")
                return 0
            approved, _ = parse_decision(answer)
            print("Approved, finalizing." if approved else "Sent back to the writer.")
            _stream(graph, Command(resume=answer), config)

        print(f"\n{RULE}\nFINAL REPORT (run {thread_id})\n{RULE}\n")
        print(graph.get_state(config).values.get("final_report", "(no report)"))
    return 0


def _stream(graph, graph_input, config) -> None:
    # stream_mode="updates" yields {node: update} as each node finishes, so the
    # user sees progress instead of a silent wait.
    for step in graph.stream(graph_input, config, stream_mode="updates"):
        for node in step:
            if node != "__interrupt__":  # the pause itself, handled by the caller
                print(f"  [{node}] done")


def _show(payload: dict) -> None:
    print(f"\n{RULE}\nDRAFT FOR REVIEW: {payload['question']}\n{RULE}\n")
    print(payload["preview"])
    critique = payload.get("critique")
    print(RULE)
    print(
        f"Revisions so far: {payload['revision_count']} | "
        f"research rounds: {payload['research_rounds']}"
    )
    if critique:
        scores = ", ".join(f"{k}={v}" for k, v in critique.get("scores", {}).items()) or "-"
        forced = " (forced: loop budget used up)" if critique.get("budget_exhausted") else ""
        print(f"Critic verdict: {critique['verdict']}{forced} | scores: {scores}")
        # Open points the critic still had: the person decides whether they matter.
        for point in feedback_points(critique):
            print(f"  - {point}")
    print(RULE)


def _ask_decision(ask: Callable[[str], str]) -> str | None:
    """Read 'approve' or rejection feedback. None means leave the run paused."""
    while True:
        try:
            answer = ask("Type 'approve', or write feedback to reject: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer:
            return answer
        # A blank line is probably a stray Enter, not a decision. Rejecting
        # would trigger a paid rewrite with no guidance, so ask again.


if __name__ == "__main__":
    # Windows consoles default to a legacy code page that can't print every
    # character a report may contain (en dashes, accents). UTF-8 avoids a crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="  %(name)s: %(message)s")
    try:
        sys.exit(main())
    finally:
        # Even when the user leaves the run paused, send its spans before exiting.
        flush()
