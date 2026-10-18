"""Eval harness: run each question through the pipeline and score the report.

    uv run python evals/run_evals.py                      # all questions
    uv run python evals/run_evals.py --only growth_2024   # one question
    uv run python evals/run_evals.py --min-answer-hit 0.75 # fail (exit 1) below 75%

Needs the sample data and index (see the README quick start) and, with
LLM_PROVIDER=gemini, a GOOGLE_API_KEY: the *pipeline* still calls the LLM.

Scoring is deterministic: regexes and set checks, no LLM-as-judge. Why:
  - repeatable: the same report always gets the same score, so a score change
    means the system changed, not the judge's mood;
  - free and fast: scoring burns no quota, so it can run in CI on every change;
  - debuggable: a failed check names the exact missing pattern or bad citation.
The trade-off: it can't judge prose quality. That's the critic's job inside
the loop, and the human's at the gate.

Metrics per question:
  answer_hit            every expected pattern appears in the final report
  citation_coverage     share of factual sentences with at least one citation
  no_invented_citations every cited source id exists in state["sources"], and
                        every [n] in the final report has a Sources entry

Human review is off (nobody is there to approve), so each run goes straight
from the critic to finalize. Results go to evals/results/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from analyst.agents.citations import SOURCE_ID
from analyst.agents.common import short_error
from analyst.agents.critic import check_citations
from analyst.config import Settings, get_settings
from analyst.graph import build_graph, run_config

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVALS_DIR / "questions.yaml"
RESULTS_DIR = EVALS_DIR / "results"

_SOURCES_HEADING = re.compile(r"\n## Sources\n")
_NUMBERED_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SOURCE_ENTRY = re.compile(r"^(\d+)\. ", re.MULTILINE)


@dataclass
class EvalCase:
    id: str
    question: str
    expect: list[str]
    source: str = ""


def load_questions(path: Path) -> list[EvalCase]:
    cases = [EvalCase(**item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question ids")
    for case in cases:
        for pattern in case.expect:
            re.compile(pattern)  # a typo in a regex should fail here, not mid-run
    return cases


# ---------------------------------------------------------------------------
# Scoring (pure functions of the final state)
# ---------------------------------------------------------------------------
def split_report(final_report: str) -> tuple[str, str]:
    """(body, sources section). Answers are matched in the body only, so a
    figure that appears just in a Sources entry doesn't count as answered."""
    parts = _SOURCES_HEADING.split(final_report, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def score(case: EvalCase, state: Mapping[str, Any]) -> dict[str, Any]:
    final_report = state.get("final_report", "")
    body, sources_section = split_report(final_report)

    # answer_hit: on the reader-facing report, which is what a user gets.
    checks = [{"pattern": p, "found": bool(re.search(p, body, re.IGNORECASE))} for p in case.expect]

    # Citations are checked on the draft, which still holds stable source ids
    # (finalize renumbers them and silently drops unknown ones, which would
    # hide an invented citation). The same checker the critic uses, so evals
    # and the loop agree on what counts as a factual sentence.
    sources = state.get("sources", {})
    citations = check_citations(state.get("draft", ""), sources)
    claims = citations.claim_sentences
    # No factual sentences at all means nothing was answered: score 0, not 1.
    coverage = (claims - len(citations.uncited)) / claims if claims else 0.0

    # Second invented-citation check, on the final report: every [n] must
    # have a matching entry in the Sources list.
    listed = {int(n) for n in _SOURCE_ENTRY.findall(sources_section)}
    cited = {int(n) for group in _NUMBERED_CITATION.findall(body) for n in group.split(",")}
    dangling = sorted(cited - listed)

    return {
        "answer_hit": all(c["found"] for c in checks),
        "checks": checks,
        "citation_coverage": round(coverage, 3),
        "claim_sentences": claims,
        "uncited_sentences": citations.uncited,
        "no_invented_citations": not citations.unknown_ids and not dangling,
        "invented_source_ids": citations.unknown_ids,
        "dangling_citation_numbers": dangling,
        "sources_cited": len(set(SOURCE_ID.findall(state.get("draft", "")))),
        "sources_collected": len(sources),
        "revisions": state.get("revision_count", 0),
        "research_rounds": state.get("research_rounds", 0),
        "critic_verdicts": [c["verdict"] for c in state.get("critiques", [])],
    }


def failed_result() -> dict[str, Any]:
    """Scores for a run that crashed: every metric at its worst value."""
    return {
        "answer_hit": False,
        "checks": [],
        "citation_coverage": 0.0,
        "claim_sentences": 0,
        "uncited_sentences": [],
        "no_invented_citations": False,
        "invented_source_ids": [],
        "dangling_citation_numbers": [],
        "sources_cited": 0,
        "sources_collected": 0,
        "revisions": 0,
        "research_rounds": 0,
        "critic_verdicts": [],
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_case(graph, case: EvalCase, settings: Settings, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    try:
        # Its own thread id per question; with Langfuse configured, each eval
        # run is a trace you can open from the results file.
        thread_id = f"eval-{run_id}-{case.id}"
        state = graph.invoke(
            {"question": case.question}, run_config(thread_id, case.question, settings)
        )
        result = score(case, state)
        report = state.get("final_report", "")
    except Exception as exc:  # noqa: BLE001 - one broken question shouldn't stop the suite
        error, result, report, thread_id = short_error(exc), failed_result(), "", None
    return {
        "id": case.id,
        "question": case.question,
        "source": case.source,
        "thread_id": thread_id,
        **result,
        "duration_s": round(time.perf_counter() - started, 2),
        "error": error,
        "report": report,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)

    def rate(key: str) -> float:
        return round(sum(bool(r[key]) for r in results) / n, 3) if n else 0.0

    return {
        "questions": n,
        "answer_hit_rate": rate("answer_hit"),
        "mean_citation_coverage": (
            round(sum(r["citation_coverage"] for r in results) / n, 3) if n else 0.0
        ),
        "no_invented_citations_rate": rate("no_invented_citations"),
        "errors": sum(r["error"] is not None for r in results),
        "total_duration_s": round(sum(r["duration_s"] for r in results), 2),
    }


def print_table(output: dict[str, Any]) -> None:
    rows = [("question", "hit", "coverage", "citations ok", "revisions", "seconds")]
    for r in output["results"]:
        rows.append(
            (
                r["id"] + (" (ERROR)" if r["error"] else ""),
                "yes" if r["answer_hit"] else "NO",
                f"{r['citation_coverage']:.0%}",
                "yes" if r["no_invented_citations"] else "NO",
                str(r["revisions"]),
                f"{r['duration_s']:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))
        if i == 0:
            print("  ".join("-" * w for w in widths))
    s = output["summary"]
    print(
        f"\nanswer hit rate {s['answer_hit_rate']:.0%} | "
        f"mean citation coverage {s['mean_citation_coverage']:.0%} | "
        f"no invented citations {s['no_invented_citations_rate']:.0%} | "
        f"errors {s['errors']} | {s['total_duration_s']:.0f}s"
    )
    # Show what was missed, so a failure is actionable without opening the JSON.
    for r in output["results"]:
        missed = [c["pattern"] for c in r["checks"] if not c["found"]]
        if r["error"]:
            print(f"  {r['id']}: {r['error']}")
        elif missed:
            print(f"  {r['id']}: missing {', '.join(missed)}")


def main(
    argv: list[str] | None = None, *, nodes: Mapping[str, Callable[..., dict]] | None = None
) -> int:
    """CLI entry point. `nodes` lets tests run the harness with fake agents."""
    parser = argparse.ArgumentParser(description="Run the eval questions and score the reports.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--out", type=Path, help="results JSON (default: evals/results/<time>.json)"
    )
    parser.add_argument("--only", action="append", metavar="ID", help="run only these question ids")
    parser.add_argument("--min-answer-hit", type=float, default=0.0, help="exit 1 below this rate")
    parser.add_argument("--min-citation-coverage", type=float, default=0.0)
    args = parser.parse_args(argv)

    cases = load_questions(args.questions)
    if args.only:
        unknown = set(args.only) - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown question id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in args.only]

    # No human to approve, so review is off. No checkpointer is needed then.
    settings = get_settings().model_copy(update={"human_review": False})
    graph = build_graph(nodes, settings=settings)
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d-%H%M%S")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.id} ...", flush=True)
        results.append(run_case(graph, case, settings, run_id))

    output = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        # What produced these numbers, so two result files can be compared fairly.
        "config": {
            "llm_provider": settings.llm_provider,
            "strong_model": settings.gemini_model,
            "fast_model": settings.gemini_fast_model,
            "max_revisions": settings.max_revisions,
            "max_research_rounds": settings.max_research_rounds,
            "questions_file": str(args.questions),
        },
        "summary": summarize(results),
        "results": results,
    }
    out = args.out or RESULTS_DIR / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print_table(output)
    print(f"\nResults written to {out}")

    s = output["summary"]
    passed = (
        s["answer_hit_rate"] >= args.min_answer_hit
        and s["mean_citation_coverage"] >= args.min_citation_coverage
    )
    return 0 if passed else 1


if __name__ == "__main__":
    # Windows consoles default to a legacy code page; reports may contain
    # characters it can't print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
