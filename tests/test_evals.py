"""Offline tests for the eval harness: output format, scoring, and the ground truth."""

import json
import re

from analyst.agents import placeholders
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER
from evals import run_evals
from scripts import make_sample_data as gen

FAKE_NODES = {
    SUPERVISOR: placeholders.supervisor,
    WEB: placeholders.web_research,
    RAG: placeholders.rag,
    DATA: placeholders.data_analysis,
    CRITIC: placeholders.critic,
}


def scripted_writer(state):
    """Writes one cited and one uncited factual sentence; fails on demand."""
    if "explode" in state["question"]:
        raise RuntimeError("writer exploded")
    sid = next(iter(state["sources"]))
    draft = (
        "# Report\n\n"
        f"- Net revenue reached $466.7 million in 2024, a record year [{sid}].\n"
        "- Margins improved in every region thanks to the new footwear line.\n"
    )
    return {"draft": draft, "revision_count": 0}


QUESTIONS = """\
- id: hit
  question: What was 2024 revenue?
  expect: ['466\\.7', 'million']
  source: CSV
- id: miss
  question: What was 2023 revenue?
  expect: ['393\\.1']
- id: crash
  question: Please explode
  expect: ['anything']
"""


def test_harness_writes_results_json_with_fake_agents(tmp_path, capsys):
    questions = tmp_path / "questions.yaml"
    questions.write_text(QUESTIONS, encoding="utf-8")
    out = tmp_path / "results.json"

    code = run_evals.main(
        ["--questions", str(questions), "--out", str(out)],
        nodes={**FAKE_NODES, WRITER: scripted_writer},
    )
    assert code == 0  # no thresholds set

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"run_id", "started_at", "config", "summary", "results"}
    assert data["config"]["llm_provider"] == "fake"
    assert data["summary"] == {
        "questions": 3,
        "answer_hit_rate": 0.333,
        "mean_citation_coverage": 0.333,  # (0.5 + 0.5 + 0.0) / 3
        "no_invented_citations_rate": 0.667,  # the crashed run counts as a failure
        "errors": 1,
        "total_duration_s": data["summary"]["total_duration_s"],
    }

    hit, miss, crash = data["results"]
    expected_keys = {
        "id", "question", "source", "thread_id", "answer_hit", "checks",
        "citation_coverage", "claim_sentences", "uncited_sentences",
        "no_invented_citations", "invented_source_ids", "dangling_citation_numbers",
        "sources_cited", "sources_collected", "revisions", "research_rounds",
        "critic_verdicts", "duration_s", "error", "report",
    }  # fmt: skip
    assert all(set(r) == expected_keys for r in data["results"])

    assert hit["answer_hit"] is True
    assert hit["checks"] == [
        {"pattern": "466\\.7", "found": True},
        {"pattern": "million", "found": True},
    ]
    assert hit["citation_coverage"] == 0.5 and hit["claim_sentences"] == 2
    assert hit["uncited_sentences"] == [
        "Margins improved in every region thanks to the new footwear line."
    ]
    assert hit["no_invented_citations"] is True
    assert hit["critic_verdicts"] == ["accept"]
    assert "## Sources" in hit["report"] and hit["thread_id"].endswith("-hit")

    assert miss["answer_hit"] is False and miss["checks"][0]["found"] is False
    assert crash["error"] == "RuntimeError: writer exploded" and crash["answer_hit"] is False

    printed = capsys.readouterr().out
    assert "answer hit rate 33%" in printed
    assert "miss: missing 393\\.1" in printed and "crash: RuntimeError" in printed


def test_threshold_failure_exits_1(tmp_path):
    questions = tmp_path / "questions.yaml"
    questions.write_text(QUESTIONS, encoding="utf-8")
    args = ["--questions", str(questions), "--out", str(tmp_path / "r.json"), "--only", "miss"]
    nodes = {**FAKE_NODES, WRITER: scripted_writer}
    assert run_evals.main([*args, "--min-answer-hit", "0.5"], nodes=nodes) == 1


def test_invented_citations_are_caught_in_draft_and_final_report():
    case = run_evals.EvalCase(id="x", question="q", expect=[])
    state = {
        "draft": "# R\n\n- Revenue rose sharply to a new record level [rag:deadbeef].\n",
        "sources": {},
        # [3] has no entry in the Sources list below.
        "final_report": "# R\n\nRevenue rose [1, 3].\n\n## Sources\n\n1. **A** (data)\n",
    }
    result = run_evals.score(case, state)
    assert result["no_invented_citations"] is False
    assert result["invented_source_ids"] == ["rag:deadbeef"]
    assert result["dangling_citation_numbers"] == [3]


def test_ground_truth_matches_the_generated_data():
    """Every expected pattern in questions.yaml must appear in the data we generate.

    If someone changes the generator (seed, trends, narrative), this fails and
    points at the questions whose ground truth went stale.
    """
    summaries = gen.summarize(gen.generate_sales(42))
    facts = []
    for year, s in summaries.items():
        facts += [gen.money(s.revenue), gen.pct(s.dtc_share), gen.pct(s.gross_margin)]
        prev = summaries.get(year - 1)
        if prev:
            facts.append(gen.change(s.revenue, prev.revenue))
        for group in ("by_region", "by_product"):
            for key, value in getattr(s, group).items():
                facts += [key, gen.money(value)]
                if prev:
                    facts.append(gen.change(value, getattr(prev, group)[key]))
    facts.append(json.dumps(gen.NARRATIVE))  # the PDF-only facts (risks, reasons)
    text = "\n".join(facts)

    cases = run_evals.load_questions(run_evals.DEFAULT_QUESTIONS)
    assert 5 <= len(cases) <= 8
    stale = [(c.id, p) for c in cases for p in c.expect if not re.search(p, text, re.IGNORECASE)]
    assert stale == []
