"""Offline tests for the eval harness: output format, scoring, and the ground truth."""

import json
import re

import pytest
import yaml

from analyst.agents import placeholders
from analyst.config import PROJECT_ROOT
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER
from evals import ground_truth, run_evals
from scripts import fetch_edgar

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
        "id", "question", "source", "ground_truth", "thread_id", "answer_hit", "checks",
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


def _matches(pattern, text):
    return bool(re.search(pattern, text, re.IGNORECASE))


def test_money_pattern_accepts_equivalent_wordings_and_rejects_wrong_figures():
    p = ground_truth.money_pattern(391_035_000_000)
    for ok in [
        "$391.04 billion",
        "391.0B",
        "$391,035 million",
        "Revenue | 391,035",
        "$391 billion",
    ]:
        assert _matches(p, ok), ok
    for wrong in ["$391.1 billion", "$1391.0 billion", "$383.29 billion", "391,036 million"]:
        assert not _matches(p, wrong), wrong
    small = ground_truth.money_pattern(466.7e6)
    assert _matches(small, "$466.7 million") and _matches(small, "466.70M")
    assert not _matches(small, "$466.8 million")


def test_percent_and_per_share_patterns():
    p = ground_truth.percent_pattern(114.2047)
    assert all(_matches(p, s) for s in ["114.2%", "114.20 percent", "114%"])
    assert not _matches(p, "114.3%") and not _matches(p, "4.2%")
    small = ground_truth.percent_pattern(4.71)
    assert _matches(small, "4.7%") and not _matches(small, "5%")  # no whole-percent slack below 10
    eps = ground_truth.per_share_pattern(7.46)
    assert _matches(eps, "$7.46") and not _matches(eps, "$17.46")


FACT_QUESTIONS = """\
- id: value
  question: KRO revenue FY2024?
  facts: [{ticker: KRO, fiscal_year: 2024, metric: revenue}]
- id: growth
  question: KRO revenue growth FY2024?
  facts: [{ticker: KRO, fiscal_year: 2024, metric: revenue, compare_to: 2023}]
- id: share
  question: KRO R&D share FY2024?
  facts: [{ticker: KRO, fiscal_year: 2024, metric: rd_expense, per: revenue}]
  expect: ['research']
"""


def test_facts_resolve_into_patterns_from_the_financials_table(tmp_path, sample_data):
    out, _ = sample_data
    questions = tmp_path / "q.yaml"
    questions.write_text(FACT_QUESTIONS, encoding="utf-8")
    value, growth, share = run_evals.load_questions(questions, out / "datasets" / "financials.csv")

    assert value.ground_truth == ["KRO FY2024 revenue = 466,700,000.00 USD"]
    assert _matches(value.expect[0], "a record $466.7 million")
    assert growth.ground_truth == ["KRO revenue FY2024 vs FY2023: +18.72%"]
    assert _matches(growth.expect[0], "up 18.7% year over year")
    assert share.expect[0] == "research"  # literal patterns are kept, facts appended
    assert _matches(share.expect[1], "3.0% of revenue")  # 14.1 / 466.7


def test_fact_missing_from_the_table_is_an_error(tmp_path, sample_data):
    out, _ = sample_data
    questions = tmp_path / "q.yaml"
    questions.write_text(
        "- id: x\n  question: q\n"
        "  facts: [{ticker: BFA, fiscal_year: 2024, metric: gross_profit}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ground_truth.GroundTruthError, match="no gross_profit for BFA FY2024"):
        run_evals.load_questions(questions, out / "datasets" / "financials.csv")


def test_facts_without_a_fetched_corpus_explain_what_to_run(tmp_path):
    questions = tmp_path / "q.yaml"
    questions.write_text(FACT_QUESTIONS, encoding="utf-8")
    with pytest.raises(SystemExit, match="fetch_edgar"):
        run_evals.load_questions(questions, tmp_path / "missing.csv")


def test_question_file_is_well_formed():
    """Offline structure check of evals/questions.yaml (no corpus needed)."""
    items = yaml.safe_load(run_evals.DEFAULT_QUESTIONS.read_text(encoding="utf-8"))
    assert 10 <= len(items) <= 15
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        case = run_evals.EvalCase(**item)
        assert case.facts or case.expect, case.id
        for pattern in case.expect:
            re.compile(pattern)
        for fact in case.facts:
            assert fact["metric"] in fetch_edgar.METRICS, case.id
            assert {"ticker", "fiscal_year", "metric"} <= set(fact)
        for where in case.evidence:
            doc, _, anchor = where.partition("#")
            assert doc.endswith(".html") and anchor.startswith("item-"), where
        # Every qualitative question says where its answer lives.
        assert case.facts or case.evidence, case.id
    assert sum(bool(i.get("facts")) for i in items) >= 5  # mostly quantitative, SQL-checkable
    assert sum(bool(i.get("evidence")) for i in items) >= 3  # and some RAG questions


REAL_CSV = PROJECT_ROOT / "data" / "datasets" / "financials.csv"
REAL_DOCS = PROJECT_ROOT / "data" / "docs"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="SEC corpus not fetched (scripts/fetch_edgar.py)")
def test_ground_truth_holds_on_the_fetched_corpus():
    """Every answer in questions.yaml must be true of the pinned filings.

    Quantitative facts must resolve from financials.csv, and each qualitative
    pattern must occur in the Item named as its evidence. If a manifest
    re-pin changes the filings, this points at the questions that went stale.
    """
    from analyst.rag.loader import load_filing

    cases = run_evals.load_questions(
        run_evals.DEFAULT_QUESTIONS, REAL_CSV
    )  # raises if a fact is missing
    stale = []
    for case in (c for c in cases if c.evidence):
        texts = []
        for where in case.evidence:
            doc, _, anchor = where.partition("#")
            sections = load_filing(REAL_DOCS / doc, items=[anchor.removeprefix("item-")])
            texts += [s.text for s in sections]
        text = "\n".join(texts)
        assert text, f"{case.id}: evidence {case.evidence} not found in data/docs"
        stale += [(case.id, p) for p in case.expect if not _matches(p, text)]
    assert stale == []
