"""Offline tests for the data-analysis agent: real SQLite + scripted fake LLM."""

import pytest

from analyst.agents import placeholders
from analyst.agents.data_analyst import MAX_REPAIRS, DataAnalystAgent
from analyst.config import Settings
from analyst.graph import DATA, RAG, build_graph
from analyst.llm import FakeChatModel
from analyst.tools.sql import DataNotReadyError, run_sql
from tests.edgar_fixture import FACTS

QUESTION = "Summarize Kestrel Ridge's revenue trend and key risks, with sources"
ROWS = sum(len(metrics) for metrics in FACTS.values())

REVENUE_BY_YEAR = (
    "SELECT fiscal_year, ROUND(value / 1e6, 1) AS revenue_musd FROM financials "
    "WHERE ticker = 'KRO' AND metric = 'revenue' ORDER BY fiscal_year"
)
GROWTH = (
    "SELECT fiscal_year, ROUND(100.0 * (value - LAG(value) OVER (ORDER BY fiscal_year)) "
    "/ LAG(value) OVER (ORDER BY fiscal_year), 1) AS yoy_growth_pct FROM financials "
    "WHERE ticker = 'KRO' AND metric = 'revenue'"
)


def _task(instruction="Net revenue trend by year", id_="t3"):
    return {
        "question": QUESTION,
        "subtask": {"id": id_, "agent": "data", "instruction": instruction},
    }


def _agent(db, *responses):
    llm = FakeChatModel(responses=list(responses))
    settings = Settings(llm_provider="fake", _env_file=None)
    return DataAnalystAgent(llm=llm, settings=settings, db_path=db), llm


def _plan(*queries):
    return {"queries": [{"purpose": p, "sql": s} for p, s in queries]}


def _text(messages):
    return "\n".join(m.content for m in messages)


def test_happy_path_cites_the_sql_query_as_source(sample_db):
    agent, llm = _agent(
        sample_db,
        _plan(("Net revenue by year", REVENUE_BY_YEAR), ("YoY growth", GROWTH)),
        "Revenue dipped to $393.1 million in 2023 [Q1] (-4.7% [Q2]) and reached "
        "$466.7 million in 2024 [Q1, Q2].",
    )
    out = agent(_task())

    (finding,) = out["findings"]
    assert finding["agent"] == "data" and finding["subtask_id"] == "t3"
    assert "[data:" in finding["summary"] and "[Q1" not in finding["summary"]
    assert len(out["sources"]) == 2 and finding["source_ids"] == list(out["sources"])

    by_title = {s["title"]: s for s in out["sources"].values()}
    revenue_src = by_title["SQL: Net revenue by year"]
    assert revenue_src["kind"] == "data"
    assert revenue_src["locator"] == REVENUE_BY_YEAR  # the citation is the query itself
    assert "2024 | 466.7" in revenue_src["snippet"]  # the evidence is the real result

    # The interpretation prompt contained the real result tables.
    interpret_prompt = _text(llm.history[-1])
    assert "2023 | 393.1" in interpret_prompt and "yoy_growth_pct" in interpret_prompt


def test_schema_with_real_values_goes_into_the_sql_prompt(sample_db):
    agent, llm = _agent(sample_db, _plan(("Revenue", REVENUE_BY_YEAR)), "Revenue [Q1].")
    agent(_task())
    plan_prompt = _text(llm.history[0])
    assert "'KRO'" in plan_prompt and "fiscal_year INTEGER: range 2022 to 2024" in plan_prompt
    assert "BFA = Bluefin Analytics Inc.: FY 2024; no gross_profit" in plan_prompt


def test_failed_query_is_repaired_with_the_error_message(sample_db):
    agent, llm = _agent(
        sample_db,
        _plan(("Net revenue by year", "SELECT fiscal_year, revnue FROM financials")),
        {"purpose": "Net revenue by year", "sql": REVENUE_BY_YEAR},  # the repair
        "Revenue reached $466.7 million in 2024 [Q1].",
    )
    out = agent(_task())

    repair_prompt = _text(llm.history[1])
    assert "no such column: revnue" in repair_prompt  # the model saw the exact error
    (source,) = out["sources"].values()
    assert source["locator"] == REVENUE_BY_YEAR  # the repaired query is what gets cited


def test_write_attempt_is_rejected_then_repaired(sample_db):
    agent, _ = _agent(
        sample_db,
        _plan(("Clean data", "DELETE FROM financials WHERE fiscal_year = 2022")),
        {"purpose": "Revenue", "sql": REVENUE_BY_YEAR},
        "Revenue [Q1].",
    )
    out = agent(_task())
    assert out["findings"][0]["source_ids"]
    assert run_sql("SELECT COUNT(*) FROM financials", sample_db).rows == [(ROWS,)]


def test_gives_up_after_max_repairs_without_touching_the_data(sample_db):
    bad = {"purpose": "x", "sql": "DROP TABLE financials"}
    agent, llm = _agent(sample_db, {"queries": [bad]}, *([bad] * MAX_REPAIRS))
    out = agent(_task())

    assert out["findings"][0]["summary"].startswith("ERROR: every SQL query failed")
    assert out["sources"] == {}
    assert len(llm.history) == 1 + MAX_REPAIRS  # bounded: plan + repairs, no interpretation
    assert run_sql("SELECT COUNT(*) FROM financials", sample_db).rows == [(ROWS,)]


def test_one_failing_query_does_not_sink_the_others(sample_db):
    bad = ("Broken", "SELECT nope FROM nowhere")
    agent, _ = _agent(
        sample_db,
        _plan(("Revenue", REVENUE_BY_YEAR), bad),
        *([{"purpose": "Broken", "sql": "SELECT nope FROM nowhere"}] * MAX_REPAIRS),
        "Revenue reached $466.7 million [Q1].",
    )
    out = agent(_task())
    (source,) = out["sources"].values()
    assert source["locator"] == REVENUE_BY_YEAR


def test_planning_failure_becomes_an_error_finding(sample_db):
    agent, _ = _agent(sample_db, "not a plan")
    out = agent(_task())
    assert out["findings"][0]["summary"].startswith("ERROR: could not plan")


def test_hallucinated_citation_is_dropped(sample_db):
    agent, _ = _agent(
        sample_db, _plan(("Revenue", REVENUE_BY_YEAR)), "Revenue rose [Q1]. Margins too [Q7]."
    )
    out = agent(_task())
    assert out["findings"][0]["summary"].endswith("Margins too.")
    assert len(out["sources"]) == 1


def test_missing_database_raises_setup_error(tmp_path):
    agent, _ = _agent(tmp_path / "missing.db", _plan(("Revenue", REVENUE_BY_YEAR)))
    with pytest.raises(DataNotReadyError):
        agent(_task())


def test_data_agent_runs_inside_the_graph(sample_db):
    agent, _ = _agent(
        sample_db,
        _plan(("Net revenue by year", REVENUE_BY_YEAR)),
        "Revenue reached $466.7 million in 2024 [Q1].",
    )
    graph = build_graph(
        {
            DATA: agent,
            RAG: placeholders.rag,
            "web_research": placeholders.web_research,
            "supervisor": placeholders.supervisor,
            "writer": placeholders.writer,
            "critic": placeholders.critic,
        },
        settings=Settings(llm_provider="fake", _env_file=None),
    )
    final = graph.invoke({"question": QUESTION})

    data_findings = [f for f in final["findings"] if f["agent"] == "data"]
    assert len(data_findings) == 1 and data_findings[0]["source_ids"]
    assert any(sid.startswith("data:") for sid in final["sources"])
