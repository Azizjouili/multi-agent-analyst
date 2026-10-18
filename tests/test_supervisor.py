"""Offline tests for the supervisor: first-round planning and gap-filling research rounds."""

import pytest

from analyst.agents import placeholders
from analyst.agents.supervisor import SupervisorAgent, describe_resources
from analyst.config import Settings
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER, build_graph
from analyst.llm import FakeChatModel

QUESTION = "Summarize the company's revenue trend and key risks, with sources"


@pytest.fixture
def settings(sample_index, sample_db):
    return Settings(llm_provider="fake", _env_file=None, index_dir=sample_index, db_path=sample_db)


def _agent(settings, *responses):
    llm = FakeChatModel(responses=list(responses))
    return SupervisorAgent(llm=llm, settings=settings), llm


def _plan(*tasks):
    return {"subtasks": [{"agent": a, "instruction": i} for a, i in tasks]}


def _text(messages):
    return "\n".join(m.content for m in messages)


FIRST_PLAN = _plan(
    ("data", "Net revenue by year 2022-2024 with year-over-year growth, by region and channel"),
    ("rag", "Key risk factors named in the 2024 annual report, with any quantified impact"),
    ("rag", "Management's explanation of the 2023 revenue decline and the 2024 recovery"),
    ("web", "Outdoor-industry demand and wholesale destocking trends in 2023-2024"),
)


# ---------------------------------------------------------------------------
# First round
# ---------------------------------------------------------------------------
def test_first_round_decomposes_the_question_into_assigned_subtasks(settings):
    agent, llm = _agent(settings, FIRST_PLAN)
    out = agent({"question": QUESTION})

    assert out["research_rounds"] == 0
    assert [t["id"] for t in out["plan"]] == ["t1", "t2", "t3", "t4"]
    assert [t["agent"] for t in out["plan"]] == ["data", "rag", "rag", "web"]
    assert out["plan"][0]["instruction"].startswith("Net revenue by year")

    # The prompt grounds the planner in what each specialist can actually reach.
    prompt = _text(llm.history[0])
    assert QUESTION in prompt
    assert "KRO_2024.html" in prompt
    # The table catalog is read live from the database, with the values a
    # planner needs (companies, metrics, years): no supervisor change needed
    # when the data changes.
    assert (
        "financials (23 rows: ticker [BFA|KRO], company [Bluefin Analytics Inc.|"
        "Kestrel Ridge Outdoor Co.], cik, fiscal_year 2022-2024, period_end" in prompt
    )
    assert "metric [eps_diluted|gross_profit|net_income|rd_expense|revenue|total_assets]" in prompt


def test_duplicates_are_dropped_and_plan_size_is_capped(settings):
    capped = settings.model_copy(update={"max_subtasks": 2})
    agent, _ = _agent(
        capped,
        _plan(
            ("data", "Net revenue by year 2022-2024"),
            ("data", "net revenue by year   2022-2024"),  # same, modulo case and spacing
            ("rag", "Key risk factors in the 2024 report"),
            ("web", "Industry demand trends 2023-2024"),
        ),
    )
    plan = agent({"question": QUESTION})["plan"]
    assert [(t["id"], t["agent"]) for t in plan] == [("t1", "data"), ("t2", "rag")]


def test_unavailable_specialists_are_marked_and_their_tasks_dropped(settings, tmp_path):
    no_docs = settings.model_copy(update={"index_dir": tmp_path / "no-index"})
    agent, llm = _agent(no_docs, FIRST_PLAN)
    plan = agent({"question": QUESTION})["plan"]

    assert {t["agent"] for t in plan} == {"data", "web"}  # rag tasks removed
    assert "(unavailable: no documents indexed)" in _text(llm.history[0])


def test_planning_failure_falls_back_to_one_task_per_available_specialist(settings):
    agent, _ = _agent(settings, "not a plan")
    plan = agent({"question": QUESTION})["plan"]
    assert [t["agent"] for t in plan] == ["web", "rag", "data"]
    assert all(t["instruction"] == QUESTION for t in plan)


def test_resource_catalog_reflects_what_exists(settings, tmp_path):
    r = describe_resources(settings)
    assert r.available == ["web", "rag", "data"]
    assert r.docs == ["BFA_2024.html", "KRO_2022.html", "KRO_2023.html", "KRO_2024.html"]
    empty = describe_resources(
        settings.model_copy(update={"index_dir": tmp_path, "db_path": tmp_path / "x.db"})
    )
    assert empty.available == ["web"]


# ---------------------------------------------------------------------------
# Research (gap-filling) round
# ---------------------------------------------------------------------------
PRIOR_FINDINGS = [
    {
        "subtask_id": "t1",
        "agent": "data",
        "summary": "Revenue fell 4.7% in 2023 then rose 18.7% [data:1].",
        "source_ids": ["data:1"],
    },
    {
        "subtask_id": "t2",
        "agent": "rag",
        "summary": "2024 risks: tariffs, supplier concentration [rag:1].",
        "source_ids": ["rag:1"],
    },
]
RESEARCH_CRITIQUE = {
    "scores": {"coverage": 2},
    "verdict": "research",
    "feedback": "The tariff risk is not quantified against industry exposure.",
    "missing_topics": [
        "Size of proposed US tariffs on Southeast Asian imports",
        "Competitor exposure to the same tariffs",
    ],
}


def _research_state(**overrides):
    state = {
        "question": QUESTION,
        "plan": [{"id": "t1", "agent": "data", "instruction": "x"}],
        "findings": PRIOR_FINDINGS,
        "critiques": [RESEARCH_CRITIQUE],
        "research_rounds": 0,
    }
    return {**state, **overrides}


def test_research_round_plans_only_the_gaps(settings):
    agent, llm = _agent(
        settings,
        _plan(
            ("web", "Size and status of proposed US tariffs on goods imported from Southeast Asia"),
            ("web", "How exposed competing outdoor brands are to Southeast Asia tariffs"),
        ),
    )
    out = agent(_research_state())

    assert out["research_rounds"] == 1
    assert [t["id"] for t in out["plan"]] == ["r1-1", "r1-2"]  # ids can't collide with round 1
    assert {t["agent"] for t in out["plan"]} == {"web"}

    # The planner saw the gaps AND what is already known, so it doesn't redo work.
    prompt = _text(llm.history[0])
    assert "Competitor exposure to the same tariffs" in prompt
    assert "Revenue fell 4.7% in 2023" in prompt
    assert "do not repeat research already done" in prompt


def test_research_round_is_capped_at_three_subtasks(settings):
    agent, _ = _agent(
        settings, _plan(*[("web", f"Gap number {i} needs research") for i in range(6)])
    )
    assert len(agent(_research_state())["plan"]) == 3


def test_research_round_uses_feedback_when_no_topics_listed(settings):
    agent, llm = _agent(
        settings, _plan(("rag", "Quantified tariff exposure from the annual reports"))
    )
    critique = {**RESEARCH_CRITIQUE, "missing_topics": []}
    agent(_research_state(critiques=[critique]))
    assert "not quantified against industry exposure" in _text(llm.history[0])


def test_research_round_failure_sends_each_gap_to_web(settings):
    agent, _ = _agent(settings, "garbled")
    plan = agent(_research_state(research_rounds=1))["plan"]
    assert [t["id"] for t in plan] == ["r2-1", "r2-2"]
    assert [t["instruction"] for t in plan] == RESEARCH_CRITIQUE["missing_topics"]


# ---------------------------------------------------------------------------
# Inside the graph: a full research loop with the real supervisor
# ---------------------------------------------------------------------------
def test_research_loop_keeps_round_one_findings_and_dispatches_only_gaps(settings):
    supervisor, _ = _agent(
        settings,
        FIRST_PLAN,
        _plan(("web", "Size and status of proposed US tariffs on Southeast Asian imports")),
    )
    verdicts = iter(["research", "accept"])

    def critic(state):
        verdict = next(verdicts)
        return {"critiques": [{**RESEARCH_CRITIQUE, "verdict": verdict}]}

    graph = build_graph(
        {
            SUPERVISOR: supervisor,
            WEB: placeholders.web_research,
            RAG: placeholders.rag,
            DATA: placeholders.data_analysis,
            CRITIC: critic,
            WRITER: placeholders.writer,
        },
        settings=settings,
    )
    visited, final = [], {}
    for mode, chunk in graph.stream({"question": QUESTION}, stream_mode=["updates", "values"]):
        if mode == "updates":
            visited.extend(chunk)
        else:
            final = chunk

    assert visited.count(SUPERVISOR) == 2
    assert final["research_rounds"] == 1
    # 4 findings from round 1 plus 1 from the gap round: nothing was thrown away.
    # (Sorted, because order within a parallel fan-out isn't guaranteed.)
    ids = sorted(f["subtask_id"] for f in final["findings"])
    assert ids == ["r1-1", "t1", "t2", "t3", "t4"]
    # Round 2 dispatched only the gap: one extra web run, no extra rag/data runs.
    assert visited.count(WEB) == 2 and visited.count(RAG) == 2 and visited.count(DATA) == 1
    assert final["plan"] == [
        {
            "id": "r1-1",
            "agent": "web",
            "instruction": "Size and status of proposed US tariffs on Southeast Asian imports",
        }
    ]
