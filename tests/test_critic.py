"""Offline tests for the critic: deterministic checks, the LLM judge, budgets, and the loop."""

from analyst.agents import placeholders
from analyst.agents.critic import CriticAgent, apply_budgets, check_citations
from analyst.agents.writer import WriterAgent
from analyst.config import Settings
from analyst.graph import (
    CRITIC,
    DATA,
    FINALIZE,
    HUMAN_GATE,
    RAG,
    SUPERVISOR,
    WEB,
    WRITER,
    build_graph,
)
from analyst.llm import FakeChatModel
from analyst.state import make_source_id

QUESTION = "Summarize the company's revenue trend and key risks, with sources"
DATA_ID = make_source_id(
    "data", "SELECT fiscal_year, value FROM financials WHERE ticker = 'KRO' AND metric = 'revenue'"
)
RAG_ID = make_source_id("rag", "KRO_2024.html#item-1a/c0-1")
SOURCES = {
    DATA_ID: {
        "id": DATA_ID,
        "kind": "data",
        "title": "SQL: Revenue by year",
        "locator": "SELECT ...",
        "snippet": "2022 | 412.6\n2023 | 393.1\n2024 | 466.7",
    },
    RAG_ID: {
        "id": RAG_ID,
        "kind": "rag",
        "title": "Kestrel Ridge Outdoor Co. (KRO) 10-K FY 2024, Item 1A. Risk Factors",
        "locator": "KRO_2024.html#item-1a/c0-1",
        "snippet": "A 10% tariff would cut margin ~2 points.",
    },
}
FINDINGS = [
    {
        "subtask_id": "t1",
        "agent": "data",
        "summary": f"Revenue: 412.6, 393.1, 466.7 [{DATA_ID}].",
        "source_ids": [DATA_ID],
    },
    {
        "subtask_id": "t2",
        "agent": "rag",
        "summary": f"Tariff risk [{RAG_ID}].",
        "source_ids": [RAG_ID],
    },
]

CLEAN_DRAFT = (
    "# Revenue dipped, then hit a record\n\n"
    "## Summary\n"
    f"- Revenue fell to $393.1 million in 2023, then reached $466.7 million in 2024 [{DATA_ID}].\n"
    "The main risks are:\n"  # intro line ending in ':' needs no citation
    f"- **Tariffs:** A 10% tariff would reduce gross margin by about 2 points [{RAG_ID}].\n\n"
    "## Limitations\n"
    "- No external industry data was collected for this report, so context is limited.\n"
)

GOOD_SCORES = {"groundedness": 5, "coverage": 4, "clarity": 4}


def _settings(**kw):
    budgets = {"max_revisions": 2, "max_research_rounds": 1, **kw}
    return Settings(llm_provider="fake", _env_file=None, **budgets)


def _state(draft=CLEAN_DRAFT, **kw):
    return {
        "question": QUESTION,
        "draft": draft,
        "sources": SOURCES,
        "findings": FINDINGS,
        "revision_count": 0,
        "research_rounds": 0,
        **kw,
    }


def _critic(*responses, **settings_kw):
    llm = FakeChatModel(responses=list(responses))
    return CriticAgent(llm=llm, settings=_settings(**settings_kw)), llm


def _judgment(verdict, scores=GOOD_SCORES, feedback=(), topics=()):
    return {
        "scores": scores,
        "verdict": verdict,
        "feedback": list(feedback),
        "missing_topics": list(topics),
    }


def _review(critic, state):
    (critique,) = critic(state)["critiques"]
    return critique


# ---------------------------------------------------------------------------
# 1. Deterministic checks
# ---------------------------------------------------------------------------
def test_clean_draft_passes_the_citation_check():
    report = check_citations(CLEAN_DRAFT, SOURCES)
    assert report.ok
    assert report.counts() == {"claim_sentences": 2, "uncited": 0, "unknown_ids": 0}


def test_uncited_factual_sentences_are_caught():
    draft = (
        f"# R\n\nRevenue fell to $393.1 million in 2023 [{DATA_ID}]. "
        "Margins collapsed to 12% because of discounting in Europe.\n"
        "- Wholesale orders fell sharply across all three regions last year.\n"
        "## Revenue trend\n"  # headings are skipped
        "Short fragment.\n"  # too short to be a claim
    )
    report = check_citations(draft, SOURCES)
    assert report.uncited == [
        "Margins collapsed to 12% because of discounting in Europe.",
        "Wholesale orders fell sharply across all three regions last year.",
    ]


def test_invented_source_ids_are_caught():
    draft = f"# R\n\nRevenue rose sharply in 2024 to a new record [{DATA_ID}, rag:deadbeef]."
    report = check_citations(draft, SOURCES)
    assert report.unknown_ids == ["rag:deadbeef"]
    assert not report.ok


def test_failed_check_returns_revise_without_calling_the_judge():
    critic, llm = _critic(_judgment("accept"))
    draft = (
        CLEAN_DRAFT.replace(f" [{DATA_ID}]", "")
        + "\nMore detail here about channels [web:0badf00d].\n"
    )
    critique = _review(critic, _state(draft=draft))

    assert critique["verdict"] == "revise"
    assert llm.history == []  # the gate saved a judge call
    assert critique["checks"]["uncited"] == 1 and critique["checks"]["unknown_ids"] == 1
    assert critique["feedback"][0].startswith(
        'Add a citation to this claim or remove it: "Revenue fell'
    )
    assert "don't exist" in critique["feedback"][1]


# ---------------------------------------------------------------------------
# 2. Judge verdicts
# ---------------------------------------------------------------------------
def test_accept_path():
    critic, llm = _critic(_judgment("accept", feedback=["Minor: tighten the title."]))
    critique = _review(critic, _state())
    assert critique["verdict"] == "accept" and critique["scores"] == GOOD_SCORES
    assert critique["budget_exhausted"] is False

    prompt = "\n".join(m.content for m in llm.history[0])
    assert "Allowed verdicts for this review: accept, revise, research." in prompt
    assert QUESTION in prompt and "[S1]" in prompt and DATA_ID not in prompt  # labels, not hashes


def test_revise_path_keeps_the_feedback_list():
    points = ["Lead the Summary with yearly totals.", "Name the main growth driver."]
    critic, _ = _critic(_judgment("revise", scores={**GOOD_SCORES, "coverage": 3}, feedback=points))
    critique = _review(critic, _state())
    assert critique["verdict"] == "revise"
    assert critique["feedback"] == points and critique["missing_topics"] == []


def test_research_path_returns_missing_topics():
    critic, _ = _critic(
        _judgment(
            "research",
            feedback=["No outside context on tariffs."],
            topics=["Proposed US tariffs on Southeast Asian imports"],
        )
    )
    critique = _review(critic, _state())
    assert critique["verdict"] == "research"
    assert critique["missing_topics"] == ["Proposed US tariffs on Southeast Asian imports"]


def test_research_without_topics_becomes_revise():
    critic, _ = _critic(_judgment("research", feedback=["Something is missing."]))
    assert _review(critic, _state())["verdict"] == "revise"


def test_accept_with_a_failing_score_becomes_revise():
    critic, _ = _critic(_judgment("accept", scores={**GOOD_SCORES, "groundedness": 2}))
    critique = _review(critic, _state())
    assert critique["verdict"] == "revise"
    assert critique["feedback"] == ["Improve the weakest criteria (groundedness=2)."]


def test_judge_output_is_clamped_and_truncated():
    critic, _ = _critic(
        _judgment(
            "revise",
            scores={"groundedness": 9, "coverage": 0, "clarity": 4},
            feedback=[f"p{i}" for i in range(9)],
        )
    )
    critique = _review(critic, _state())
    assert critique["scores"] == {"groundedness": 5, "coverage": 1, "clarity": 4}
    assert len(critique["feedback"]) == 6


def test_judge_failure_accepts_for_human_review():
    critic, _ = _critic("not a judgment")
    critique = _review(critic, _state())
    assert critique["verdict"] == "accept"
    assert "human review needed" in critique["feedback"][0]


# ---------------------------------------------------------------------------
# 3. Budgets
# ---------------------------------------------------------------------------
def test_revision_budget_exhausted_forces_accept_but_keeps_feedback():
    critic, llm = _critic(_judgment("revise", feedback=["Still too long."]))
    critique = _review(critic, _state(revision_count=2))  # max_revisions=2

    assert critique["verdict"] == "accept" and critique["budget_exhausted"] is True
    assert critique["feedback"] == ["Still too long."]  # the human still sees it
    assert "Allowed verdicts for this review: accept, research." in "\n".join(
        m.content for m in llm.history[0]
    )


def test_research_budget_exhausted_turns_research_into_a_limitations_revision():
    critic, _ = _critic(
        _judgment(
            "research", feedback=["Needs industry data."], topics=["Industry tariff exposure"]
        )
    )
    critique = _review(critic, _state(research_rounds=1))  # max_research_rounds=1

    assert critique["verdict"] == "revise" and critique["budget_exhausted"] is True
    assert critique["missing_topics"] == []
    assert critique["feedback"][-1] == (
        "No research rounds left: state these gaps under Limitations: Industry tariff exposure"
    )


def test_both_budgets_exhausted_accepts_without_calling_the_judge():
    critic, llm = _critic(_judgment("revise"))
    uncited = CLEAN_DRAFT.replace(
        "## Limitations",
        "The company plans four new stores in 2025 and a Mexico factory.\n\n## Limitations",
    )
    critique = _review(critic, _state(draft=uncited, revision_count=2, research_rounds=1))

    assert critique["verdict"] == "accept" and critique["budget_exhausted"] is True
    assert llm.history == []  # nothing could change the draft, so no judge call
    # The mechanical issue is still reported, for the human at the gate.
    assert critique["checks"]["uncited"] == 1
    assert "four new stores" in critique["feedback"][0]


def test_apply_budgets_rules():
    assert apply_budgets("accept", [], [], 0, 0) == ("accept", [], [], False)
    assert apply_budgets("revise", ["x"], [], 1, 0) == ("revise", ["x"], [], False)
    assert apply_budgets("revise", ["x"], [], 0, 1) == ("accept", ["x"], [], True)
    assert apply_budgets("research", [], ["t"], 1, 1) == ("research", [], ["t"], False)
    assert apply_budgets("research", [], ["t"], 0, 0) == ("accept", [], [], True)


# ---------------------------------------------------------------------------
# The reflection loop in the graph: real writer + real critic
# ---------------------------------------------------------------------------
def _loop_graph(writer_responses, critic_responses, **settings_kw):
    settings = _settings(**settings_kw)
    writer = WriterAgent(llm=FakeChatModel(responses=writer_responses), settings=settings)
    critic = CriticAgent(llm=FakeChatModel(responses=critic_responses), settings=settings)
    return build_graph(
        {
            SUPERVISOR: placeholders.supervisor,
            WEB: placeholders.web_research,
            RAG: placeholders.rag,
            DATA: placeholders.data_analysis,
            WRITER: writer,
            CRITIC: critic,
        },
        settings=settings,
    ), critic


def _run(graph):
    visited, final = [], {}
    for mode, chunk in graph.stream({"question": QUESTION}, stream_mode=["updates", "values"]):
        if mode == "updates":
            visited.extend(chunk)
        else:
            final = chunk
    return visited, final


def test_loop_uncited_draft_is_revised_then_accepted():
    sloppy = (
        "# R\n\n## Summary\n- Every region grew strongly and margins improved across the board.\n"
    )
    fixed = "# R\n\n## Summary\n- The specialists' findings agree on the main points [S1, S2].\n"
    graph, critic = _loop_graph([sloppy, fixed], [_judgment("accept")])
    visited, final = _run(graph)

    assert visited.count(WRITER) == 2 and visited.count(CRITIC) == 2
    assert [c["verdict"] for c in final["critiques"]] == ["revise", "accept"]
    assert len(critic.llm.history) == 1  # the first review never reached the judge
    assert visited[-2:] == [HUMAN_GATE, FINALIZE] and "[1, 2]" in final["final_report"]


def test_loop_stops_when_revision_budget_runs_out():
    draft = "# R\n\n- The specialists' findings agree on the main points [S1].\n"
    graph, _ = _loop_graph(
        [draft], [_judgment("revise", feedback=["Be sharper."])], max_revisions=1
    )
    visited, final = _run(graph)

    assert visited.count(WRITER) == 2  # first draft + 1 revision
    assert [c["verdict"] for c in final["critiques"]] == ["revise", "accept"]
    assert final["critiques"][-1]["budget_exhausted"] is True
    assert visited[-2:] == [HUMAN_GATE, FINALIZE]


def test_loop_research_round_then_budget_forces_a_limitations_revision():
    draft = "# R\n\n- The specialists' findings agree on the main points [S1].\n"
    research = _judgment("research", topics=["Competitor tariff exposure"])
    graph, _ = _loop_graph([draft], [research, research, _judgment("accept")])
    visited, final = _run(graph)

    assert visited.count(SUPERVISOR) == 2  # one research round (the budget)
    assert [c["verdict"] for c in final["critiques"]] == ["research", "revise", "accept"]
    assert "state these gaps under Limitations" in final["critiques"][1]["feedback"][-1]
    assert visited[-2:] == [HUMAN_GATE, FINALIZE]
