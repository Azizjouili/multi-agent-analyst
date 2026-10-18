"""Offline tests for the writer agent and the finalize node."""

from pydantic import SecretStr

from analyst.agents import placeholders
from analyst.agents.citations import rewrite_citations
from analyst.agents.finalize import finalize, render_final_report
from analyst.agents.writer import WriterAgent, collect_feedback
from analyst.config import Settings
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER, build_graph
from analyst.llm import FakeChatModel
from analyst.state import make_source_id

QUESTION = "Summarize the company's revenue trend and key risks, with sources"

SQL = (
    "SELECT year, ROUND(SUM(revenue_usd) / 1e6, 1) AS revenue_musd "
    "FROM quarterly_sales GROUP BY year"
)
DATA_ID = make_source_id("data", SQL)
RAG_ID = make_source_id("rag", "annual_report_2024.pdf#p4")
WEB_ID = make_source_id("web", "https://news.example.com/outdoor-2023")

SOURCES = {
    DATA_ID: {
        "id": DATA_ID,
        "kind": "data",
        "title": "SQL: Net revenue by year",
        "locator": SQL,
        "snippet": "year | revenue_musd\n2022 | 412.6\n2023 | 393.1\n2024 | 466.7",
    },
    RAG_ID: {
        "id": RAG_ID,
        "kind": "rag",
        "title": "Kestrel Ridge Outdoor Co. Annual Report 2024, p. 4",
        "locator": "annual_report_2024.pdf#p4",
        "snippet": "Tariffs and trade policy. A 10% tariff...",
    },
    WEB_ID: {
        "id": WEB_ID,
        "kind": "web",
        "title": "Outdoor gear sales dropped in 2023",
        "locator": "https://news.example.com/outdoor-2023",
        "snippet": "Sales fell across the industry.",
    },
}
FINDINGS = [
    {
        "subtask_id": "t1",
        "agent": "data",
        "summary": f"Revenue fell to $393.1 million in 2023 [{DATA_ID}].",
        "source_ids": [DATA_ID],
    },
    {
        "subtask_id": "t2",
        "agent": "rag",
        "summary": f"A 10% tariff would cut margin ~2 points [{RAG_ID}].",
        "source_ids": [RAG_ID],
    },
    {
        "subtask_id": "t3",
        "agent": "web",
        "summary": f"Industry sales fell in 2023 [{WEB_ID}].",
        "source_ids": [WEB_ID],
    },
]
# Labels follow the order in which findings cite sources: S1=data, S2=rag, S3=web.
LABELS = {"S1": DATA_ID, "S2": RAG_ID, "S3": WEB_ID}

DRAFT_TEXT = (
    "# Revenue dipped, then hit a record\n\n## Summary\n"
    "- Revenue fell to $393.1 million in 2023 [S1], in line with the industry [S3].\n\n"
    "## Key risks\n- Tariffs could cut gross margin by about 2 points [S2]."
)


def _state(**overrides):
    return {"question": QUESTION, "findings": FINDINGS, "sources": SOURCES, **overrides}


def _writer(*responses):
    llm = FakeChatModel(responses=list(responses))
    return WriterAgent(llm=llm, settings=Settings(llm_provider="fake", _env_file=None)), llm


def _prompt(llm, call=0):
    return "\n".join(m.content for m in llm.history[call])


def _stored(text):
    """What a draft looks like in state: labels resolved to source ids."""
    return rewrite_citations(text, LABELS)[0]


# ---------------------------------------------------------------------------
# Writer: first draft
# ---------------------------------------------------------------------------
def test_first_draft_stores_source_ids_and_strips_a_bibliography():
    writer, llm = _writer(DRAFT_TEXT + "\n\n## Sources\n- [S1] some query\n- [S2] a report")
    out = writer(_state())

    assert out["revision_count"] == 0
    draft = out["draft"]
    assert f"[{DATA_ID}]" in draft and f"[{RAG_ID}]" in draft and f"[{WEB_ID}]" in draft
    assert "[S1]" not in draft  # labels are resolved before storing
    assert "## Sources" not in draft  # finalize generates the bibliography, not the model


def test_prompt_shows_findings_and_sources_with_short_labels():
    writer, llm = _writer(DRAFT_TEXT)
    writer(_state())
    prompt = _prompt(llm)
    assert QUESTION in prompt
    assert "[F1] (data, sub-task t1) Revenue fell to $393.1 million in 2023 [S1]." in prompt
    assert "[S2] (rag) Kestrel Ridge Outdoor Co. Annual Report 2024, p. 4" in prompt
    assert DATA_ID not in prompt  # the model never sees raw hash ids
    assert "Your previous draft" not in prompt


def test_invented_labels_are_dropped_from_the_draft():
    writer, _ = _writer("# R\n\nRevenue fell [S1]. Margins collapsed [S9]. Both [S2, S7].")
    draft = writer(_state())["draft"]
    assert draft == f"# R\n\nRevenue fell [{DATA_ID}]. Margins collapsed. Both [{RAG_ID}]."


def test_keeps_markdown_indentation_intact():
    writer, _ = _writer("# R\n\n- Revenue [S1]\n  - by region [S1]\n    - Asia-Pacific grew [S9]")
    draft = writer(_state())["draft"]
    assert "\n  - by region" in draft and "\n    - Asia-Pacific grew" in draft


def test_heading_that_merely_starts_with_sources_is_kept():
    writer, _ = _writer("# R\n\n## Sources of revenue growth\nDTC grew [S1].")
    assert "## Sources of revenue growth" in writer(_state())["draft"]


# ---------------------------------------------------------------------------
# Writer: revisions
# ---------------------------------------------------------------------------
REVISE = {
    "scores": {"coverage": 3},
    "verdict": "revise",
    "feedback": "Quantify the 2024 recovery and name the main driver.",
    "missing_topics": ["Direct-to-consumer share of revenue"],
}


def test_revision_with_critic_feedback():
    writer, llm = _writer(DRAFT_TEXT + "\n\nIn 2024 revenue rose to $466.7 million [S1].")
    out = writer(_state(draft=_stored(DRAFT_TEXT), critiques=[REVISE], revision_count=0))

    assert out["revision_count"] == 1
    assert "$466.7 million" in out["draft"]
    prompt = _prompt(llm)
    # The previous draft is shown with labels, and every point is numbered.
    assert "Your previous draft:\n# Revenue dipped" in prompt and "[S1]" in prompt
    assert "1. [Critic] Quantify the 2024 recovery and name the main driver." in prompt
    assert "2. [Critic] Cover: Direct-to-consumer share of revenue" in prompt
    assert "Address EVERY feedback point" in prompt


def test_revision_with_human_feedback_takes_priority():
    accept = {**REVISE, "verdict": "accept", "feedback": "Looks good.", "missing_topics": []}
    writer, llm = _writer(DRAFT_TEXT + "\n\n## Channel mix\nDTC grew [S1].")
    out = writer(
        _state(
            draft=_stored(DRAFT_TEXT),
            critiques=[accept],
            human_feedback="Add a short section on the channel mix.",
            approved=False,
            revision_count=1,
        )
    )
    assert out["revision_count"] == 2
    prompt = _prompt(llm)
    assert (
        "1. [Human reviewer - highest priority] Add a short section on the channel mix." in prompt
    )
    assert "Looks good" not in prompt  # an "accept" critique has nothing to fix


def test_feedback_ordering_puts_human_before_critic():
    points = collect_feedback(
        {
            "question": QUESTION,
            "human_feedback": "Shorter, please.",
            "approved": False,
            "critiques": [REVISE],
        }
    )
    assert points[0].startswith("[Human reviewer")
    assert points[1] == "[Critic] Quantify the 2024 recovery and name the main driver."


def test_revision_after_research_round_sees_new_findings():
    new_src = make_source_id("web", "https://trade.example.org/tariffs")
    sources = {
        **SOURCES,
        new_src: {
            "id": new_src,
            "kind": "web",
            "title": "Tariff tracker",
            "locator": "https://trade.example.org/tariffs",
            "snippet": "25% proposed.",
        },
    }
    findings = [
        *FINDINGS,
        {
            "subtask_id": "r1-1",
            "agent": "web",
            "summary": f"A 25% tariff is proposed [{new_src}].",
            "source_ids": [new_src],
        },
    ]
    research = {**REVISE, "verdict": "research", "feedback": "Size the tariff threat."}
    writer, llm = _writer(DRAFT_TEXT + "\nA 25% tariff is proposed [S4].")
    out = writer(
        _state(draft=_stored(DRAFT_TEXT), critiques=[research], findings=findings, sources=sources)
    )
    assert f"[{new_src}]" in out["draft"]
    assert "[F4] (web, sub-task r1-1) A 25% tariff is proposed [S4]." in _prompt(llm)


# ---------------------------------------------------------------------------
# Writer: failures
# ---------------------------------------------------------------------------
class BrokenLLM(FakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")


def _broken_writer():
    return WriterAgent(llm=BrokenLLM(), settings=Settings(llm_provider="fake", _env_file=None))


def test_first_draft_failure_falls_back_to_a_cited_findings_report():
    draft = _broken_writer()(_state())["draft"]
    assert draft.startswith(f"# Research report: {QUESTION}")
    assert f"[{DATA_ID}]" in draft and "## Limitations" in draft and "503 UNAVAILABLE" in draft


def test_revision_failure_keeps_the_previous_draft_but_counts_the_revision():
    previous = _stored(DRAFT_TEXT)
    out = _broken_writer()(_state(draft=previous, critiques=[REVISE], revision_count=1))
    assert out == {"draft": previous, "revision_count": 2}


def test_writer_uses_the_strong_tier():
    s = Settings(
        llm_provider="gemini",
        google_api_key=SecretStr("k"),
        gemini_model="strong-x",
        gemini_fast_model="fast-x",
        _env_file=None,
    )
    assert WriterAgent(settings=s).llm.model.endswith("strong-x")


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------
def test_finalize_renumbers_in_order_of_first_appearance():
    draft = (
        f"Industry fell [{WEB_ID}]. Revenue fell [{DATA_ID}, {WEB_ID}]. "
        f"Tariffs [{RAG_ID}]. Again revenue [{DATA_ID}]."
    )
    report, cited = render_final_report(draft, SOURCES)
    body = report.split("\n\n## Sources")[0]
    assert body == "Industry fell [1]. Revenue fell [1, 2]. Tariffs [3]. Again revenue [2]."
    assert cited == [WEB_ID, DATA_ID, RAG_ID]


def test_finalize_bibliography_has_label_type_and_description():
    draft = f"Industry fell [{WEB_ID}]. Revenue fell [{DATA_ID}]. Tariffs [{RAG_ID}]."
    report, _ = render_final_report(draft, SOURCES)
    bibliography = report.split("## Sources\n\n")[1].strip().splitlines()
    assert bibliography == [
        "1. **Outdoor gear sales dropped in 2023** (web: web page) - <https://news.example.com/outdoor-2023>",
        f"2. **Net revenue by year** (data: SQL query) - `{SQL}`",
        "3. **Kestrel Ridge Outdoor Co. Annual Report 2024, p. 4** (rag: internal document) - "
        "annual_report_2024.pdf, page 4",
    ]


def test_finalize_drops_invented_ids_and_lists_only_cited_sources():
    fake_id = "rag:deadbeef"  # well-formed, but not in sources
    draft = (
        f"Revenue fell [{DATA_ID}, {fake_id}]. Made up [{fake_id}]. "
        "See [the OIA report](https://x.org) [sic]."
    )
    report, cited = render_final_report(draft, SOURCES)
    assert report.startswith(
        "Revenue fell [1]. Made up. See [the OIA report](https://x.org) [sic]."
    )
    assert cited == [DATA_ID]
    assert "deadbeef" not in report and WEB_ID not in report and "Annual Report" not in report


def test_finalize_truncates_long_sql_to_one_line():
    long_sql = "SELECT " + ", ".join(f"col_{i}" for i in range(60)) + "\nFROM quarterly_sales"
    sid = make_source_id("data", long_sql)
    src = {
        sid: {"id": sid, "kind": "data", "title": "SQL: Wide", "locator": long_sql, "snippet": ""}
    }
    line = render_final_report(f"x [{sid}]", src)[0].splitlines()[-1]
    assert "\n" not in line and line.endswith("...`") and len(line) < 230


def test_finalize_without_citations_says_so():
    report = finalize(
        {"question": QUESTION, "draft": "# Empty\n\nNo evidence.", "sources": SOURCES}
    )
    assert report["final_report"].endswith("## Sources\n\n_No sources were cited._\n")


# ---------------------------------------------------------------------------
# End to end: real writer + real finalize inside the graph
# ---------------------------------------------------------------------------
def test_writer_and_finalize_inside_the_graph():
    writer, _ = _writer("# Report\n\n## Summary\n- Placeholder findings agree [S1, S2]. Also [S3].")
    graph = build_graph(
        {
            SUPERVISOR: placeholders.supervisor,
            WEB: placeholders.web_research,
            RAG: placeholders.rag,
            DATA: placeholders.data_analysis,
            WRITER: writer,
            CRITIC: placeholders.critic,
        },
        settings=Settings(llm_provider="fake", _env_file=None),
    )
    report = graph.invoke({"question": QUESTION})["final_report"]
    assert "- Placeholder findings agree [1, 2]. Also [3]." in report
    assert report.count("\n1. **") == 1 and "\n3. **" in report
