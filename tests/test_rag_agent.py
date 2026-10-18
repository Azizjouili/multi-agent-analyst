"""Offline tests for the RAG agent node: fake embedder + scripted fake LLM."""

import pytest

from analyst.agents import placeholders
from analyst.agents.rag import RagAgent, group_by_section, resolve_citations
from analyst.config import Settings
from analyst.graph import RAG, build_graph
from analyst.llm import FakeChatModel
from analyst.rag.retriever import Retriever
from analyst.rag.vector_store import Hit, VectorStore
from tests.fakes import BagOfWordsEmbedder

QUESTION = "Summarize Kestrel Ridge's revenue trend and key risks, with sources"


def _settings():
    return Settings(llm_provider="fake", _env_file=None, retrieval_k=5)


def _task(instruction, id_="t2"):
    return {
        "question": QUESTION,
        "subtask": {"id": id_, "agent": "rag", "instruction": instruction},
    }


@pytest.fixture(scope="module")
def retriever(sample_index):
    return Retriever(VectorStore.load(sample_index), BagOfWordsEmbedder())


def _agent(retriever, *responses):
    llm = FakeChatModel(responses=list(responses))
    return RagAgent(retriever=retriever, llm=llm, settings=_settings()), llm


def _prompt_text(messages) -> str:
    return "\n".join(m.content for m in messages)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _hit(doc, anchor, index, text, score=1.0):
    rec = {
        "doc": doc,
        "anchor": anchor,
        "section": f"Item {anchor.removeprefix('item-').upper()}. Risk Factors",
        "index": index,
        "text": text,
        "title": f"Acme Corp (ACME) 10-K FY 2024 [{doc}]",
    }
    return Hit(record=rec, score=score)


def test_adjacent_chunks_merge_into_one_passage_in_rank_order():
    hits = [
        _hit("B_2024.html", "item-1a", 4, "middle. Then the end."),
        _hit("A_2024.html", "item-1a", 0, "another filing"),
        _hit("B_2024.html", "item-1a", 3, "The start. In the middle."),  # overlaps c4
        _hit("B_2024.html", "item-1a", 9, "far away"),  # same Item, not adjacent
    ]
    excerpts = group_by_section(hits)
    assert [e.label for e in excerpts] == ["D1", "D2", "D3"]
    first, second, third = (e.source for e in excerpts)
    assert first["locator"] == "B_2024.html#item-1a/c3-4"  # names the chunks
    assert first["id"].startswith("rag:")
    # Reading order restored, and the overlap the chunker repeated is printed once.
    assert first["snippet"] == "The start. In the middle. Then the end."
    assert first["title"] == "Acme Corp (ACME) 10-K FY 2024 [B_2024.html], Item 1A. Risk Factors"
    assert second["locator"] == "A_2024.html#item-1a/c0"
    assert third["locator"] == "B_2024.html#item-1a/c9"


def test_passage_ids_are_stable_content_hashes():
    from analyst.state import make_source_id

    hits = [_hit("B_2024.html", "item-7", 2, "text")]
    (excerpt,) = group_by_section(hits)
    assert excerpt.source["id"] == make_source_id("rag", "B_2024.html#item-7/c2")
    assert group_by_section(hits)[0].source["id"] == excerpt.source["id"]  # same on every run


def test_resolve_citations_maps_labels_and_drops_hallucinations():
    labels = {"D1": "rag:aaaa", "D2": "rag:bbbb"}
    text = "Margin fell [D1]. Tariffs loom [D2, D1]. Made up [D9]. Quote [sic] kept."
    out, used = resolve_citations(text, labels)
    assert (
        out
        == "Margin fell [rag:aaaa]. Tariffs loom [rag:bbbb, rag:aaaa]. Made up. Quote [sic] kept."
    )
    assert used == ["rag:aaaa", "rag:bbbb"]


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------
def test_answers_from_2024_filing_with_year_filter_and_passage_sources(retriever):
    agent, llm = _agent(
        retriever,
        {"queries": ["risk factors tariffs", "supplier concentration"], "years": [2024]},
        "A 10% tariff would cut gross margin by about 2 points [D1]. "
        "Tanh Loc produced about 31% of units [D1, D2].",
    )
    out = agent(_task("What are the key risks in 2024?"))

    (finding,) = out["findings"]
    assert finding["agent"] == "rag" and finding["subtask_id"] == "t2"
    assert "[rag:" in finding["summary"] and "[D1" not in finding["summary"]
    assert finding["source_ids"] == list(out["sources"])  # every cited source is returned
    for source in out["sources"].values():
        assert "_2024.html#item-" in source["locator"]  # FY2024 filings only
        assert source["kind"] == "rag"
    best = out["sources"][finding["source_ids"][0]]  # D1, the top-ranked passage
    assert best["locator"].startswith("KRO_2024.html#item-1a/c")  # Item + chunk span
    assert best["title"] == "Kestrel Ridge Outdoor Co. (KRO) 10-K FY 2024, Item 1A. Risk Factors"

    # The year filter reached retrieval: the answer prompt saw FY2024 excerpts only.
    answer_prompt = _prompt_text(llm.history[1])
    assert "KRO_2024.html#item-1a/c" in answer_prompt
    assert "KRO_2022" not in answer_prompt and "KRO_2023" not in answer_prompt


def test_plan_keeps_original_wording_and_drops_unknown_years(retriever):
    agent, _ = _agent(retriever, {"queries": ["revenue outlook"], "years": [2031, 2024]})
    queries, years = agent.plan(QUESTION, "Outlook for 2025")
    assert queries == ["revenue outlook", "Outlook for 2025"]
    assert years == [2024]  # 2031 isn't in the index, so it's ignored


def test_falls_back_to_raw_subtask_when_planning_fails(retriever):
    # The first scripted response isn't a dict, so structured output fails.
    agent, llm = _agent(retriever, "garbled", "Net revenue rose [D1].")
    out = agent(_task("net revenue by region"))
    assert out["findings"][0]["source_ids"]  # still retrieved and cited
    assert len(llm.history) == 2


def test_not_found_answer_returns_no_sources(retriever):
    agent, _ = _agent(
        retriever,
        {"queries": ["employee stock options"], "years": []},
        "NOT FOUND: the excerpts do not discuss employee stock options.",
    )
    out = agent(_task("Details of the employee stock option plan"))
    assert out["findings"][0]["summary"].startswith("NOT FOUND")
    assert out["sources"] == {} and out["findings"][0]["source_ids"] == []


def test_answer_with_only_hallucinated_citations_returns_no_sources(retriever):
    agent, _ = _agent(retriever, {"queries": ["revenue"], "years": []}, "Revenue grew [D42].")
    out = agent(_task("revenue"))
    assert out["sources"] == {}
    assert out["findings"][0]["summary"] == "Revenue grew."


class ExplodingAnswerLLM(FakeChatModel):
    """Plans fine (structured output) but every plain call fails, like a 503."""

    def _generate(self, *args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")


def test_llm_failure_becomes_an_error_finding_instead_of_crashing(retriever):
    llm = ExplodingAnswerLLM(responses=[{"queries": ["revenue"], "years": []}])
    agent = RagAgent(retriever=retriever, llm=llm, settings=_settings())
    out = agent(_task("revenue"))
    assert out["findings"][0]["summary"].startswith("ERROR:")
    assert out["sources"] == {}


def test_rag_agent_runs_inside_the_graph(retriever):
    agent, _ = _agent(
        retriever,
        {"queries": ["net revenue"], "years": [2024]},
        "Net revenue reached a record $466.7 million [D1].",
    )
    graph = build_graph(
        {
            RAG: agent,
            "data_analysis": placeholders.data_analysis,
            "web_research": placeholders.web_research,
            "supervisor": placeholders.supervisor,
            "writer": placeholders.writer,
            "critic": placeholders.critic,
        },
        settings=_settings(),
    )
    final = graph.invoke({"question": QUESTION})

    rag_findings = [f for f in final["findings"] if f["agent"] == "rag"]
    assert len(rag_findings) == 1
    assert any(sid.startswith("rag:") for sid in final["sources"])  # merged by the reducer
    assert {f["agent"] for f in final["findings"]} == {"web", "rag", "data"}
    assert final["final_report"]
