"""Offline tests for evals/compare.py: baseline vs pipeline wiring, scoring, cache, quota stop."""

import json

import pytest

from analyst.agents import placeholders
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER
from analyst.llm import FakeChatModel
from analyst.rag.retriever import Retriever
from analyst.rag.vector_store import VectorStore
from analyst.state import make_source_id
from evals import compare
from tests.fakes import BagOfWordsEmbedder

QUESTIONS = """\
- id: kro_revenue
  question: What was Kestrel Ridge's net revenue in fiscal 2024, and what tariff risk does it face?
  expect: ['466\\.7', 'tariff']
- id: second
  question: Second question
  expect: ['anything']
"""


def scripted_writer(state):
    sid = next(iter(state["sources"]))
    draft = f"# Report\n\n- Net revenue reached $466.7 million and tariffs loom [{sid}].\n"
    return {"draft": draft, "revision_count": 0}


NODES = {
    SUPERVISOR: placeholders.supervisor,
    WEB: placeholders.web_research,
    RAG: placeholders.rag,
    DATA: placeholders.data_analysis,
    WRITER: scripted_writer,
    CRITIC: placeholders.critic,
}

NO_CONTEXT_ANSWER = "Kestrel Ridge's revenue was roughly $466.7 million in fiscal 2024."
BASELINE_ANSWER = (
    "Net revenue reached $466.7 million in fiscal 2024 [D1]. "
    "New tariffs on Southeast Asian imports would cut gross margin [D1, D2]. "
    "Its largest supplier is in Vietnam [D42]."  # D42 does not exist: an invented citation
)


@pytest.fixture
def retriever(sample_index):
    return Retriever(VectorStore.load(sample_index), BagOfWordsEmbedder())


@pytest.fixture
def paths(tmp_path):
    q = tmp_path / "questions.yaml"
    q.write_text(QUESTIONS, encoding="utf-8")
    return q, tmp_path / "cache.json", tmp_path / "out"


def _run(paths, retriever, llm, *extra, nodes=NODES):
    q, cache, out = paths
    args = ["--questions", str(q), "--cache", str(cache), "--out-dir", str(out), *extra]
    return compare.main(args, nodes=nodes, retriever=retriever, llm=llm)


def _result(out_dir):
    (path,) = out_dir.glob("compare-*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_methods_are_scored_with_the_eval_scorer(paths, retriever):
    llm = FakeChatModel(responses=[NO_CONTEXT_ANSWER, BASELINE_ANSWER])
    code = _run(paths, retriever, llm, "--only", "kro_revenue",
                "--methods", "no_context", "baseline", "pipeline")  # fmt: skip
    assert code == 0
    cells = {c["method"]: c for c in _result(paths[2])["cells"]}

    base = cells["baseline"]
    assert base["answer_hit"] is True
    assert base["llm_calls"] == 1  # one call, no loop
    # The invented [D42] is kept visible to the scorer (as an id no source has).
    assert base["no_invented_citations"] is False
    assert base["invented_source_ids"] == [make_source_id("rag", "invented:D42")]
    assert base["sources_collected"] >= 2

    # The baseline's prompt held real retrieved passages, labelled like the RAG agent's.
    prompt = "\n".join(m.content for m in llm.history[1])
    assert "[D1] " in prompt and "KRO_2024.html#item-1a/c" in prompt and "Southeast Asia" in prompt

    nc = cells["no_context"]
    assert nc["answer_hit"] is False  # says revenue, but nothing about tariffs
    assert nc["citation_coverage"] == 0.0 and nc["sources_collected"] == 0

    pipe = cells["pipeline"]
    assert pipe["answer_hit"] is True and pipe["no_invented_citations"] is True
    assert pipe["llm_calls"] == 0  # placeholder agents make no LLM call

    agg = _result(paths[2])["aggregate"]
    assert agg["baseline"]["invented_citation_rate"] == 1.0
    assert agg["pipeline"]["invented_citation_rate"] == 0.0


def test_markdown_table_has_aggregate_and_per_question_rows(paths, retriever):
    _run(paths, retriever, FakeChatModel(responses=[BASELINE_ANSWER]), "--only", "kro_revenue")
    (md,) = paths[2].glob("compare-*.md")
    text = md.read_text(encoding="utf-8")
    assert "| Method | Answer hit | Citation coverage | Invented-citation rate" in text
    assert "| baseline | 100% |" in text and "| pipeline | 100% |" in text
    assert "| kro_revenue | ✅ ·" in text


def test_rerun_uses_the_cache_and_makes_no_calls(paths, retriever):
    llm = FakeChatModel(responses=[BASELINE_ANSWER])
    _run(paths, retriever, llm, "--only", "kro_revenue")
    calls = len(llm.history)
    assert calls == 1

    for f in paths[2].glob("*"):
        f.unlink()
    assert _run(paths, retriever, llm, "--only", "kro_revenue") == 0
    assert len(llm.history) == calls  # nothing re-run
    cells = _result(paths[2])["cells"]
    assert all(c["cached"] for c in cells) and cells[0]["answer_hit"] is True

    cache = json.loads(paths[1].read_text(encoding="utf-8"))
    assert set(cache) == {"kro_revenue::baseline", "kro_revenue::pipeline"}


class QuotaLLM(FakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")


def test_quota_error_stops_the_run_and_is_not_cached(paths, retriever, capsys):
    code = _run(paths, retriever, QuotaLLM(), "--limit", "2", "--methods", "baseline")
    assert code == 2
    assert "STOPPED: quota exhausted at kro_revenue / baseline" in capsys.readouterr().out
    cells = _result(paths[2])["cells"]
    assert [c["id"] for c in cells] == ["kro_revenue"]  # the second question never ran
    assert cells[0]["error"].startswith("RuntimeError: 429")
    assert not paths[1].exists() or json.loads(paths[1].read_text()) == {}


def test_degraded_pipeline_run_is_reported_but_retried_next_time(paths, retriever):
    def failing_rag(inp):
        out = placeholders.rag(inp)
        out["findings"][0]["summary"] = "ERROR: document answer failed (429)."
        return out

    nodes = {**NODES, RAG: failing_rag}
    _run(paths, retriever, None, "--only", "kro_revenue", "--methods", "pipeline", nodes=nodes)
    (cell,) = _result(paths[2])["cells"]
    assert cell["degraded"] is True
    assert not paths[1].exists() or "kro_revenue::pipeline" not in json.loads(paths[1].read_text())


def test_default_is_a_small_subset():
    assert compare.DEFAULT_LIMIT <= 3
    assert compare.DEFAULT_METHODS == ("baseline", "pipeline")


def test_fallback_to_the_fast_model_is_flagged_for_single_call_methods():
    def cell(method, calls, models):
        return {"method": method, "llm_calls": calls, "models": models}

    strong = "gemini-3.8-flash"
    assert not compare.fell_back(cell("baseline", 1, {strong: 1}), strong)
    assert compare.fell_back(cell("baseline", 2, {strong: 1, "flash-lite": 1}), strong)  # 429 retry
    assert compare.fell_back(cell("no_context", 1, {"flash-lite": 1}), strong)
    # The pipeline uses the fast tier for light steps by design: not a fallback.
    assert not compare.fell_back(cell("pipeline", 9, {strong: 6, "flash-lite": 3}), strong)
