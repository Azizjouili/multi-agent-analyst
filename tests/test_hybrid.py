"""Tests for BM25, reciprocal rank fusion, metadata filters and the hybrid retriever."""

import json

import numpy as np
import pytest

from analyst.rag.bm25 import BM25, tokenize
from analyst.rag.retriever import Retriever, reciprocal_rank_fusion
from analyst.rag.vector_store import VectorIndexError, VectorStore, matches
from tests.fakes import BagOfWordsEmbedder

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_stems_plurals_and_drops_stopwords():
    assert tokenize("Key Risks in 2024!") == ["key", "risk", "2024"]
    assert tokenize("categories business analysis") == ["category", "business", "analysis"]


def test_bm25_scores_only_documents_sharing_terms():
    bm = BM25(["tariff risk", "revenue growth", "tariff exposure from tariffs"])
    s = bm.scores("tariffs")
    assert s[1] == 0
    assert s[0] > 0 and s[2] > 0


def test_bm25_rare_terms_weigh_more():
    docs = ["revenue 2024", "revenue 2023", "revenue 2022", "revenue recall"]
    bm = BM25(docs)
    assert bm.idf["recall"] > bm.idf["revenue"]  # in 1 doc vs in all 4
    s = bm.scores("revenue recall")
    assert int(np.argmax(s)) == 3


def test_bm25_length_normalization_prefers_focused_chunks():
    bm = BM25(["tariff", "tariff " + "filler words " * 30])
    s = bm.scores("tariff")
    assert s[0] > s[1]  # same term count, but the short chunk is more about it


def test_bm25_tf_saturates():
    bm = BM25(["tariff", "tariff tariff tariff tariff tariff tariff", "other"])
    s = bm.scores("tariff")
    assert s[1] < 3 * s[0]  # six mentions are not worth six times one mention


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------
def test_rrf_rewards_agreement_over_a_single_first_place():
    # Doc 7 is first in one ranking only. Doc 2 is second in both.
    fused = reciprocal_rank_fusion([[7, 2, 3], [9, 2, 3]])
    assert fused[2] > fused[7]
    assert fused[2] == pytest.approx(2 / 62)


def test_rrf_ignores_score_scale():
    # Only ranks are used, so wildly different raw scores can't dominate.
    assert reciprocal_rank_fusion([[1, 2]]) == reciprocal_rank_fusion([[1, 2]], k=60)
    assert reciprocal_rank_fusion([[1, 2]])[1] == pytest.approx(1 / 61)


# ---------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------
def test_matches_equality_membership_and_missing_keys():
    rec = {"year": 2024, "doc": "a.html"}
    assert matches(rec, None)
    assert matches(rec, {"year": 2024})
    assert matches(rec, {"year": [2023, 2024]})
    assert not matches(rec, {"year": 2023})
    assert not matches({"doc": "a.html"}, {"year": 2024})  # missing field never matches


def test_vector_search_filters_before_top_k():
    store = VectorStore(dim=2)
    store.add(
        np.array([[1.0, 0.0], [0.6, 0.8]]), [{"id": "a", "year": 2023}, {"id": "b", "year": 2024}]
    )
    # "a" is the better match, but only "b" passes the filter. Filtering
    # after taking the top-1 would return nothing.
    hits = store.search(np.array([1.0, 0.0]), k=1, filters={"year": 2024})
    assert [h.record["id"] for h in hits] == ["b"]
    assert store.search(np.array([1.0, 0.0]), filters={"year": 1999}) == []


def test_load_rejects_outdated_index_format(tmp_path):
    store = VectorStore(dim=2, model_name="m")
    store.add(np.eye(2), [{"id": "a"}, {"id": "b"}])
    store.save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["version"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(VectorIndexError, match="Rebuild"):
        VectorStore.load(tmp_path)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
class FixedQueryEmbedder:
    """Returns a preset vector for every query, so we control the vector ranking."""

    model_name = "fixed"

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_query(self, _text):
        return self.vector


def _record(i, text):
    return {
        "id": f"c{i}", "doc": "d.html", "anchor": f"item-{i}", "section": f"Item {i}",
        "index": 0, "text": text, "title": "",
    }  # fmt: skip


def test_hybrid_prefers_documents_both_methods_agree_on():
    # Vector ranking (query = [1, 0]): A > C > B
    # Keyword ranking ("tariff"):      B > C   (A has no keyword match)
    # A is first for vectors only. C is second in both, so it wins under RRF.
    store = VectorStore(dim=2)
    store.add(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]]),
        [
            _record(0, "unrelated words"),
            _record(1, "tariff tariff"),
            _record(2, "tariff and more words here"),
        ],
    )
    r = Retriever(store, FixedQueryEmbedder([1.0, 0.0]))
    assert r.retrieve("tariff", k=1, mode="vector")[0].record["id"] == "c0"
    assert r.retrieve("tariff", k=1, mode="keyword")[0].record["id"] == "c1"
    hybrid = [h.record["id"] for h in r.retrieve("tariff", k=3, mode="hybrid")]
    assert hybrid[0] in {"c1", "c2"} and hybrid[-1] == "c0"
    # c2 (in both rankings) outranks c0 (top of only one).
    assert hybrid.index("c2") < hybrid.index("c0")


@pytest.fixture(scope="module")
def retriever(sample_index):
    return Retriever(VectorStore.load(sample_index), BagOfWordsEmbedder())


def test_retriever_lists_years_in_index(retriever):
    assert retriever.years == [2022, 2023, 2024]


def test_context_header_makes_item_and_company_searchable(retriever):
    # The risk-factor chunks never say "risk factors" or "Kestrel": only their
    # context header ("Kestrel Ridge Outdoor Co. (KRO) 10-K FY 2024, Item 1A.
    # Risk Factors") ties them to the Item and the company.
    hits = retriever.retrieve("risk factors", k=5, mode="keyword")
    assert hits and all(h.record["anchor"] == "item-1a" for h in hits)
    # Every chunk of a filing carries its company's name, so a name finds them all.
    hits = retriever.retrieve("Bluefin", k=10, mode="keyword")
    assert {h.record["doc"] for h in hits} == {"BFA_2024.html"} and len(hits) == 5


def test_ticker_filter_restricts_every_hit(retriever):
    hits = retriever.retrieve("risk factors", k=5, filters={"ticker": "BFA"})
    assert hits and {h.record["doc"] for h in hits} == {"BFA_2024.html"}


def test_year_filter_restricts_every_hit(retriever):
    hits = retriever.retrieve("risk factors", k=5, filters={"year": 2023})
    assert hits and {h.record["year"] for h in hits} == {2023}


def test_multiple_queries_are_fused(retriever):
    hits = retriever.retrieve(["tariff policy", "factory in Mexico"], k=1)
    assert hits[0].record["doc"] == "KRO_2024.html"
    assert hits[0].record["anchor"] == "item-1a"


def test_filter_matching_nothing_returns_empty(retriever):
    assert retriever.retrieve("revenue", filters={"year": 1999}) == []


# ---------------------------------------------------------------------------
# Real model (opt-in: uv run pytest -m model)
# ---------------------------------------------------------------------------
@pytest.mark.model
def test_real_model_hybrid_beats_vector_on_year_query(sample_data, tmp_path):
    from analyst.rag.embedder import Embedder
    from analyst.rag.index import build_index

    out, _ = sample_data
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    build_index(out / "docs", tmp_path, embedder, chunk_size=800, chunk_overlap=120)
    r = Retriever(VectorStore.load(tmp_path), embedder)

    top = r.retrieve("Kestrel risk factors 2024", k=1, mode="hybrid", filters={"year": 2024})
    top = top[0].record
    assert (top["doc"], top["anchor"]) == ("KRO_2024.html", "item-1a")
