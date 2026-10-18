"""Tests for the from-scratch RAG pipeline and the sample-data generator."""

import csv
import re

import numpy as np
import pytest

from analyst.rag.chunker import chunk_pages, split_text
from analyst.rag.index import build_index
from analyst.rag.loader import Page, load_dir, load_pdf
from analyst.rag.retriever import Retriever
from analyst.rag.vector_store import VectorIndexError, VectorStore
from scripts import make_sample_data as gen
from tests.fakes import BagOfWordsEmbedder

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
LONG_TEXT = "\n".join(
    f"Paragraph {i}. " + " ".join(f"Sentence {i}.{j} has some words in it." for j in range(6))
    for i in range(12)
)


def test_short_text_is_one_chunk():
    assert split_text("A short page.", size=100, overlap=10) == ["A short page."]


def test_chunks_respect_size():
    chunks = split_text(LONG_TEXT, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(LONG_TEXT, size=300, overlap=80)
    for a, b in zip(chunks, chunks[1:], strict=False):
        # The start of each chunk repeats text from the end of the previous one.
        assert b[:20] in a


def test_overlap_with_long_single_sentence_lines():
    # Shaped like the real PDFs: each line is one long sentence ending in "\n".
    # This used to produce zero overlap, because the trailing "\n" was taken
    # as the boundary.
    text = "\n".join(
        f"Line {i} describes a risk factor in considerable detail, with several clauses, "
        f"qualifications and numbers such as {i * 7}% so that it runs long."
        for i in range(10)
    )
    chunks = split_text(text, size=400, overlap=120)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b[:30] in a, (a[-80:], b[:80])
    assert all(len(c) <= 400 for c in chunks)


def test_no_text_is_lost():
    chunks = split_text(LONG_TEXT, size=300, overlap=50)
    for sentence in re.findall(r"Sentence \d+\.\d+", LONG_TEXT):
        assert any(sentence in c for c in chunks), sentence


def test_prefers_sentence_boundaries():
    chunks = split_text(LONG_TEXT, size=300, overlap=0)
    # With natural boundaries available, chunks end on a full sentence, not mid-word.
    assert all(c.endswith(".") for c in chunks)


def test_text_without_separators_is_hard_split():
    chunks = split_text("x" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text(LONG_TEXT, size=100, overlap=100)


def test_chunk_ids_pages_and_context_header():
    pages = [Page("r.pdf", 1, "Alpha.", "Report 2024"), Page("r.pdf", 2, LONG_TEXT, "Report 2024")]
    chunks = chunk_pages(pages, size=300, overlap=50)
    assert chunks[0].id == "r.pdf#p1-c0"
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)  # unique
    assert chunks[0].embedding_text == "Report 2024, page 1\nAlpha."
    assert "Report 2024" not in chunks[0].text  # the header is embedded, not stored


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def _store():
    store = VectorStore(dim=3, model_name="m")
    store.add(np.eye(3), [{"id": "x"}, {"id": "y"}, {"id": "z"}])
    return store


def test_search_returns_nearest_first():
    hits = _store().search(np.array([0.9, 0.1, 0.0]), k=2)
    assert [h.record["id"] for h in hits] == ["x", "y"]
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(0.9 / np.sqrt(0.82))  # cosine similarity


def test_search_k_larger_than_store():
    assert len(_store().search(np.array([1.0, 0, 0]), k=10)) == 3


def test_add_normalizes_vectors():
    store = VectorStore(dim=2)
    store.add(np.array([[10.0, 0.0]]), [{"id": "a"}])
    assert store.search(np.array([1.0, 0.0]))[0].score == pytest.approx(1.0)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError):
        _store().add(np.ones((1, 4)), [{"id": "bad"}])
    with pytest.raises(ValueError):
        _store().search(np.ones(4))


def test_save_load_roundtrip(tmp_path):
    _store().save(tmp_path, extra={"chunk_size": 800})
    loaded = VectorStore.load(tmp_path, expected_model="m")
    assert len(loaded) == 3
    assert loaded.manifest["chunk_size"] == 800
    assert loaded.search(np.array([0, 0, 1.0]), k=1)[0].record["id"] == "z"


def test_load_rejects_a_different_model(tmp_path):
    _store().save(tmp_path)
    with pytest.raises(VectorIndexError, match="Rebuild"):
        VectorStore.load(tmp_path, expected_model="another-model")


def test_load_missing_index_explains_how_to_build(tmp_path):
    with pytest.raises(VectorIndexError, match="analyst.rag.index"):
        VectorStore.load(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Sample data + loader + full pipeline
# ---------------------------------------------------------------------------
def test_csv_shape(sample_data):
    out, _ = sample_data
    with (out / "datasets" / "quarterly_sales.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 4 * 3 * 4 * 2  # years x quarters x regions x products x channels
    assert {r["period"] for r in rows} >= {"2022-Q1", "2024-Q4"}


def test_pdf_figures_match_the_csv(sample_data):
    # The documents and the dataset must never contradict each other.
    out, summaries = sample_data
    for year, s in summaries.items():
        text = " ".join(p.text for p in load_pdf(out / "docs" / f"annual_report_{year}.pdf"))
        assert gen.money(s.revenue) in text
        assert gen.pct(s.gross_margin) in text
        for region in gen.REGIONS:
            assert gen.money(s.by_region[region]) in text


def test_loader_strips_footer_and_unwraps_lines(sample_data):
    out, _ = sample_data
    pages = load_pdf(out / "docs" / "annual_report_2024.pdf")
    assert [p.page for p in pages] == [1, 2, 3, 4, 5]
    assert all("- page" not in p.text for p in pages)  # footer removed
    assert pages[0].title == "Kestrel Ridge Outdoor Co. Annual Report 2024"
    # A sentence that the PDF wrapped across lines is back on one line.
    assert "Growth came from direct-to-consumer" in pages[0].text


def test_build_index_and_retrieve_end_to_end(sample_data, tmp_path):
    out, _ = sample_data
    embedder = BagOfWordsEmbedder()
    stats = build_index(out / "docs", tmp_path, embedder, chunk_size=800, chunk_overlap=120)
    assert stats["documents"] == 3
    assert stats["chunks"] >= stats["pages"] == 15

    store = VectorStore.load(tmp_path, expected_model=embedder.model_name)
    assert set(store.manifest["files"]) == {f"annual_report_{y}.pdf" for y in gen.YEARS}

    hits = Retriever(store, embedder).retrieve("tariffs Southeast Asia Mexico factory", k=1)
    assert hits[0].record["doc"] == "annual_report_2024.pdf"
    assert hits[0].record["page"] == 4  # the Risk Factors page


def test_load_dir_is_sorted(sample_data):
    out, _ = sample_data
    docs = [p.doc for p in load_dir(out / "docs")]
    assert docs == sorted(docs)


# ---------------------------------------------------------------------------
# Real model (opt-in: uv run pytest -m model)
# ---------------------------------------------------------------------------
@pytest.mark.model
def test_real_embedder_shape_and_semantics():
    from analyst.rag.embedder import Embedder

    e = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    docs = e.embed_documents(["Revenue grew strongly this year.", "The tent is green."])
    assert docs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    q = e.embed_query("Sales increased")
    # Similar meaning wins even with no shared words.
    assert q @ docs[0] > q @ docs[1]
