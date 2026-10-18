"""Tests for the from-scratch RAG pipeline: 10-K/PDF loading, chunking, indexing, retrieval."""

import re

import numpy as np
import pytest

from analyst.rag.chunker import chunk_sections, split_text
from analyst.rag.edgar import html_to_pages, split_items
from analyst.rag.index import build_index
from analyst.rag.loader import Section, load_dir, load_filing, load_pdf
from analyst.rag.retriever import Retriever
from analyst.rag.vector_store import VectorIndexError, VectorStore
from tests import edgar_fixture as fx
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


def test_chunk_ids_sections_and_context_header():
    title = "Apple Inc. (AAPL) 10-K FY 2024"
    sections = [
        Section("AAPL_2024.html", "item-1", "Item 1. Business", "Alpha.", title, 2024, "AAPL"),
        Section("AAPL_2024.html", "item-1a", "Item 1A. Risk Factors", LONG_TEXT, title, 2024),
    ]
    chunks = chunk_sections(sections, size=300, overlap=50)
    assert chunks[0].id == "AAPL_2024.html#item-1-c0"
    assert chunks[1].id == "AAPL_2024.html#item-1a-c0"
    assert {c.anchor for c in chunks} == {"item-1", "item-1a"}
    assert len({c.id for c in chunks}) == len(chunks)  # unique
    assert [c.index for c in chunks if c.anchor == "item-1a"][:3] == [0, 1, 2]
    assert chunks[0].embedding_text == f"{title}, Item 1. Business\nAlpha."
    assert title not in chunks[0].text  # the header is embedded, not stored
    assert chunks[0].to_record()["ticker"] == "AAPL" and chunks[0].year == 2024


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
# 10-K HTML: text extraction and Item segmentation
# ---------------------------------------------------------------------------
def test_html_text_drops_hidden_xbrl_and_glues_table_cells():
    html = (
        '<html><body><div style="display:none"><ix:header>SECRET</ix:header></div>'
        "<p>Visible&#160;text</p><table><tr><td>Net sales</td><td>$</td><td>391,035</td>"
        "<td>(</td><td>1,234</td><td>)</td></tr></table>"
        '<hr style="page-break-after:always"/><p>Page two</p></body></html>'
    )
    assert html_to_pages(html) == ["Visible text\nNet sales $391,035 (1,234)", "Page two"]


def test_split_items_skips_table_of_contents_and_running_headers():
    text = "\n".join(
        [
            "Item 1. Business 3",  # table of contents: no body before the next entry
            "Item 1A. Risk Factors 5",
            "Item 7. Management's Discussion and Analysis 9",
            "Item 1. Business",
            "We make tents.",
            "Item 1A.",
            "Risk Factors",  # the title on a line of its own is dropped
            "Tariffs could hurt us.",
            "PART I Item 1A",  # a running page header: the same Item continues
            "Recalls could hurt us.",
            "Item 7. Management's Discussion and Analysis",
            "Revenue rose.",
        ]
    )
    items = {s.item: s for s in split_items(text)}
    assert list(items) == ["1", "1A", "7"]
    assert items["1"].text == "We make tents."
    assert items["1A"].text == "Tariffs could hurt us.\nRecalls could hurt us."
    assert items["1A"].label == "Item 1A. Risk Factors"
    assert items["7"].text == "Revenue rose."


def test_load_filing_segments_items_and_reads_the_sidecar(sample_data):
    out, _ = sample_data
    sections = load_filing(out / "docs" / "KRO_2024.html")
    assert [s.anchor for s in sections] == ["item-1", "item-1a", "item-3", "item-7", "item-7a"]
    risk = sections[1]
    assert risk.title == "Kestrel Ridge Outdoor Co. (KRO) 10-K FY 2024"
    assert (risk.year, risk.ticker, risk.label) == (2024, "KRO", "Item 1A. Risk Factors")
    # Both printed pages of Item 1A, without the footer or the running header.
    assert risk.text.startswith("Tariffs.") and risk.text.endswith("every year.")
    assert "Form 10-K |" not in risk.text and "PART I" not in risk.text
    every = " ".join(s.text for s in sections)
    assert fx.HIDDEN_MARKER not in every  # hidden inline-XBRL text never leaks
    assert fx.ITEM8_MARKER not in every  # Item 8 isn't indexed by default
    assert "Net revenue $466.7 million" in sections[3].text  # the table row, re-glued


def test_item_selection_is_configurable(sample_data):
    out, _ = sample_data
    sections = load_filing(out / "docs" / "KRO_2024.html", items=["8"])
    assert [s.label for s in sections] == ["Item 8. Financial Statements and Supplementary Data"]
    assert fx.ITEM8_MARKER in sections[0].text


def test_filing_without_sidecar_falls_back_to_the_file_name(tmp_path, sample_data):
    out, _ = sample_data
    copy = tmp_path / "ABC_2021.html"
    copy.write_text((out / "docs" / "KRO_2024.html").read_text(encoding="utf-8"), encoding="utf-8")
    section = load_filing(copy)[0]
    assert (section.ticker, section.year, section.title) == ("ABC", 2021, "ABC (ABC) 10-K FY 2021")


# ---------------------------------------------------------------------------
# PDFs (your own documents) still load page by page
# ---------------------------------------------------------------------------
def test_pdf_loader_strips_footer_and_unwraps_lines(tmp_path):
    path = tmp_path / "brochure.pdf"
    fx.write_pdf(
        path,
        [
            ["Growth came from direct-to-consumer", "sales and new stores.", "Acme - page 1"],
            ["Risks include tariffs.", "Acme - page 2"],
        ],
        title="Acme Annual Report 2024",
    )
    pages = load_pdf(path)
    assert [(p.anchor, p.label) for p in pages] == [("p1", "page 1"), ("p2", "page 2")]
    assert pages[0].title == "Acme Annual Report 2024" and pages[0].year == 2024
    assert all("Acme - page" not in p.text for p in pages)  # footer removed
    # A sentence wrapped across lines is back on one line.
    assert "Growth came from direct-to-consumer sales and new stores." in pages[0].text


# ---------------------------------------------------------------------------
# The full pipeline over the fixture corpus
# ---------------------------------------------------------------------------
def test_build_index_and_retrieve_end_to_end(sample_data, tmp_path):
    out, filings = sample_data
    embedder = BagOfWordsEmbedder()
    stats = build_index(out / "docs", tmp_path, embedder, chunk_size=800, chunk_overlap=120)
    assert stats["documents"] == len(filings) == 4
    assert stats["companies"] == 2
    assert stats["sections"] == 4 * 5  # Items 1, 1A, 3, 7, 7A of each filing
    assert stats["chunks"] > stats["sections"]  # the long Item 1A spans several chunks

    store = VectorStore.load(tmp_path, expected_model=embedder.model_name)
    assert set(store.manifest["files"]) == {f"{f.stem}.html" for f in filings}
    assert store.manifest["items"] == ["1", "1A", "3", "7", "7A"]

    hits = Retriever(store, embedder).retrieve("tariffs Southeast Asia Mexico factory", k=1)
    rec = hits[0].record
    assert (rec["doc"], rec["anchor"], rec["section"]) == (
        "KRO_2024.html",
        "item-1a",
        "Item 1A. Risk Factors",
    )


def test_load_dir_is_sorted_and_skips_sidecars(sample_data):
    out, _ = sample_data
    docs = [s.doc for s in load_dir(out / "docs")]
    assert docs == sorted(docs)
    assert all(d.endswith(".html") for d in docs)  # the .json sidecars are metadata, not docs


def test_build_index_without_documents_explains_how_to_fetch(tmp_path):
    with pytest.raises(SystemExit, match="fetch_edgar"):
        build_index(tmp_path, tmp_path / "idx", BagOfWordsEmbedder(), 800, 120)


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
