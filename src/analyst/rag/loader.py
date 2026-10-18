"""Documents -> clean text sections, each with a citable address.

Two formats:
- SEC 10-K filings (.htm/.html, fetched by scripts/fetch_edgar.py): one
  section per Item ("Item 1A. Risk Factors"), see edgar.py. Company, ticker
  and fiscal year come from the .json sidecar written next to each filing.
- PDFs (your own documents): one section per page.

Why sections carry their address all the way to the chunks: the RAG agent
cites "Apple Inc. (AAPL) 10-K FY2024, Item 1A. Risk Factors". Keeping the
boundary with every chunk makes that citation exact for free.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from analyst.rag.edgar import html_to_pages, split_items

# The narrative Items indexed by default. Item 8 (financial statements) is
# left out: it is mostly tables, which flatten into number soup that embeds
# poorly, and the data agent already has those figures exactly, from XBRL.
# Leaving it out also roughly halves the index (and the embedding time).
# Override with EDGAR_ITEMS, e.g. "1,1A,3,7,7A,8".
DEFAULT_ITEMS = ("1", "1A", "3", "7", "7A")
DOC_PATTERNS = ("*.pdf", "*.htm", "*.html")


@dataclass(frozen=True)
class Section:
    doc: str  # file name, e.g. "AAPL_2024.html"
    anchor: str  # short, stable key within the doc: "item-1a", or "p4" for a PDF page
    label: str  # how people cite it: "Item 1A. Risk Factors", "page 4"
    text: str
    title: str  # document title, e.g. "Apple Inc. (AAPL) 10-K FY2024"
    year: int | None = None  # fiscal / report year, used as a metadata filter at query time
    ticker: str | None = None


def load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(path)
    raw = [p.extract_text() or "" for p in reader.pages]
    boilerplate = _repeated_lines(raw)
    # Prefer the title stored in the PDF's metadata; fall back to the file name.
    title = (reader.metadata.title if reader.metadata else None) or path.stem.replace("_", " ")
    year = _year(title) or _year(path.stem)
    return [
        Section(
            doc=path.name,
            anchor=f"p{i}",
            label=f"page {i}",
            text=_clean(text, boilerplate),
            title=title,
            year=year,
        )
        for i, text in enumerate(raw, start=1)
        if text.strip()
    ]


def load_filing(path: Path, items: Iterable[str] = DEFAULT_ITEMS) -> list[Section]:
    """One section per selected Item of a 10-K."""
    meta = _sidecar(path)
    ticker = meta.get("ticker") or path.stem.split("_")[0].upper()
    year = meta.get("fiscal_year") or _year(path.stem)
    company = meta.get("company") or ticker
    # "FY 2024" with a space: BM25 tokenizes "FY2024" as one word, which a
    # query for "risks in 2024" would never match.
    title = f"{company} ({ticker}) {meta.get('form', '10-K')} FY {year}"

    pages = html_to_pages(path.read_text(encoding="utf-8", errors="replace"))
    # Running headers and page numbers: lines repeated on at least half the
    # printed pages. A lower bar than for PDFs, because a 10-K's cover and
    # exhibit pages often lack the header and dilute the count.
    boilerplate = _repeated_lines(pages, threshold=0.5)
    text = "\n".join(
        ln for page in pages for ln in page.splitlines() if _signature(ln) not in boilerplate
    )
    wanted = {i.upper() for i in items}
    return [
        Section(
            doc=path.name,
            anchor=f"item-{s.item.lower()}",
            label=s.label,
            text=s.text,
            title=title,
            year=int(year) if year else None,
            ticker=ticker,
        )
        for s in split_items(text)
        if s.item in wanted
    ]


def load_dir(docs_dir: Path, items: Iterable[str] = DEFAULT_ITEMS) -> list[Section]:
    # Sorted, so the index (and its chunk ids) is identical on every machine.
    sections: list[Section] = []
    for path in doc_files(docs_dir):
        if path.suffix.lower() == ".pdf":
            sections.extend(load_pdf(path))
        else:
            sections.extend(load_filing(path, items))
    return sections


def doc_files(docs_dir: Path) -> list[Path]:
    return sorted({p for pattern in DOC_PATTERNS for p in docs_dir.glob(pattern)})


def _sidecar(path: Path) -> dict:
    """The filing's metadata (fetch_edgar writes AAPL_2024.json next to AAPL_2024.html).

    Why a sidecar rather than parsing the cover page: the cover's layout
    varies by filer, while the manifest knows the company, ticker and fiscal
    year for certain. Without one, the file name is the fallback.
    """
    meta = path.with_suffix(".json")
    if not meta.exists():
        return {}
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _year(text: str) -> int | None:
    """The first 4-digit year in a title or file name, e.g. "Annual Report 2024" -> 2024.

    Why store it as metadata: "key risks in 2024" should search *only* the 2024
    documents. Filtering on a field is exact. Hoping the similarity search ranks
    the right year first is not.
    """
    match = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", text)
    return int(match.group()) if match else None


def _signature(line: str) -> str:
    # "page 3" and "page 4" should count as the same footer line.
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_lines(pages: list[str], threshold: float = 0.6) -> set[str]:
    """Find headers and footers: lines that appear on most pages of a document.

    Why remove them: a footer such as "Company - Annual Report - page 3"
    would be embedded into every chunk. All chunks would then look alike to
    the similarity search, and it would cost tokens in every prompt.
    """
    if len(pages) < 2:
        return set()
    counts = Counter(sig for text in pages for sig in {_signature(ln) for ln in text.splitlines()})
    return {sig for sig, n in counts.items() if sig and n / len(pages) >= threshold}


def _clean(text: str, boilerplate: set[str]) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and _signature(ln) not in boilerplate]

    # Undo visual line wrapping. PDF text has a hard break at the end of each
    # rendered line, even mid-sentence. If a line doesn't end a sentence and
    # the next line continues it (starts lowercase, a digit or a bracket), join
    # them with a space. Otherwise keep the newline: it is probably a heading
    # or paragraph boundary, which the chunker prefers to split on.
    out: list[str] = []
    for ln in lines:
        if out and not out[-1].endswith((".", ":", "!", "?")) and re.match(r"[a-z0-9(]", ln):
            if out[-1].endswith("-") and ln[0].islower():
                out[-1] = out[-1][:-1] + ln  # "manufac-" + "turer" -> "manufacturer"
            else:
                out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    return re.sub(r"[ \t]+", " ", "\n".join(out)).strip()
