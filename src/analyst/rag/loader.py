"""PDF -> clean text, one record per page.

Why per page: the RAG agent cites "annual_report_2024.pdf, p. 4". Keeping
page boundaries all the way to the chunks makes citations exact for free.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    doc: str  # file name, e.g. "annual_report_2024.pdf"
    page: int  # 1-based, the way people cite pages
    text: str
    title: str  # document title, e.g. "Kestrel Ridge Outdoor Co. Annual Report 2024"
    year: int | None = None  # report year, used as a metadata filter at query time


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(path)
    raw = [p.extract_text() or "" for p in reader.pages]
    boilerplate = _repeated_lines(raw)
    # Prefer the title stored in the PDF's metadata; fall back to the file name.
    title = (reader.metadata.title if reader.metadata else None) or path.stem.replace("_", " ")
    year = _year(title) or _year(path.stem)
    return [
        Page(doc=path.name, page=i, text=_clean(text, boilerplate), title=title, year=year)
        for i, text in enumerate(raw, start=1)
        if text.strip()
    ]


def load_dir(docs_dir: Path) -> list[Page]:
    # Sorted, so the index (and its chunk ids) is identical on every machine.
    pages: list[Page] = []
    for path in sorted(docs_dir.glob("*.pdf")):
        pages.extend(load_pdf(path))
    return pages


def _year(text: str) -> int | None:
    """The first 4-digit year in a title or file name, e.g. "Annual Report 2024" -> 2024.

    Why store it as metadata: "key risks in 2024" should search *only* the 2024
    report. Filtering on a field is exact. Hoping the similarity search ranks
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
