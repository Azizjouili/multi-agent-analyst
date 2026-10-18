"""SEC 10-K (inline XBRL HTML) -> clean text, segmented by Item.

A 10-K is one long HTML file with a legally fixed outline: Part I, Item 1
Business, Item 1A Risk Factors, ... Part II, Item 7 MD&A, Item 8 Financial
Statements, and so on. Segmenting by Item gives every chunk a meaningful,
checkable address ("Apple 10-K FY2024, Item 1A") and lets the index skip the
Items that are boilerplate for our purposes (exhibit lists, signatures).

Three things make the raw HTML hard to use directly:
- Inline XBRL: the machine-readable facts sit in a hidden <ix:header> block,
  and many elements are styled display:none. Their text is not part of the
  document a reader sees, so it is dropped.
- Layout tables: cells hold "$", "391,035" and ")" separately. Rows become
  one line each, cells joined by spaces, then "$ 391,035" is re-glued.
- Page furniture: page numbers and running headers ("Apple Inc. | 2024 Form
  10-K | 21") repeat on every printed page. They are found the same way as
  the PDF footers in loader.py, by counting lines across page breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Never content. ix:header holds the hidden XBRL facts and contexts.
SKIP = {"script", "style", "head", "title", "ix:header", "noscript", "template", "svg"}
BLOCK = {
    "p", "div", "section", "article", "li", "ul", "ol", "table", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dd", "dt", "center",
}  # fmt: skip
CELL = {"td", "th"}
VOID = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"}
PAGE_BREAK = "\f"
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_BREAK_STYLE = re.compile(
    r"page-break-(?:before|after)\s*:\s*always|break-(?:before|after)\s*:\s*page", re.I
)

# Canonical Item titles (Regulation S-K). Used instead of the heading text in
# the file, which varies ("RISK FACTORS", "Risk Factors.", split across cells),
# so every filing's citations read the same way.
ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
# A heading line: "Item 1A.", "ITEM 7 - MANAGEMENT'S ...", "Item 7A:" at the
# start of a short line. Cross-references inside prose ("see Item 7 of this
# report") are mid-sentence, so anchoring at the line start skips most of them.
_ITEM_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*(?:[.:\-–—]|$|\s)", re.I)
MAX_HEADING_CHARS = 160
# A bare "Item 1A" / "PART II Item 7" / "Item 9B, 9C, 10" line: a running page
# header, not text. Dropped from section bodies.
_RUNNING_HEADER = re.compile(
    r"^(?:part\s+[iv]+\s*)?items?\s+\d{1,2}[a-c]?(?:\s*,\s*\d{1,2}[a-c]?)*\.?$", re.I
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []  # (tag, hidden) for open elements
        self.hidden = 0  # how many open elements hide their content

    def handle_starttag(self, tag: str, attrs) -> None:
        style = dict(attrs).get("style") or ""
        if tag in BLOCK:
            self.parts.append("\n")
        if _BREAK_STYLE.search(style):
            self.parts.append(PAGE_BREAK)
        if tag in VOID:
            return
        hides = tag in SKIP or bool(_HIDDEN.search(style))
        self.stack.append((tag, hides))
        self.hidden += hides

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        # Pop up to the matching open tag. Inline XBRL is XHTML, so this is
        # normally exact; the loop also tolerates a stray unclosed element.
        if not any(t == tag for t, _ in self.stack):
            return
        while self.stack:
            t, hides = self.stack.pop()
            self.hidden -= hides
            if t == tag:
                break
        if tag in BLOCK:
            self.parts.append("\n")
        elif tag in CELL:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def html_to_pages(html: str) -> list[str]:
    """Visible text of the document, split at its printed page breaks."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ").replace("​", "")
    pages = []
    for page in text.split(PAGE_BREAK):
        lines = [_tidy(ln) for ln in page.splitlines()]
        pages.append("\n".join(ln for ln in lines if ln))
    return [p for p in pages if p]


def _tidy(line: str) -> str:
    line = re.sub(r"[ \t\r]+", " ", line).strip()
    line = re.sub(r"\$ (?=[\d(])", "$", line)  # "$ 391,035" -> "$391,035"
    line = re.sub(r"\( (?=\d)", "(", line)  # "( 1,234 )" -> "(1,234)"
    line = re.sub(r"(?<=\d) (?=[)%])", "", line)
    return line


@dataclass(frozen=True)
class ItemSection:
    item: str  # "1A"
    title: str  # "Risk Factors"
    text: str

    @property
    def label(self) -> str:
        return f"Item {self.item}. {self.title}"


def split_items(text: str) -> list[ItemSection]:
    """Segment a 10-K's text into its Items, in document order.

    Every Item heading appears at least twice: in the table of contents and at
    the start of the Item itself (and sometimes again in a cross-reference).
    For each Item we keep the occurrence followed by the most text before the
    next heading. A table-of-contents entry is followed by almost nothing, so
    the real section wins.
    """
    lines = text.split("\n")
    heads: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS:
            continue
        m = _ITEM_HEADING.match(line)
        if not m or m.group(1).upper() not in ITEM_TITLES:
            continue
        # Some filers (Microsoft) print "Item 1A" as a running header on every
        # page of the Item. A heading for the Item we're already in continues
        # it rather than starting a new one; otherwise each page would become
        # its own section.
        if heads and heads[-1][1] == m.group(1).upper():
            continue
        heads.append((i, m.group(1).upper()))

    best: dict[str, tuple[int, int]] = {}  # item -> (body length, heading index)
    for n, (i, item) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        size = sum(len(ln) + 1 for ln in lines[i + 1 : end])
        if size > best.get(item, (-1, 0))[0]:
            best[item] = (size, n)

    sections = []
    for item, (_, n) in sorted(best.items(), key=lambda kv: heads[kv[1][1]][0]):
        start = heads[n][0]
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body = "\n".join(ln for ln in lines[start + 1 : end] if not _RUNNING_HEADER.match(ln))
        body = _drop_heading_title(body.strip(), ITEM_TITLES[item])
        if body:
            sections.append(ItemSection(item, ITEM_TITLES[item], body))
    return sections


def _drop_heading_title(body: str, title: str) -> str:
    """Remove a title line that the heading left on its own line ("Item 1A." / "Risk Factors")."""
    first, _, rest = body.partition("\n")
    if len(first) <= MAX_HEADING_CHARS and first.lower().rstrip(".").startswith(title.lower()[:12]):
        return rest.strip()
    return body
