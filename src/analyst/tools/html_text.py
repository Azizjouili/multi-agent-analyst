"""HTML -> readable text, using only the standard library's HTMLParser.

A web page is mostly not the article: menus, cookie banners, scripts, footers.
Passing all of it to the LLM wastes tokens and buries the content. This
extractor:
  - drops whole subtrees that are never content (script, style, nav, footer...);
  - inserts line breaks at block elements, so paragraphs stay separate;
  - prefers the text inside <article>/<main> when the page marks it up.

Why not a library: trafilatura gives better results on messy sites, but
brings lxml and more. A small heuristic extractor handles most article pages
and shows what "main content extraction" means. Trafilatura is the upgrade if
evals show extraction quality is the bottleneck.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

SKIP = {
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "nav", "header", "footer", "aside", "form", "button", "select",
}  # fmt: skip
BLOCK = {
    "p", "div", "section", "article", "main", "li", "ul", "ol", "table", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "br", "hr", "dd", "dt",
}  # fmt: skip
MAIN = {"article", "main"}
MIN_MAIN_CHARS = 200  # below this, an <article> is probably a teaser card, not the page


@dataclass(frozen=True)
class ExtractedPage:
    title: str
    text: str


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # &amp; -> &, &#8217; -> ’
        self.all_parts: list[str] = []
        self.main_parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.main_depth = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        if tag in MAIN:
            self.main_depth += 1
        if tag in BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False
        if tag in MAIN and self.main_depth:
            self.main_depth -= 1
        if tag in BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        elif not self.skip_depth:
            self._emit(data)

    def _emit(self, text: str) -> None:
        self.all_parts.append(text)
        if self.main_depth:
            self.main_parts.append(text)


def _tidy(parts: list[str]) -> str:
    lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(parts).splitlines())
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> ExtractedPage:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    main = _tidy(parser.main_parts)
    text = main if len(main) >= MIN_MAIN_CHARS else _tidy(parser.all_parts)
    return ExtractedPage(title=_tidy(parser.title_parts), text=text)
