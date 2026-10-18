"""Recursive character splitter with overlap, written from scratch.

The idea (the same as LangChain's RecursiveCharacterTextSplitter):
1. **Split** the text on the "largest" natural boundary first (paragraphs),
   and only use smaller ones (lines, sentences, words, then characters) for
   pieces that are still too big. Chunks therefore break at meaningful places
   whenever possible.
2. **Merge** the small pieces back together, greedily, into chunks of up to
   `size` characters, starting each new chunk with the last ~`overlap`
   characters of the previous one, cut at a sentence or word boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from analyst.rag.loader import Page

# Largest boundary first. "" means "cut anywhere", the last resort for text
# with no spaces at all (e.g. a long URL).
SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def contextual_text(title: str, page: int, text: str) -> str:
    """The text that gets *searched*: a context header plus the chunk.

    Why: a chunk of a "Risk Factors" page may not say which company or year it
    belongs to, because that was only in the footer or title. Without the
    header, the risk pages from every year look nearly identical, and "risks in
    2024" can't find the 2024 one. Both the embedding and the BM25 index use
    this text, while the stored `text` stays clean for prompts and citations.
    """
    return f"{title}, page {page}\n{text}" if title else text


@dataclass(frozen=True)
class Chunk:
    id: str  # "annual_report_2024.pdf#p4-c1": unique and stable across rebuilds
    doc: str
    page: int
    index: int  # position within the page
    text: str
    title: str = ""
    year: int | None = None

    @property
    def embedding_text(self) -> str:
        return contextual_text(self.title, self.page, self.text)

    def to_record(self) -> dict:
        return asdict(self)


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= size:
        # Each new chunk would start with a full chunk of old text and never advance.
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    return _merge(_split(text, SEPARATORS, size), size, overlap)


def _split(text: str, separators: tuple[str, ...], size: int) -> list[str]:
    """Break text into pieces of at most `size` chars, preferring large boundaries."""
    if len(text) <= size:
        return [text] if text else []
    sep, smaller = separators[0], separators[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator attached to its piece, so joining the pieces
        # restores the original text exactly (no lost periods or newlines).
        piece = part + sep if i < len(parts) - 1 else part
        if not piece:
            continue
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, smaller, size))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Pack pieces into chunks of up to `size` chars, carrying ~`overlap` chars forward."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            # Start the next chunk with the end of this one. The overlap is
            # cut from the chunk *text*, not taken as whole pieces: a single
            # paragraph piece is often longer than `overlap`, and dropping it
            # whole would leave no overlap at all. It is also capped by the
            # room left for the next piece, so chunks never exceed `size`.
            current = _tail(current, min(overlap, size - len(piece)))
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _tail(text: str, budget: int) -> str:
    """The last <= `budget` chars of `text`, starting at a sentence or word boundary.

    Starting mid-word ("ers of camping gear") would add noise to the next
    chunk's embedding, so we move forward to the first clean boundary.
    """
    # Look for a boundary in the text *without* its trailing separator.
    # Otherwise the final "\n" itself would match and we'd keep nothing. The
    # separator is added back so the tail and the next piece stay apart.
    stripped = text.rstrip()
    trailing = text[len(stripped) :] or " "
    budget -= len(trailing)
    if budget <= 0:
        return ""
    tail = stripped[-budget:]
    for sep in (". ", "\n", " "):
        cut = tail.find(sep)
        if cut != -1:
            return tail[cut + len(sep) :] + trailing
    return ""  # no clean boundary within the budget: skip the overlap


def chunk_pages(pages: list[Page], size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk each page separately.

    Trade-off: a sentence that crosses a page break ends up in two chunks. In
    return, every chunk has exactly one page number, so citations are always
    correct. For cited reports, that is the better deal.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{page.doc}#p{page.page}-c{i}",
                    doc=page.doc,
                    page=page.page,
                    index=i,
                    text=text,
                    title=page.title,
                    year=page.year,
                )
            )
    return chunks
