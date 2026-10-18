"""Citation handling shared by every agent that writes cited text.

Agents show the LLM short local labels ([D1] for a document page, [Q1] for a
SQL result, [S1] for a source in the writer's prompt) and map them back to
stable source ids afterwards. Why not let the model cite the ids directly:
copying "rag:3f9a1c2b" invites typos, and with short labels an invented
citation ("[D9]" when only D1-D4 exist) is trivial to detect and drop.

The stored text (findings, draft) always uses the stable ids. Only finalize
turns them into reader-facing numbers [1]..[n].
"""

from __future__ import annotations

import re
from collections.abc import Callable

_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
_REMOVED = "\x00"  # marks where a citation group was deleted
LABEL = re.compile(r"[A-Z]\d+")  # D1, Q2, S3, ...
SOURCE_ID = re.compile(r"(?:web|rag|data):[0-9a-f]{8}")  # see state.make_source_id


def _default_join(items: list[str]) -> str:
    return f"[{', '.join(items)}]"


def rewrite_citations(
    text: str,
    mapping: dict[str, str],
    part: re.Pattern = LABEL,
    join: Callable[[list[str]], str] = _default_join,
) -> tuple[str, list[str]]:
    """Rewrite every citation group, e.g. [D1, D3] -> [rag:ab12cd34, rag:9f8e7d6c].

    A bracket is a citation group only if every comma/semicolon-separated part
    matches `part`. Brackets such as "[sic]" or a Markdown link's "[text]" are
    left alone. Parts missing from `mapping` (invented citations) are removed,
    and a group left empty disappears, so a made-up citation can't survive.
    Returns the new text and the mapped values in order of first appearance.
    """
    used: list[str] = []

    def replace(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,;]", match.group(1))]
        if not all(part.fullmatch(p) for p in parts):
            return match.group(0)
        values = list(dict.fromkeys(mapping[p] for p in parts if p in mapping))
        for value in values:
            if value not in used:
                used.append(value)
        return join(values) if values else _REMOVED

    rewritten = _BRACKETS.sub(replace, text)
    # Remove each deleted group together with the space before it ("fell [D9]."
    # -> "fell."). Only those spots are touched: collapsing whitespace globally
    # would break Markdown, where indentation nests lists.
    rewritten = re.sub(rf"[ \t]*{_REMOVED}", "", rewritten)
    return rewritten.strip(), used


def resolve_citations(text: str, labels: dict[str, str]) -> tuple[str, list[str]]:
    """Local labels -> stable source ids: [D1, D3] -> [rag:ab12cd34, rag:9f8e7d6c]."""
    return rewrite_citations(text, labels, part=LABEL)


def relabel_source_ids(text: str, id_to_label: dict[str, str]) -> str:
    """Stable source ids -> local labels, for showing stored text to an LLM.

    The inverse of resolve_citations. Ids missing from the mapping are dropped.
    """
    return rewrite_citations(text, id_to_label, part=SOURCE_ID)[0]
