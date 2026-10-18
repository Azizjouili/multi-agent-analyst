"""Small helpers shared by the specialist agents."""

from __future__ import annotations

ERROR_CHARS = 200


def short_error(exc: BaseException, limit: int = ERROR_CHARS) -> str:
    """A one-line, length-capped description of an exception for an ERROR finding.

    Why cap it: provider errors can include kilobytes of JSON (a Gemini 429
    carries the whole quota report). That text flows into the writer's and
    critic's prompts, where it wastes tokens and can confuse them. The full
    error is still in the logs.
    """
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
