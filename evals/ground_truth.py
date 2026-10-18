"""Derive eval answers from the XBRL financials table, deterministically.

A quantitative question in questions.yaml names its facts instead of hard-coding
a number:

    facts:
      - {ticker: AAPL, fiscal_year: 2024, metric: revenue}                   # the value
      - {ticker: NVDA, fiscal_year: 2025, metric: revenue, compare_to: 2024} # growth, %
      - {ticker: META, fiscal_year: 2024, metric: rd_expense, per: revenue}  # share, %

Each fact becomes a regex built from the exact value in financials.csv, which
fetch_edgar extracted from the pinned 10-K's XBRL. Why derive it instead of
typing it in: a hand-copied "391.04" can be mistyped or go stale, while this
is computed from the same table the data agent queries, and the manifest pins
that table, so it can't drift.

Why a *tolerant* regex: the writer may say "$391.04 billion", "$391.0B" or
"$391,035 million". All are correct, so all must match. A wrong figure won't:
the patterns only allow the roundings of the true value.
"""

from __future__ import annotations

import csv
import math
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

Table = dict[tuple[str, int, str], tuple[float, str]]  # (ticker, fy, metric) -> (value, unit)


class GroundTruthError(ValueError):
    """A fact names a value that isn't in the financials table."""


def load_financials(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (r["ticker"], int(r["fiscal_year"]), r["metric"]): (float(r["value"]), r["unit"])
            for r in csv.DictReader(f)
        }


def _roundings(x: float, decimals: int) -> list[str]:
    """`x` rounded half-up and truncated: a writer may do either."""
    q = Decimal(1).scaleb(-decimals)
    d = Decimal(repr(abs(x)))
    return sorted({str(d.quantize(q, ROUND_HALF_UP)), str(d.quantize(q, ROUND_DOWN))})


def _number(alternatives: list[str]) -> str:
    """Alternatives as one regex group, not glued to a longer number."""
    alts = "|".join(sorted({a.replace(".", r"\.") for a in alternatives}, key=len, reverse=True))
    return rf"(?<![\d.,])(?:{alts})(?![\d])"


def money_pattern(usd: float) -> str:
    """$391,035,000,000 -> 391.0/391.03/391.04 billion, or 391,035 million."""
    alts = []
    b, m = abs(usd) / 1e9, abs(usd) / 1e6
    if b >= 1:
        decimals = (1, 2, 3) if b < 100 else (0, 1, 2, 3)
        unit = r"\s*(?:billion|bn|b\b)"
        alts.append(_number([r for d in decimals for r in _roundings(b, d)]) + unit)
    # Millions, with or without thousands separators: 391,035 / 391035 / 466.7
    millions = [r for d in (0, 1, 2) for r in _roundings(m, d)]
    grouped = [f"{int(Decimal(r)):,}" + r[len(str(int(Decimal(r)))) :] for r in millions]
    alts.append(_number(millions + grouped) + r"\s*(?:million|mn|m\b)")
    if b >= 1:
        # A comma-grouped millions figure is distinctive enough on its own,
        # e.g. in a table: "Revenue | 391,035".
        alts.append(_number([g for g in grouped if "," in g]))
    return "|".join(f"(?:{a})" for a in alts)


def per_share_pattern(value: float) -> str:
    """EPS: $6.08 (also "6.1")."""
    return _number(_roundings(value, 2) + _roundings(value, 1))


def percent_pattern(pct: float) -> str:
    """18.72 -> 18.7 / 18.72 / 19 percent (whole percents only above 10, where they're precise)."""
    alts = _roundings(pct, 1) + _roundings(pct, 2)
    if abs(pct) >= 10:
        alts += _roundings(pct, 0)
    return _number(alts) + r"\s*(?:%|percent|per cent)"


def expected_pattern(fact: dict[str, Any], table: Table) -> str:
    ticker, fy, metric = fact["ticker"], int(fact["fiscal_year"]), fact["metric"]
    value, unit = _get(table, ticker, fy, metric)
    if "compare_to" in fact:
        base, _ = _get(table, ticker, int(fact["compare_to"]), metric)
        return percent_pattern(100 * (value - base) / abs(base))
    if "per" in fact:
        whole, _ = _get(table, ticker, fy, fact["per"])
        return percent_pattern(100 * value / whole)
    if unit == "USD/shares":
        return per_share_pattern(value)
    return money_pattern(value)


def describe_fact(fact: dict[str, Any], table: Table) -> str:
    """The derived answer in words, for the results file ("AAPL FY2024 revenue = ... USD")."""
    ticker, fy, metric = fact["ticker"], int(fact["fiscal_year"]), fact["metric"]
    value, unit = _get(table, ticker, fy, metric)
    if "compare_to" in fact:
        base, _ = _get(table, ticker, int(fact["compare_to"]), metric)
        growth = 100 * (value - base) / abs(base)
        return f"{ticker} {metric} FY{fy} vs FY{fact['compare_to']}: {growth:+.2f}%"
    if "per" in fact:
        whole, _ = _get(table, ticker, fy, fact["per"])
        return f"{ticker} FY{fy} {metric}/{fact['per']}: {100 * value / whole:.2f}%"
    return f"{ticker} FY{fy} {metric} = {value:,.2f} {unit}"


def _get(table: Table, ticker: str, fy: int, metric: str) -> tuple[float, str]:
    try:
        value, unit = table[(ticker, fy, metric)]
    except KeyError:
        raise GroundTruthError(f"no {metric} for {ticker} FY{fy} in financials.csv") from None
    if not math.isfinite(value):
        raise GroundTruthError(f"{metric} for {ticker} FY{fy} is not a number")
    return value, unit
