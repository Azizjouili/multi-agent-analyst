"""Fetch the pinned SEC EDGAR corpus: 10-K documents and XBRL financials.

    uv run python scripts/fetch_edgar.py                    # everything in the manifest
    uv run python scripts/fetch_edgar.py --only AAPL MSFT   # some companies
    uv run python scripts/fetch_edgar.py --force            # re-download existing files
    uv run python scripts/fetch_edgar.py --pin AAPL MSFT --years 3
                                    # (re)pin the latest 10-Ks of these tickers in the manifest

Reads data/corpus/manifest.yaml (committed) and writes, into git-ignored folders:
  data/docs/<TICKER>_<FY>.html        the primary 10-K document, as filed
  data/docs/<TICKER>_<FY>.json        who/what/when of that filing (read by the RAG loader)
  data/datasets/companyfacts/*.json   raw XBRL company facts, cached
  data/datasets/financials.csv        key annual line items, one row per value

Why pinned accession numbers and not "the latest 10-K": "latest" changes every
year, so the corpus, the index and the eval ground truth would silently drift.
An accession number names one immutable filing. Anyone who runs this gets the
same documents and the same numbers, today or in five years. `--pin` is the
explicit, reviewable step that moves the corpus forward.

Why the corpus isn't committed: it's ~40 filings of several MB each, all
public and re-downloadable. The manifest is what makes it reproducible.

SEC's fair-access rules: every request carries a User-Agent with a name and
contact email (SEC_USER_AGENT), and requests are throttled to ~2 per second,
well under SEC's limit of 10. SEC answers 403 to requests without a proper
User-Agent and to clients that go too fast; we stop with a clear message
instead of retrying into a longer ban. Downloads are resumable: existing files
are skipped, and files are written atomically, so an interrupted run leaves no
truncated file behind.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "manifest.yaml"
DOCS_DIR = ROOT / "data" / "docs"
DATASETS_DIR = ROOT / "data" / "datasets"

DEFAULT_USER_AGENT = "multi-agent-analyst azizjouili@hotmail.fr"
# Seconds between requests: ~2 requests/second, a fifth of SEC's 10/s limit.
MIN_INTERVAL = 0.5
RETRIES = 3  # for timeouts and 5xx only; 403/429 stop immediately (see module docstring)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

FINANCIALS_COLUMNS = [
    "ticker", "company", "cik", "fiscal_year", "period_end",
    "metric", "value", "unit", "concept", "accession",
]  # fmt: skip

# metric -> (XBRL us-gaap concepts in order of preference, unit, kind).
# Several concepts per metric because companies tag the same line differently
# (Apple's "net sales" is RevenueFromContractWithCustomer..., NVIDIA's is
# Revenues). "Revenues" comes first: when both exist, it is the total (e.g.
# Walmart's total revenues include membership income; net sales don't).
# kind: "duration" values cover the fiscal year (income, cash flow); "instant"
# values are balances on the last day of it.
METRICS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "revenue": (
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
        "USD",
        "duration",
    ),
    "gross_profit": (("GrossProfit",), "USD", "duration"),
    "operating_income": (("OperatingIncomeLoss",), "USD", "duration"),
    "net_income": (("NetIncomeLoss", "ProfitLoss"), "USD", "duration"),
    "rd_expense": (
        (
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ),
        "USD",
        "duration",
    ),
    "eps_diluted": (("EarningsPerShareDiluted",), "USD/shares", "duration"),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities",), "USD", "duration"),
    "capital_expenditure": (
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        "USD",
        "duration",
    ),
    "total_assets": (("Assets",), "USD", "instant"),
    "total_liabilities": (("Liabilities",), "USD", "instant"),
    "stockholders_equity": (
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        "USD",
        "instant",
    ),
    "cash_and_equivalents": (("CashAndCashEquivalentsAtCarryingValue",), "USD", "instant"),
}


class SECBlockedError(SystemExit):
    """SEC refused us (403/429). Retrying would only extend the block."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: int
    fiscal_year: int
    accession: str  # "0000320193-24-000123"
    period_end: str  # ISO date the fiscal year ended
    filed: str
    primary_document: str

    @property
    def stem(self) -> str:
        return f"{self.ticker}_{self.fiscal_year}"

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.primary_document
        )

    def meta(self) -> dict[str, Any]:
        """The sidecar the RAG loader reads: it can't learn these from the HTML reliably."""
        return {
            "ticker": self.ticker,
            "company": self.company,
            "cik": self.cik,
            "form": "10-K",
            "fiscal_year": self.fiscal_year,
            "period_end": self.period_end,
            "filed": self.filed,
            "accession": self.accession,
            "url": self.url,
        }


def load_manifest(path: Path = MANIFEST) -> list[Filing]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    filings = []
    for company in data.get("companies", []):
        for f in company.get("filings", []):
            filings.append(
                Filing(
                    ticker=company["ticker"],
                    company=company["name"],
                    cik=int(company["cik"]),
                    fiscal_year=int(f["fiscal_year"]),
                    accession=f["accession"],
                    period_end=str(f["period_end"]),
                    filed=str(f["filed"]),
                    primary_document=f["primary_document"],
                )
            )
    stems = [f.stem for f in filings]
    if len(stems) != len(set(stems)):
        raise SystemExit(f"{path}: two filings share a ticker and fiscal year")
    return filings


MANIFEST_HEADER = """\
# The SEC EDGAR corpus, pinned. scripts/fetch_edgar.py downloads exactly these filings.
#
# Each 10-K is identified by its accession number, which names one immutable
# filing, so the corpus (and the eval ground truth derived from it) is the same
# on every machine and never drifts. To move to newer filings, run
#     uv run python scripts/fetch_edgar.py --pin <TICKER ...> --years 3
# and review the diff of this file.
#
# fiscal_year is the company's own label (XBRL DocumentFiscalYearFocus): Walmart's
# year ending January 2025 is "fiscal 2025". period_end is the last day of it.
"""


def write_manifest(companies: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"companies": companies}, sort_keys=False, allow_unicode=True)
    path.write_text(MANIFEST_HEADER + "\n" + body, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class SECClient:
    """httpx with SEC's required User-Agent, a throttle, and clear failures."""

    def __init__(
        self,
        user_agent: str | None = None,
        min_interval: float = MIN_INTERVAL,
        transport: httpx.BaseTransport | None = None,  # tests pass a MockTransport
    ):
        # The environment wins, then .env (like the app's own settings), then the default.
        self.user_agent = (
            user_agent
            or os.environ.get("SEC_USER_AGENT")
            or dotenv_values(ROOT / ".env").get("SEC_USER_AGENT")
            or DEFAULT_USER_AGENT
        )
        self.min_interval = min_interval
        self._last = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60,
            follow_redirects=True,
            transport=transport,
        )
        self.requests = 0

    def get(self, url: str) -> bytes:
        for attempt in range(1, RETRIES + 1):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                response = self.client.get(url)
            except httpx.TransportError as exc:
                if attempt == RETRIES:
                    raise SystemExit(f"Network error fetching {url}: {exc}") from exc
                time.sleep(2**attempt)
                continue
            if response.status_code in (403, 429):
                raise SECBlockedError(
                    f"SEC returned {response.status_code} for {url}.\n"
                    "SEC refuses requests without a descriptive User-Agent and throttles "
                    "clients that go too fast. Set SEC_USER_AGENT to your name and email, e.g.\n"
                    '    SEC_USER_AGENT="Jane Doe jane@example.com"\n'
                    f"(current: {self.user_agent!r}), wait ~10 minutes, and run again: "
                    "files already downloaded are skipped."
                )
            if response.status_code >= 500 and attempt < RETRIES:
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise SystemExit(f"SEC returned {response.status_code} for {url}")
            return response.content
        raise AssertionError("unreachable")

    def get_json(self, url: str) -> Any:
        return json.loads(self.get(url))


def write_atomic(path: Path, data: bytes) -> None:
    # Write next to the target, then rename: a crash mid-download leaves a .part
    # file, never a truncated one that a resumed run would mistake for complete.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def cached(path: Path, url: str, sec: SECClient, force: bool) -> Path:
    if force or not path.exists() or path.stat().st_size == 0:
        write_atomic(path, sec.get(url))
    return path


# ---------------------------------------------------------------------------
# XBRL facts -> tidy rows
# ---------------------------------------------------------------------------
def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def annual_value(facts: dict, concepts: tuple[str, ...], unit: str, kind: str, filing: Filing):
    """(value, concept) for the filing's own fiscal year, as reported in that filing.

    A 10-K also restates the two prior years, and later filings may revise a
    figure, so "the FY2024 revenue" must name its source: we take the value
    *this* accession reported for the period ending on its period_end. That is
    also the number printed in the document the RAG agent reads.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        for entry in gaap.get(concept, {}).get("units", {}).get(unit, []):
            if entry.get("accn") != filing.accession or entry.get("end") != filing.period_end:
                continue
            if not str(entry.get("form", "")).startswith("10-K"):
                continue
            start = entry.get("start")
            if kind == "instant" and start is not None:
                continue
            # About a year: excludes the quarterly values some 10-Ks also tag.
            if kind == "duration" and (
                start is None or not 350 <= _days(start, entry["end"]) <= 380
            ):
                continue
            return entry["val"], concept
    return None


def financial_rows(facts: dict, filing: Filing) -> list[dict[str, Any]]:
    rows = []
    for metric, (concepts, unit, kind) in METRICS.items():
        found = annual_value(facts, concepts, unit, kind, filing)
        if found is None:
            continue  # e.g. Walmart reports no R&D line, Amazon no gross profit
        value, concept = found
        rows.append(
            {
                "ticker": filing.ticker,
                "company": filing.company,
                "cik": filing.cik,
                "fiscal_year": filing.fiscal_year,
                "period_end": filing.period_end,
                "metric": metric,
                "value": value,
                "unit": unit,
                "concept": concept,
                "accession": filing.accession,
            }
        )
    return rows


def write_financials(rows: list[dict[str, Any]], path: Path) -> None:
    # Sorted, so the file (and the SQLite table built from it) is byte-identical
    # on every machine.
    order = list(METRICS)
    rows = sorted(rows, key=lambda r: (r["ticker"], r["fiscal_year"], order.index(r["metric"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINANCIALS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def fetch(
    filings: list[Filing],
    sec: SECClient,
    docs_dir: Path = DOCS_DIR,
    datasets_dir: Path = DATASETS_DIR,
    force: bool = False,
    all_filings: list[Filing] | None = None,
) -> dict[str, int]:
    """Download `filings`, then rebuild financials.csv.

    The CSV covers `all_filings` (default: `filings`) whose company facts are
    cached, so `--only AAPL` refreshes Apple without dropping the others' rows.
    """
    all_filings = all_filings or filings
    downloaded = skipped = 0
    for i, filing in enumerate(filings, start=1):
        doc = docs_dir / f"{filing.stem}.html"
        if doc.exists() and doc.stat().st_size and not force:
            skipped += 1
        else:
            print(f"[{i}/{len(filings)}] {filing.stem}: {filing.url}", flush=True)
            write_atomic(doc, sec.get(filing.url))
            downloaded += 1
        # Always rewritten: tiny, and it must follow manifest edits.
        meta = json.dumps(filing.meta(), indent=2) + "\n"
        (docs_dir / f"{filing.stem}.json").write_text(meta, encoding="utf-8")

    # The financials CSV is rebuilt from the cached facts on every run. That's
    # cheap, and a change to METRICS then applies without re-downloading.
    selected = {f.cik for f in filings}
    rows: list[dict[str, Any]] = []
    for cik in sorted({f.cik for f in all_filings}):
        path = datasets_dir / "companyfacts" / f"CIK{cik:010d}.json"
        if cik in selected:
            cached(path, FACTS_URL.format(cik=cik), sec, force)
        elif not path.exists():
            continue
        facts = json.loads(path.read_text(encoding="utf-8"))
        for filing in (f for f in all_filings if f.cik == cik):
            found = financial_rows(facts, filing)
            if not any(r["metric"] == "revenue" for r in found):
                print(f"  warning: no revenue found for {filing.stem}", file=sys.stderr)
            rows.extend(found)
    write_financials(rows, datasets_dir / "financials.csv")
    return {
        "filings": len(filings),
        "downloaded": downloaded,
        "skipped (already present)": skipped,
        "financial rows": len(rows),
        "companies": len({r["ticker"] for r in rows}),
        "requests": sec.requests,
    }


def fiscal_year_of(facts: dict, accession: str, period_end: str) -> int:
    """The company's own label for the year: the `fy` XBRL records for this filing.

    Fallback when absent: the calendar year the period ended in, minus one for
    52/53-week years that end in the first days of January (they belong to the
    year before).
    """
    for concept in facts.get("facts", {}).get("dei", {}).values():
        for entries in concept.get("units", {}).values():
            for e in entries:
                if e.get("accn") == accession and e.get("fy"):
                    return int(e["fy"])
    end = date.fromisoformat(period_end)
    return end.year - 1 if end.month == 1 and end.day <= 7 else end.year


def _submission_pages(sub: dict, sec: SECClient):
    """The filing lists of a submissions response, newest first.

    `recent` holds only the last ~1,000 filings. A company that files many
    Form 4s (insider trades), like Meta, pushes its older 10-Ks into the
    extra pages listed under `files`, fetched only when needed.
    """
    yield sub["filings"]["recent"]
    for extra in sub["filings"].get("files", []):
        yield sec.get_json(f"https://data.sec.gov/submissions/{extra['name']}")


def pin(tickers: list[str], years: int, sec: SECClient, path: Path = MANIFEST) -> list[dict]:
    """Pin the latest `years` original 10-Ks (no amendments) of each ticker."""
    directory = sec.get_json(TICKERS_URL)
    ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in directory.values()}
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    companies = {c["ticker"]: c for c in (existing or {}).get("companies", [])}

    for ticker in (t.upper() for t in tickers):
        if ticker not in ciks:
            raise SystemExit(f"Unknown ticker {ticker!r} (not in {TICKERS_URL})")
        cik = ciks[ticker]
        sub = sec.get_json(SUBMISSIONS_URL.format(cik=cik))
        facts_path = cached(
            DATASETS_DIR / "companyfacts" / f"CIK{cik:010d}.json", FACTS_URL.format(cik=cik), sec,
            force=True,  # pinning should see the newest facts
        )  # fmt: skip
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        chosen = []
        for page in _submission_pages(sub, sec):
            for i, form in enumerate(page["form"]):
                if form != "10-K" or len(chosen) == years:
                    continue
                accession, period_end = page["accessionNumber"][i], page["reportDate"][i]
                chosen.append(
                    {
                        "fiscal_year": fiscal_year_of(facts, accession, period_end),
                        "accession": accession,
                        "period_end": period_end,
                        "filed": page["filingDate"][i],
                        "primary_document": page["primaryDocument"][i],
                    }
                )
            if len(chosen) == years:
                break
        companies[ticker] = {
            "ticker": ticker,
            # Keep a name already edited in the manifest ("Microsoft Corporation"
            # reads better in citations than SEC's conformed "MICROSOFT CORP").
            "name": companies.get(ticker, {}).get("name") or sub["name"],
            "cik": cik,
            "filings": sorted(chosen, key=lambda f: f["fiscal_year"]),
        }
        print(f"{ticker}: {sub['name']} -> FY {[f['fiscal_year'] for f in chosen]}")

    ordered = [companies[t] for t in sorted(companies)]
    write_manifest(ordered, path)
    return ordered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned SEC EDGAR corpus.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--only", nargs="+", metavar="TICKER", help="fetch only these companies")
    parser.add_argument("--force", action="store_true", help="re-download files that exist")
    parser.add_argument("--pin", nargs="+", metavar="TICKER", help="pin these tickers, then exit")
    parser.add_argument("--years", type=int, default=3, help="10-Ks per ticker for --pin")
    args = parser.parse_args(argv)

    sec = SECClient()
    print(f"User-Agent: {sec.user_agent}")
    if args.pin:
        pin(args.pin, args.years, sec, args.manifest)
        print(f"Wrote {args.manifest}. Review the diff, then run without --pin to fetch.")
        return

    filings = all_filings = load_manifest(args.manifest)
    if args.only:
        wanted = {t.upper() for t in args.only}
        filings = [f for f in filings if f.ticker in wanted]
        if not filings:
            raise SystemExit(f"No filings in {args.manifest} for {', '.join(sorted(wanted))}")
    stats = fetch(filings, sec, args.docs_dir, args.datasets_dir, args.force, all_filings)
    print(f"Corpus ready in {args.docs_dir} and {args.datasets_dir / 'financials.csv'}")
    for key, value in stats.items():
        print(f"  {key:>26}: {value}")


if __name__ == "__main__":
    main()
