"""A tiny synthetic corpus in SEC EDGAR's formats, for offline tests.

The real corpus (scripts/fetch_edgar.py) is ~40 filings of several MB, fetched
over the network, so tests never use it. This module writes a stand-in with the
same *shapes* the real code must handle, small enough to read in full:

- 10-K HTML in the inline-XBRL style: a hidden <ix:header> with facts that
  must not leak into the text, a table of contents listing every Item, page
  footers repeated on each printed page, a running "Item 1A" page header (as
  Microsoft prints), and layout tables that split "$" from the number;
- a .json sidecar per filing, as fetch_edgar writes it;
- XBRL company-facts JSON, including the traps the extraction must avoid: the
  prior year's comparative value, a quarterly value and a later restatement;
- financials.csv, built from those facts by fetch_edgar's own code.

Kestrel Ridge Outdoor Co. (KRO) and Bluefin Analytics Inc. (BFA) are fictional.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from scripts import fetch_edgar

KRO = ("KRO", "Kestrel Ridge Outdoor Co.", 9000001)
BFA = ("BFA", "Bluefin Analytics Inc.", 9000002)

# (ticker, fiscal year) -> metric -> value. The ground truth of every test.
FACTS: dict[tuple[str, int], dict[str, float]] = {
    ("KRO", 2022): {"revenue": 412.6e6, "net_income": 31.2e6, "total_assets": 520.0e6,
                    "eps_diluted": 1.52, "rd_expense": 12.5e6, "gross_profit": 210.4e6},
    ("KRO", 2023): {"revenue": 393.1e6, "net_income": 18.4e6, "total_assets": 505.3e6,
                    "eps_diluted": 0.9, "rd_expense": 12.9e6, "gross_profit": 192.6e6},
    ("KRO", 2024): {"revenue": 466.7e6, "net_income": 42.9e6, "total_assets": 571.8e6,
                    "eps_diluted": 2.1, "rd_expense": 14.1e6, "gross_profit": 247.4e6},
    # No gross profit line: a metric some companies simply don't report.
    ("BFA", 2024): {"revenue": 88.0e6, "net_income": -5.5e6, "total_assets": 140.2e6,
                    "eps_diluted": -0.41, "rd_expense": 21.4e6},
}  # fmt: skip
PERIOD_END = {2022: "2022-12-31", 2023: "2023-12-31", 2024: "2024-12-31"}
BFA_PERIOD_END = "2024-06-30"  # a June fiscal year, like Microsoft's

# XBRL concept used by each company for each metric. They differ on purpose:
# fetch_edgar must find revenue under either name.
CONCEPTS = {
    "KRO": {"revenue": "RevenueFromContractWithCustomerExcludingAssessedTax"},
    "BFA": {"revenue": "Revenues"},
}
DEFAULT_CONCEPTS = {
    "net_income": "NetIncomeLoss",
    "total_assets": "Assets",
    "eps_diluted": "EarningsPerShareDiluted",
    "rd_expense": "ResearchAndDevelopmentExpense",
    "gross_profit": "GrossProfit",
}

# The narrative, by filing and Item. Item 8 holds a marker word that must NOT
# be indexed by default (financial statements are left to the SQL agent).
ITEM8_MARKER = "Zymurgical"
HIDDEN_MARKER = "HiddenXbrlContext"

NARRATIVE: dict[tuple[str, int], dict[str, list[str]]] = {
    ("KRO", 2022): {
        "1": [
            "Kestrel Ridge Outdoor Co. designs tents, packs, apparel and footwear for hikers "
            "and campers. Our products are sold through wholesale partners and our own stores.",
        ],
        "1A": [
            "Supplier concentration. Tanh Loc Manufacturing produced about 42% of our units in "
            "fiscal 2022. A disruption at its plants would delay shipments for several months.",
            "Weather. Unusually wet summers reduce demand for camping equipment.",
        ],
        "3": ["We are not party to any material legal proceedings."],
        "7": [
            "Net revenue grew 11.1% to $412.6 million in fiscal 2022, led by apparel.",
        ],
        "7A": ["Our foreign currency exposure comes mainly from euro-denominated sales."],
        "8": [f"{ITEM8_MARKER} consolidated balance sheets and statements of operations."],
    },
    ("KRO", 2023): {
        "1": ["Kestrel Ridge Outdoor Co. sells outdoor equipment in three regions."],
        "1A": [
            "Wholesale destocking. Our wholesale partners reduced their inventory in fiscal "
            "2023, cutting orders well below sell-through to consumers.",
            "Supplier concentration. Tanh Loc Manufacturing produced about 37% of our units.",
        ],
        "3": ["We are not party to any material legal proceedings."],
        "7": [
            "Net revenue fell 4.7% to $393.1 million in fiscal 2023 as wholesale partners "
            "worked through excess inventory.",
        ],
        "7A": ["Our foreign currency exposure comes mainly from euro-denominated sales."],
        "8": [f"{ITEM8_MARKER} consolidated balance sheets and statements of operations."],
    },
    ("KRO", 2024): {
        "1": [
            "Kestrel Ridge Outdoor Co. designs tents, packs, apparel and footwear. "
            "Direct-to-consumer sales reached 46% of net revenue in fiscal 2024.",
        ],
        "1A": [
            "Tariffs. Most of our products are made in Southeast Asia. New tariffs on imports "
            "from the region would raise our product costs; a 10% tariff would reduce gross "
            "margin by about 2 percentage points. We are evaluating a second factory in Mexico "
            "to reduce this exposure, which would take at least two years to qualify.",
            "Supplier concentration. Tanh Loc Manufacturing produced about 31% of our units in "
            "fiscal 2024, down from 42% in fiscal 2022, as we qualified additional suppliers in "
            "Vietnam and Indonesia. A prolonged disruption at its plants would still delay "
            "shipments of our best-selling tents for several months.",
            "Product safety. In fiscal 2024 we recalled the Summit backpacking stove after "
            "reports of a faulty valve. Further recalls could harm our brand and require us to "
            "record additional warranty reserves.",
            "Cybersecurity. Our e-commerce site stores customer data; a breach could expose us "
            "to fines and lost sales. We test our defenses with outside firms every year.",
        ],
        "3": ["We are not party to any material legal proceedings."],
        "7": [
            "Net revenue rose 18.7% to $466.7 million in fiscal 2024, a record year, driven by "
            "direct-to-consumer growth and the new trail-running footwear line.",
            "Gross margin improved as freight costs normalized and promotions eased.",
        ],
        "7A": ["Our foreign currency exposure comes mainly from euro-denominated sales."],
        "8": [f"{ITEM8_MARKER} consolidated balance sheets and statements of operations."],
    },
    ("BFA", 2024): {
        "1": [
            "Bluefin Analytics Inc. sells catch-forecasting software to commercial fisheries "
            "on annual subscriptions.",
        ],
        "1A": [
            "Customer concentration. Our five largest fishery customers provide most of our "
            "subscription revenue; losing one would reduce revenue materially.",
        ],
        "3": ["We are not party to any material legal proceedings."],
        "7": ["Revenue reached $88.0 million in fiscal 2024; we reported a net loss."],
        "7A": ["We hold no derivative instruments."],
        "8": [f"{ITEM8_MARKER} consolidated balance sheets and statements of operations."],
    },
}

ITEM_ORDER = ["1", "1A", "1B", "2", "3", "4", "5", "7", "7A", "8", "9A", "15"]


def filings() -> list[fetch_edgar.Filing]:
    out = []
    for ticker, fy in NARRATIVE:
        _, company, cik = KRO if ticker == "KRO" else BFA
        end = PERIOD_END[fy] if ticker == "KRO" else BFA_PERIOD_END
        out.append(
            fetch_edgar.Filing(
                ticker=ticker,
                company=company,
                cik=cik,
                fiscal_year=fy,
                accession=accession(cik, fy),
                period_end=end,
                filed=f"{fy + 1}-02-15",
                primary_document=f"{ticker.lower()}-{end.replace('-', '')}.htm",
            )
        )
    return out


def accession(cik: int, fy: int) -> str:
    return f"{cik:010d}-{str(fy + 1)[2:]}-000010"


# ---------------------------------------------------------------------------
# 10-K HTML
# ---------------------------------------------------------------------------
def filing_html(filing: fetch_edgar.Filing) -> str:
    items = NARRATIVE[(filing.ticker, filing.fiscal_year)]
    footer = f"{escape(filing.company)} | {filing.fiscal_year} Form 10-K | {{n}}"
    pages: list[str] = []

    cover = (
        f"<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 10-K</p>"
        f"<p>{escape(filing.company)}</p>"
        "<table>"
        + "".join(
            f"<tr><td>Item {i}.</td><td>{escape(fetch_edgar_title(i))}</td><td>{n + 3}</td></tr>"
            for n, i in enumerate(ITEM_ORDER)
        )
        + "</table>"
    )
    pages.append(cover)
    for item in ITEM_ORDER:
        paragraphs = items.get(item, ["None."])
        body = "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
        if item == "7":
            # A layout table: "$" and the number sit in separate cells.
            body += (
                "<table><tr><td>Net revenue</td><td>$</td><td>"
                f"{FACTS[(filing.ticker, filing.fiscal_year)]['revenue'] / 1e6:.1f}"
                "</td><td>million</td></tr></table>"
            )
        page = (
            f'<p style="font-weight:bold">Item {item}. {escape(fetch_edgar_title(item))}</p>{body}'
        )
        if item == "1A":
            # Risk factors run over two printed pages, the second with a
            # running "Item 1A" header (as Microsoft prints it).
            half = len(paragraphs) // 2 or 1
            first = "".join(f"<p>{escape(p)}</p>" for p in paragraphs[:half])
            second = "".join(f"<p>{escape(p)}</p>" for p in paragraphs[half:])
            pages.append(f'<p style="font-weight:bold">Item 1A. Risk Factors</p>{first}')
            page = f"<p>PART I Item 1A</p>{second}"
        pages.append(page)

    html_pages = "".join(
        f'<div>{page}<p>{footer.format(n=n)}</p></div><hr style="page-break-after:always"/>'
        for n, page in enumerate(pages, start=1)
    )
    hidden = (
        '<div style="display:none"><ix:header><ix:hidden>'
        f"<ix:nonNumeric>{HIDDEN_MARKER} dei:EntityRegistrantName</ix:nonNumeric>"
        "</ix:hidden></ix:header></div>"
    )
    return (
        "<?xml version='1.0' encoding='ASCII'?>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">'
        f"<head><title>{filing.ticker.lower()}-10k</title><style>p {{margin:0}}</style></head>"
        f"<body>{hidden}{html_pages}</body></html>"
    )


def fetch_edgar_title(item: str) -> str:
    from analyst.rag.edgar import ITEM_TITLES

    return ITEM_TITLES[item]


# ---------------------------------------------------------------------------
# XBRL company facts
# ---------------------------------------------------------------------------
def company_facts(ticker: str) -> dict:
    own = [f for f in filings() if f.ticker == ticker]
    gaap: dict[str, dict] = {}

    def add(concept: str, unit: str, entry: dict) -> None:
        gaap.setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(entry)

    for filing in own:
        facts = FACTS[(ticker, filing.fiscal_year)]
        end = filing.period_end
        start = f"{int(end[:4]) - 1}{end[4:]}"  # a year earlier (+1 day doesn't matter here)
        for metric, value in facts.items():
            concept = CONCEPTS.get(ticker, {}).get(metric) or DEFAULT_CONCEPTS[metric]
            _, unit, kind = fetch_edgar.METRICS[metric]
            entry = {
                "end": end, "val": value, "accn": filing.accession, "fy": filing.fiscal_year,
                "fp": "FY", "form": "10-K", "filed": filing.filed,
            }  # fmt: skip
            if kind == "duration":
                entry["start"] = start
            add(concept, unit, entry)
            if metric == "revenue":
                # Trap 1: the same filing also tags its fourth quarter.
                add(concept, unit, {**entry, "start": f"{end[:5]}10-01", "val": value / 4})
                # Trap 2: the prior year's comparative, in this same filing.
                prev = FACTS.get((ticker, filing.fiscal_year - 1), {}).get("revenue")
                if prev:
                    prev_end = f"{int(end[:4]) - 1}{end[4:]}"
                    prev_start = f"{int(end[:4]) - 2}{end[4:]}"
                    add(concept, unit, {**entry, "end": prev_end, "start": prev_start, "val": prev})
                # Trap 3: a later filing restating this year's value.
                add(concept, unit, {**entry, "accn": "0000000000-99-000001", "val": value + 1e6})
    dei = {
        "DocumentFiscalYearFocus": {
            "units": {
                "pure": [
                    {
                        "accn": f.accession,
                        "fy": f.fiscal_year,
                        "val": f.fiscal_year,
                        "end": f.period_end,
                    }
                    for f in own
                ]  # fmt: skip
            }
        }
    }
    _, company, cik = KRO if ticker == "KRO" else BFA
    return {"cik": cik, "entityName": company, "facts": {"us-gaap": gaap, "dei": dei}}


# ---------------------------------------------------------------------------
# A small PDF, written by hand
# ---------------------------------------------------------------------------
def write_pdf(path: Path, pages: list[list[str]], title: str) -> None:
    """A valid PDF with one Helvetica text line per string. No library needed.

    Why by hand: the PDF loader still needs a test file, and this is ~30 lines
    instead of a reportlab dependency kept only for the tests.
    """

    def text(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n_pages = len(pages)
    page_ids = [5 + 2 * i for i in range(n_pages)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] "
        f"/Count {n_pages} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: f"<< /Title ({text(title)}) >>".encode(),
    }
    for pid, lines in zip(page_ids, pages, strict=True):
        ops = "BT /F1 11 Tf 14 TL 72 720 Td " + " ".join(f"({text(ln)}) Tj T*" for ln in lines)
        stream = (ops + " ET").encode()
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode()
        objects[pid + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += b"%d 0 obj\n" % oid + objects[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offsets[i] for i in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 4 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        size,
        xref,
    )
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
def generate(out: Path) -> list[fetch_edgar.Filing]:
    """Write docs/, datasets/ (facts + financials.csv) and manifest.yaml under `out`."""
    docs, datasets = out / "docs", out / "datasets"
    docs.mkdir(parents=True, exist_ok=True)
    (datasets / "companyfacts").mkdir(parents=True, exist_ok=True)

    all_filings = filings()
    rows = []
    for ticker, _, cik in (KRO, BFA):
        facts = company_facts(ticker)
        (datasets / "companyfacts" / f"CIK{cik:010d}.json").write_text(
            json.dumps(facts), encoding="utf-8"
        )
        for filing in (f for f in all_filings if f.ticker == ticker):
            rows.extend(fetch_edgar.financial_rows(facts, filing))
    fetch_edgar.write_financials(rows, datasets / "financials.csv")

    for filing in all_filings:
        (docs / f"{filing.stem}.html").write_text(filing_html(filing), encoding="utf-8")
        (docs / f"{filing.stem}.json").write_text(json.dumps(filing.meta()), encoding="utf-8")

    companies = []
    for ticker, company, cik in (KRO, BFA):
        companies.append(
            {
                "ticker": ticker,
                "name": company,
                "cik": cik,
                "filings": [
                    {
                        "fiscal_year": f.fiscal_year,
                        "accession": f.accession,
                        "period_end": f.period_end,
                        "filed": f.filed,
                        "primary_document": f.primary_document,
                    }
                    for f in all_filings
                    if f.ticker == ticker
                ],
            }
        )
    fetch_edgar.write_manifest(companies, out / "manifest.yaml")
    return all_filings
