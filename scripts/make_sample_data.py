"""Generate the synthetic sample data for a fictional company.

    uv run python scripts/make_sample_data.py

Writes:
  data/datasets/quarterly_sales.csv   -> queried by the data-analysis agent (SQL)
  data/docs/annual_report_<year>.pdf  -> searched by the RAG agent

Design rules:
1. The CSV is generated first, and every figure in the PDFs is computed from
   it. The documents and the dataset therefore never contradict each other,
   and a test checks this.
2. The two sources *complement* each other so that a demo question needs
   every specialist:
   - the PDFs have narrative facts the CSV lacks (risks, supplier
     concentration, a product recall, store count, guidance);
   - the CSV has detail the PDFs lack (quarter x region x product x channel);
   - the web agent adds outside context (industry trends, tariffs).
3. It is deterministic (fixed seed), so evals can rely on exact numbers.

Kestrel Ridge Outdoor Co. and every person and partner named here are fictional.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

COMPANY = "Kestrel Ridge Outdoor Co."
YEARS = (2022, 2023, 2024)
QUARTERS = (1, 2, 3, 4)
REGIONS = ("North America", "Europe", "Asia-Pacific")
PRODUCTS = ("Tents & Shelters", "Packs & Bags", "Apparel", "Footwear")
DTC, WHOLESALE = "Direct-to-Consumer", "Wholesale"
CHANNELS = (DTC, WHOLESALE)

# 2022 annual revenue, USD millions, by region and product line.
BASE_2022 = {
    "North America": {"Tents & Shelters": 70, "Packs & Bags": 55, "Apparel": 85, "Footwear": 40},
    "Europe": {"Tents & Shelters": 30, "Packs & Bags": 25, "Apparel": 40, "Footwear": 20},
    "Asia-Pacific": {"Tents & Shelters": 12, "Packs & Bags": 10, "Apparel": 18, "Footwear": 8},
}
# Multipliers relative to 2022. The intended story: a 2023 dip (wholesale
# destocking, weak European consumer), then a 2024 rebound to a record year
# (direct-to-consumer, Asia-Pacific, the new trail-running footwear line).
REGION_TREND = {
    "North America": {2022: 1.00, 2023: 0.95, 2024: 1.00},
    "Europe": {2022: 1.00, 2023: 0.90, 2024: 0.93},
    "Asia-Pacific": {2022: 1.00, 2023: 1.18, 2024: 1.45},
}
PRODUCT_TREND = {
    "Tents & Shelters": {2022: 1.00, 2023: 0.97, 2024: 1.04},
    "Packs & Bags": {2022: 1.00, 2023: 1.00, 2024: 1.08},
    "Apparel": {2022: 1.00, 2023: 0.98, 2024: 1.08},
    "Footwear": {2022: 1.00, 2023: 1.02, 2024: 1.22},
}
DTC_SHARE = {2022: 0.35, 2023: 0.40, 2024: 0.46}
# Share of a year's revenue in each quarter. Gear peaks in summer; apparel
# and footwear peak at the holidays.
SEASONALITY = {
    "Tents & Shelters": (0.18, 0.32, 0.30, 0.20),
    "Packs & Bags": (0.18, 0.32, 0.30, 0.20),
    "Apparel": (0.22, 0.22, 0.24, 0.32),
    "Footwear": (0.22, 0.22, 0.24, 0.32),
}
AVG_PRICE = {"Tents & Shelters": 240, "Packs & Bags": 95, "Apparel": 60, "Footwear": 130}
GROSS_MARGIN = {DTC: 0.62, WHOLESALE: 0.44}
MARGIN_ADJ = {2022: 0.0, 2023: -0.02, 2024: 0.01}  # 2023: freight + promotional clearance
REVENUE_2021 = 371.5e6  # appears only in the 2022 report (not in the CSV)


# ---------------------------------------------------------------------------
# 1. The dataset
# ---------------------------------------------------------------------------
def generate_sales(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for year in YEARS:
        for q in QUARTERS:
            for region in REGIONS:
                for product in PRODUCTS:
                    for channel in CHANNELS:
                        share = DTC_SHARE[year] if channel == DTC else 1 - DTC_SHARE[year]
                        annual = (
                            BASE_2022[region][product]
                            * 1e6
                            * REGION_TREND[region][year]
                            * PRODUCT_TREND[product][year]
                            * share
                        )
                        # +/-3% noise so the quarters don't look machine-made.
                        revenue = round(
                            annual * SEASONALITY[product][q - 1] * rng.uniform(0.97, 1.03), 2
                        )
                        margin = GROSS_MARGIN[channel] + MARGIN_ADJ[year]
                        rows.append(
                            {
                                "period": f"{year}-Q{q}",
                                "year": year,
                                "quarter": q,
                                "region": region,
                                "product_line": product,
                                "channel": channel,
                                "units_sold": round(revenue / AVG_PRICE[product]),
                                "revenue_usd": revenue,
                                "cost_of_goods_usd": round(revenue * (1 - margin), 2),
                            }
                        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class YearSummary:
    year: int
    revenue: float = 0.0
    cogs: float = 0.0
    by_region: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_product: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_channel: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def gross_margin(self) -> float:
        return 1 - self.cogs / self.revenue

    @property
    def dtc_share(self) -> float:
        return self.by_channel[DTC] / self.revenue


def summarize(rows: list[dict]) -> dict[int, YearSummary]:
    out = {y: YearSummary(y) for y in YEARS}
    for r in rows:
        s = out[r["year"]]
        s.revenue += r["revenue_usd"]
        s.cogs += r["cost_of_goods_usd"]
        s.by_region[r["region"]] += r["revenue_usd"]
        s.by_product[r["product_line"]] += r["revenue_usd"]
        s.by_channel[r["channel"]] += r["revenue_usd"]
    return out


# ---------------------------------------------------------------------------
# 2. Narrative (facts that exist ONLY in the PDFs)
# ---------------------------------------------------------------------------
def money(x: float) -> str:
    return f"${x / 1e6:,.1f} million"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def change(cur: float, prev: float) -> str:
    g = cur / prev - 1
    return f"{'up' if g >= 0 else 'down'} {abs(g) * 100:.1f}%"


NARRATIVE = {
    2022: {
        "theme": "Strong demand and channel expansion",
        "letter": [
            "{company} delivered record net revenue of {revenue}, {growth} from {prev_revenue} "
            "in 2021. Demand for camping and hiking gear stayed high after the pandemic-era "
            "boom in outdoor recreation, and our wholesale partners rebuilt inventories "
            "throughout the year.",
            "Direct-to-consumer sales reached {dtc_share} of revenue. We opened two flagship "
            "stores, in Denver and Portland, bringing our store count to 6, and relaunched our "
            "e-commerce site. Gross margin was {gm}.",
            "Ocean freight costs peaked in the first half before easing, and we ended the year "
            "with elevated inventory, which we expect to work down in 2023.",
        ],
        "segments": {
            "Tents & Shelters": "Family camping tents led the category; the Aspen 4P tent was our "
            "best-selling product.",
            "Packs & Bags": "Daypacks grew fastest as hiking participation rose.",
            "Apparel": "Insulated jackets drove the strong fourth quarter.",
            "Footwear": "Hiking boots performed well; trail-running remains a gap in our range.",
        },
        "risks": [
            (
                "Supplier concentration",
                "Approximately 42% of our finished-goods units were produced by a single contract "
                "manufacturer, Tanh Loc Manufacturing in Vietnam. A disruption at this facility "
                "would materially affect our ability to meet demand.",
            ),
            (
                "Customer concentration",
                "Our largest wholesale partner, Summit Outfitters, accounted for 17% of net "
                "revenue. A reduction in its orders would significantly affect our results.",
            ),
            (
                "Freight and input costs",
                "Ocean freight rates and nylon and polyester prices were volatile in 2022. "
                "Increases we cannot pass on to customers reduce gross margin.",
            ),
            (
                "Foreign exchange",
                "About a quarter of revenue is earned in euros and other non-US currencies, "
                "while most product costs are in US dollars.",
            ),
            (
                "Weather and seasonality",
                "Sales of tents and packs depend on spring and summer camping seasons, and "
                "apparel on cold winters. Unusual weather can shift demand between quarters.",
            ),
        ],
        "outlook": "For 2023 we expect revenue to be broadly flat as wholesale partners reduce "
        "the inventory they built in 2022, partly offset by continued direct-to-consumer growth.",
        "metrics": {"Flagship stores": "6", "Employees": "1,140", "Wholesale doors": "2,350"},
    },
    2023: {
        "theme": "A reset year for wholesale",
        "letter": [
            "2023 was a challenging year. Net revenue was {revenue}, {growth} from {prev_revenue}. "
            "Wholesale partners, including Summit Outfitters, cut orders sharply to reduce excess "
            "inventory across the outdoor industry, and European consumer spending weakened.",
            "Direct-to-consumer kept growing and reached {dtc_share} of revenue. We opened three "
            "stores (9 in total). Gross margin declined to {gm} because of promotional clearance "
            "of 2022 inventory and residual freight costs.",
            "Asia-Pacific was a bright spot, growing for the third straight year as we expanded "
            "our presence in Japan and South Korea.",
        ],
        "segments": {
            "Tents & Shelters": "Wholesale destocking hit tents hardest.",
            "Packs & Bags": "Packs were stable, supported by back-to-school demand.",
            "Apparel": "Apparel softened in Europe after a mild winter.",
            "Footwear": "We began development of the Ridgeline Trail running-shoe line for 2024.",
        },
        "risks": [
            (
                "Customer concentration",
                "Summit Outfitters' share of net revenue fell to 13% as it reduced orders. Further "
                "consolidation among outdoor retailers could reduce our wholesale reach.",
            ),
            (
                "Supplier concentration",
                "We reduced our reliance on Tanh Loc Manufacturing to about 38% of units by "
                "onboarding a second factory in Indonesia, but concentration remains high.",
            ),
            (
                "Cybersecurity",
                "In August 2023 our e-commerce platform was targeted by a credential-stuffing "
                "attack. No payment data was taken, but about 2,800 customer accounts required "
                "password resets. As direct-to-consumer grows, so does our exposure.",
            ),
            (
                "Inventory and markdowns",
                "Industry-wide excess inventory pressured prices. Further markdowns would reduce "
                "gross margin.",
            ),
            (
                "Foreign exchange",
                "A weaker euro in the first half reduced the US-dollar value of European sales.",
            ),
        ],
        "outlook": "For 2024 we expect a return to growth, led by direct-to-consumer, Asia-Pacific "
        "and the launch of Ridgeline Trail footwear, as wholesale inventories normalize.",
        "metrics": {"Flagship stores": "9", "Employees": "1,185", "Wholesale doors": "2,120"},
    },
    2024: {
        "theme": "Record revenue, new risks",
        "letter": [
            "Net revenue reached a record {revenue}, {growth} from {prev_revenue}. Growth came "
            "from direct-to-consumer, now {dtc_share} of revenue, from Asia-Pacific, and from "
            "the successful launch of our Ridgeline Trail footwear line.",
            "Gross margin recovered to {gm} as clearance activity ended and our channel mix "
            "shifted toward direct-to-consumer. We opened five stores, for 14 in total.",
            "In June we voluntarily recalled 3,100 units of the Aspen 4P tent after a pole "
            "defect was identified. The recall cost approximately $1.2 million, and no injuries "
            "were reported.",
        ],
        "segments": {
            "Tents & Shelters": "Tents returned to growth despite the Aspen 4P recall.",
            "Packs & Bags": "Packs grew on strong direct-to-consumer demand.",
            "Apparel": "Apparel grew, led by the Asia-Pacific region.",
            "Footwear": "Ridgeline Trail made footwear our fastest-growing product line.",
        },
        "risks": [
            (
                "Tariffs and trade policy",
                "Proposed US tariffs on goods imported from Southeast Asia, where most of our "
                "products are made, could raise costs materially. We estimate that a 10% tariff "
                "would reduce gross margin by about 2 percentage points if not offset by price "
                "increases.",
            ),
            (
                "Supplier concentration",
                "Tanh Loc Manufacturing produced about 31% of units, down from 42% in 2022. "
                "We plan to add a factory in Mexico in 2025 to diversify further and reduce "
                "tariff exposure.",
            ),
            (
                "Product quality and recalls",
                "The Aspen 4P recall showed that quality failures can be costly and damage our "
                "brand. We have added independent pre-shipment testing for all tent poles.",
            ),
            (
                "Execution of Asia-Pacific growth",
                "Asia-Pacific growth depends on new distribution partners in Japan and South "
                "Korea, and on hiring local teams.",
            ),
            (
                "Climate and weather",
                "A warmer-than-normal winter reduced demand for insulated apparel in North "
                "America. Longer-term climate change may shift our seasonal patterns.",
            ),
        ],
        "outlook": "For 2025 we expect revenue growth of 8% to 10%, assuming no escalation of US "
        "tariffs. Our plans include opening four more stores and starting production in Mexico.",
        "metrics": {"Flagship stores": "14", "Employees": "1,260", "Wholesale doors": "2,180"},
    },
}


# ---------------------------------------------------------------------------
# 3. The PDFs
# ---------------------------------------------------------------------------
def build_report(year: int, s: YearSummary, prev: YearSummary | None, path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    n = NARRATIVE[year]
    prev_revenue = prev.revenue if prev else REVENUE_2021
    fields = {
        "company": COMPANY,
        "revenue": money(s.revenue),
        "prev_revenue": money(prev_revenue),
        "growth": change(s.revenue, prev_revenue),
        "dtc_share": pct(s.dtc_share),
        "gm": pct(s.gross_margin),
    }
    styles = getSampleStyleSheet()
    body, h1, h2 = styles["BodyText"], styles["Title"], styles["Heading2"]

    def para(text: str) -> Paragraph:
        # reportlab parses a small HTML subset, so escape the '&' in product names.
        return Paragraph(text.replace("&", "&amp;"), body)

    def table(rows: list[list[str]]) -> Table:
        t = Table(rows, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ]
            )
        )
        return t

    story = [
        Paragraph(f"{COMPANY}<br/>Annual Report {year}", h1),
        para(f"<i>{n['theme']}</i>"),
        para("<i>Fictional company. Synthetic data generated for a software demonstration.</i>"),
        Spacer(1, 0.2 * inch),
        Paragraph("Letter to Shareholders", h2),
        *[para(p.format(**fields)) for p in n["letter"]],
        para("Dana Whitfield, Chief Executive Officer"),
        PageBreak(),
        Paragraph("Financial Highlights", h2),
        para(
            f"Net revenue for {year} was {fields['revenue']} ({fields['growth']}). Gross margin "
            f"was {fields['gm']}, and direct-to-consumer sales were {fields['dtc_share']} of "
            "net revenue."
        ),
    ]

    region_rows = [["Region", f"{year} revenue", "Change"]]
    for region in REGIONS:
        cur = s.by_region[region]
        delta = change(cur, prev.by_region[region]) if prev else "n/a"
        region_rows.append([region, money(cur), delta])
    region_rows.append(["Total", money(s.revenue), fields["growth"]])
    story += [table(region_rows), Spacer(1, 0.2 * inch)]

    channel_rows = [["Channel", f"{year} revenue", "Share"]]
    for channel in CHANNELS:
        channel_rows.append(
            [channel, money(s.by_channel[channel]), pct(s.by_channel[channel] / s.revenue)]
        )
    story += [table(channel_rows), PageBreak(), Paragraph("Product Line Review", h2)]

    for product in PRODUCTS:
        cur = s.by_product[product]
        delta = f", {change(cur, prev.by_product[product])} from {year - 1}" if prev else ""
        story.append(
            para(f"<b>{product}.</b> Revenue was {money(cur)}{delta}. {n['segments'][product]}")
        )

    story += [PageBreak(), Paragraph("Risk Factors", h2)]
    for title, text in n["risks"]:
        story.append(para(f"<b>{title}.</b> {text}"))

    story += [PageBreak(), Paragraph("Outlook", h2), para(n["outlook"])]
    story += [
        Paragraph("Operating Metrics", h2),
        table([["Metric", str(year)], *n["metrics"].items()]),
    ]

    def footer(canvas, doc) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.drawString(inch, 0.5 * inch, f"{COMPANY} - Annual Report {year} - page {doc.page}")

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title=f"{COMPANY} Annual Report {year}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def generate(out_root: Path, seed: int = 42) -> dict[int, YearSummary]:
    rows = generate_sales(seed)
    write_csv(rows, out_root / "datasets" / "quarterly_sales.csv")
    summaries = summarize(rows)

    # Make sure the numbers tell the story the narrative describes.
    assert summaries[2023].revenue < summaries[2022].revenue, "2023 must be a down year"
    assert summaries[2024].revenue > summaries[2022].revenue, "2024 must be a record year"

    for year in YEARS:
        prev = summaries.get(year - 1)
        build_report(year, summaries[year], prev, out_root / "docs" / f"annual_report_{year}.pdf")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summaries = generate(args.out, args.seed)
    print(
        f"Wrote {args.out / 'datasets' / 'quarterly_sales.csv'} and 3 PDFs in {args.out / 'docs'}"
    )
    for y, s in summaries.items():
        gm, dtc = pct(s.gross_margin), pct(s.dtc_share)
        print(f"  {y}: revenue {money(s.revenue)}, gross margin {gm}, DTC {dtc}")
