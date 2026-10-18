"""Offline tests for scripts/fetch_edgar.py: an in-memory SEC, no network."""

import csv
import json

import httpx
import pytest

from scripts import fetch_edgar
from tests import edgar_fixture as fx


class FakeSEC:
    """Serves the fixture's filings and company facts like sec.gov would."""

    def __init__(self, status: int = 200):
        self.status = status
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, bytes] = {}
        for filing in fx.filings():
            self.routes[filing.url] = fx.filing_html(filing).encode()
        for ticker, _, cik in (fx.KRO, fx.BFA):
            url = fetch_edgar.FACTS_URL.format(cik=cik)
            self.routes[url] = json.dumps(fx.company_facts(ticker)).encode()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, request=request)
        body = self.routes.get(str(request.url))
        return httpx.Response(200 if body else 404, content=body or b"", request=request)

    def client(self, user_agent="Test Runner test@example.com") -> fetch_edgar.SECClient:
        return fetch_edgar.SECClient(
            user_agent, min_interval=0, transport=httpx.MockTransport(self.handler)
        )


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fetch_downloads_documents_sidecars_and_builds_financials(tmp_path):
    sec = FakeSEC()
    stats = fetch_edgar.fetch(fx.filings(), sec.client(), tmp_path / "docs", tmp_path / "datasets")
    assert stats["filings"] == stats["downloaded"] == 4 and stats["companies"] == 2

    html = (tmp_path / "docs" / "KRO_2024.html").read_text(encoding="utf-8")
    assert "Tanh Loc" in html
    meta = json.loads((tmp_path / "docs" / "KRO_2024.json").read_text(encoding="utf-8"))
    assert meta["company"] == "Kestrel Ridge Outdoor Co." and meta["fiscal_year"] == 2024
    assert meta["url"].startswith("https://www.sec.gov/Archives/edgar/data/9000001/")

    rows = _read_csv(tmp_path / "datasets" / "financials.csv")
    assert list(rows[0]) == fetch_edgar.FINANCIALS_COLUMNS
    assert len(rows) == sum(len(m) for m in fx.FACTS.values())
    for r in rows:
        assert float(r["value"]) == pytest.approx(
            fx.FACTS[(r["ticker"], int(r["fiscal_year"]))][r["metric"]]
        )


def test_every_request_carries_the_sec_user_agent(tmp_path):
    sec = FakeSEC()
    fetch_edgar.fetch(fx.filings(), sec.client("Jane Doe jane@example.com"), tmp_path, tmp_path)
    assert sec.requests
    assert {r.headers["user-agent"] for r in sec.requests} == {"Jane Doe jane@example.com"}


def test_user_agent_comes_from_the_environment_then_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_edgar, "ROOT", tmp_path)  # not the developer's real .env
    monkeypatch.setenv("SEC_USER_AGENT", "Env Person env@example.com")
    assert fetch_edgar.SECClient().user_agent == "Env Person env@example.com"
    monkeypatch.delenv("SEC_USER_AGENT")
    assert fetch_edgar.SECClient().user_agent == fetch_edgar.DEFAULT_USER_AGENT
    (tmp_path / ".env").write_text("SEC_USER_AGENT=Dot Env dot@example.com\n", encoding="utf-8")
    assert fetch_edgar.SECClient().user_agent == "Dot Env dot@example.com"


def test_second_run_resumes_without_downloading_again(tmp_path):
    sec = FakeSEC()
    fetch_edgar.fetch(fx.filings(), sec.client(), tmp_path / "docs", tmp_path / "datasets")
    first = len(sec.requests)
    stats = fetch_edgar.fetch(fx.filings(), sec.client(), tmp_path / "docs", tmp_path / "datasets")
    assert len(sec.requests) == first  # nothing fetched twice
    assert stats["downloaded"] == 0 and stats["skipped (already present)"] == 4
    assert not list(tmp_path.rglob("*.part"))  # no leftovers from atomic writes


def test_only_one_company_keeps_the_others_rows(tmp_path):
    sec = FakeSEC()
    docs, datasets = tmp_path / "docs", tmp_path / "datasets"
    fetch_edgar.fetch(fx.filings(), sec.client(), docs, datasets)
    kro = [f for f in fx.filings() if f.ticker == "KRO"]
    fetch_edgar.fetch(kro, sec.client(), docs, datasets, all_filings=fx.filings())
    assert {r["ticker"] for r in _read_csv(datasets / "financials.csv")} == {"BFA", "KRO"}


@pytest.mark.parametrize("status", [403, 429])
def test_sec_refusal_stops_with_a_clear_message(tmp_path, status):
    sec = FakeSEC(status=status)
    with pytest.raises(fetch_edgar.SECBlockedError, match="SEC_USER_AGENT") as err:
        fetch_edgar.fetch(fx.filings(), sec.client(), tmp_path, tmp_path)
    assert str(status) in str(err.value)
    assert len(sec.requests) == 1  # no retry storm against a block
    assert not list(tmp_path.glob("*.html"))


def test_annual_value_skips_quarters_comparatives_and_restatements():
    """The fixture facts hold all three traps next to the real FY2024 value."""
    kro_2024 = next(f for f in fx.filings() if f.stem == "KRO_2024")
    facts = fx.company_facts("KRO")
    concepts, unit, kind = fetch_edgar.METRICS["revenue"]
    value, concept = fetch_edgar.annual_value(facts, concepts, unit, kind, kro_2024)
    assert value == 466.7e6  # not the Q4 value, FY2023's comparative, or the restatement
    assert concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_fiscal_year_label_comes_from_the_filing():
    facts = fx.company_facts("KRO")
    kro_2023 = next(f for f in fx.filings() if f.stem == "KRO_2023")
    assert fetch_edgar.fiscal_year_of(facts, kro_2023.accession, kro_2023.period_end) == 2023
    # Without XBRL: the calendar year, except 52/53-week years ending in early January.
    assert fetch_edgar.fiscal_year_of({}, "x", "2025-01-31") == 2025  # Walmart: "fiscal 2025"
    assert fetch_edgar.fiscal_year_of({}, "x", "2022-01-01") == 2021


def test_manifest_round_trips(tmp_path, sample_data):
    out, filings = sample_data
    assert fetch_edgar.load_manifest(out / "manifest.yaml") == filings
    text = (out / "manifest.yaml").read_text(encoding="utf-8")
    assert text.startswith("# The SEC EDGAR corpus, pinned.")


def test_committed_manifest_pins_a_tractable_corpus():
    filings = fetch_edgar.load_manifest()
    tickers = {f.ticker for f in filings}
    assert 10 <= len(tickers) <= 15 and 30 <= len(filings) <= 45
    for f in filings:
        # Exact accession numbers, not "latest": the corpus can't drift.
        assert len(f.accession) == 20 and f.accession.count("-") == 2, f
        assert f.primary_document.endswith(".htm"), f
