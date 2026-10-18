"""Tests for the CSV loader and the read-only SQL tool's guardrails."""

import sqlite3

import pytest

from analyst.data.load import load_datasets
from analyst.tools.sql import (
    DataNotReadyError,
    SQLExecutionError,
    SQLGuardError,
    SQLTimeoutError,
    describe_schema,
    open_readonly,
    run_sql,
    table_catalog,
    validate_sql,
)
from tests.edgar_fixture import FACTS

ROWS = sum(len(metrics) for metrics in FACTS.values())  # 23: one row per company, year, metric


def _count(db):
    return run_sql("SELECT COUNT(*) FROM financials", db).rows[0][0]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_loader_creates_typed_table(sample_db):
    with sqlite3.connect(sample_db) as conn:
        cols = {name: typ for _, name, typ, *_ in conn.execute("PRAGMA table_info(financials)")}
    assert cols["fiscal_year"] == "INTEGER" and cols["cik"] == "INTEGER"
    assert cols["value"] == "REAL" and cols["metric"] == "TEXT" and cols["ticker"] == "TEXT"
    assert _count(sample_db) == ROWS


def test_loaded_values_match_the_xbrl_facts(sample_db):
    result = run_sql("SELECT ticker, fiscal_year, metric, value FROM financials", sample_db)
    assert len(result.rows) == ROWS
    for ticker, year, metric, value in result.rows:
        assert value == pytest.approx(FACTS[(ticker, year)][metric])


def test_loader_without_csvs_explains_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="fetch_edgar"):
        load_datasets(tmp_path, tmp_path / "x.db")


# ---------------------------------------------------------------------------
# Layer 1: static validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO financials (fiscal_year) VALUES (1999)",
        "UPDATE financials SET value = 0",
        "DELETE FROM financials",
        "DROP TABLE financials",
        "CREATE TABLE evil (x)",
        "ALTER TABLE financials ADD COLUMN x",
        "REPLACE INTO financials (fiscal_year) VALUES (1)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA writable_schema = ON",
        "VACUUM",
        "SELECT 1; DROP TABLE financials",
        "WITH x AS (SELECT 1) DELETE FROM financials",
        "SELECT * FROM financials WHERE 1 = 1; INSERT INTO financials (fiscal_year) VALUES (1)",
        "",
        "   ;  ",
    ],
)
def test_guard_rejects_anything_but_a_single_select(sql, sample_db):
    with pytest.raises(SQLGuardError):
        run_sql(sql, sample_db)
    assert _count(sample_db) == ROWS  # nothing changed


def test_guard_ignores_keywords_inside_literals_and_comments(sample_db):
    sql = "SELECT COUNT(*) FROM financials WHERE ticker = 'DROP TABLE; DELETE' -- ; DROP"
    assert run_sql(sql, sample_db).rows == [(0,)]


def test_guard_allows_cte_trailing_semicolon_and_replace_function(sample_db):
    sql = (
        "WITH t AS (SELECT REPLACE(metric, '_', ' ') AS m FROM financials) "
        "SELECT COUNT(DISTINCT m) FROM t;"
    )
    assert run_sql(sql, sample_db).rows == [(6,)]
    assert validate_sql("SELECT 1;  ") == "SELECT 1"


# ---------------------------------------------------------------------------
# Layers 2 + 3: the engine blocks writes even if validation is bypassed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM financials",
        "UPDATE financials SET value = 0",
        "CREATE TABLE evil (x)",
        "ATTACH DATABASE ':memory:' AS other",
    ],
)
def test_readonly_connection_blocks_writes_without_the_regex(sql, sample_db):
    # Call the engine directly, skipping validate_sql(), to prove the deeper
    # layers hold on their own.
    with open_readonly(sample_db) as conn, pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql)
    assert _count(sample_db) == ROWS


def test_authorizer_alone_blocks_writes(sample_db, tmp_path):
    # Even on a *writable* connection, the authorizer denies the write. It
    # runs on a throwaway copy, so a regression couldn't damage the shared
    # test database.
    import shutil

    from analyst.tools.sql import _authorizer

    copy = tmp_path / "copy.db"
    shutil.copy(sample_db, copy)
    conn = sqlite3.connect(copy)
    conn.set_authorizer(_authorizer)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute("DELETE FROM financials")
    conn.close()
    assert _count(copy) == ROWS


# ---------------------------------------------------------------------------
# Layer 4: row cap and timeout
# ---------------------------------------------------------------------------
def test_row_cap_truncates_and_flags(sample_db):
    capped = run_sql("SELECT * FROM financials", sample_db, max_rows=10)
    assert len(capped.rows) == 10 and capped.truncated
    assert "more than 10 rows" in capped.to_text()

    full = run_sql("SELECT * FROM financials", sample_db, max_rows=500)
    assert len(full.rows) == ROWS and not full.truncated


def test_runaway_query_times_out(sample_db):
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    with pytest.raises(SQLTimeoutError, match="stopped"):
        run_sql(endless, sample_db, timeout_seconds=0.2)


def test_execution_errors_are_reported_for_repair(sample_db):
    with pytest.raises(SQLExecutionError, match="no such column: revnue"):
        run_sql("SELECT revnue FROM financials", sample_db)


def test_missing_database_is_a_setup_error(tmp_path):
    with pytest.raises(DataNotReadyError, match="analyst.data.load"):
        run_sql("SELECT 1", tmp_path / "missing.db")


# ---------------------------------------------------------------------------
# Schema description and formatting
# ---------------------------------------------------------------------------
def test_schema_lists_exact_category_values_and_ranges(sample_db):
    schema = describe_schema(sample_db)
    assert f"Table financials ({ROWS} rows)" in schema
    assert "ticker TEXT: values: 'BFA', 'KRO'" in schema
    assert "unit TEXT: values: 'USD', 'USD/shares'" in schema
    assert "fiscal_year INTEGER: range 2022 to 2024" in schema


def test_schema_profiles_metrics_and_each_companys_coverage(sample_db):
    schema = describe_schema(sample_db)
    # Which unit each metric has (EPS is per share), and how many companies report it.
    assert "eps_diluted (USD/shares; 2)" in schema and "gross_profit (USD; 1)" in schema
    # Fiscal years per company, and the metrics a company doesn't report: a
    # query for BFA's gross profit would otherwise silently return nothing.
    assert "KRO = Kestrel Ridge Outdoor Co.: FY 2022, 2023, 2024" in schema
    assert "BFA = Bluefin Analytics Inc.: FY 2024; no gross_profit" in schema


def test_catalog_labels_show_the_values_a_planner_needs(sample_db):
    ((table, rows, labels),) = table_catalog(sample_db)
    assert (table, rows) == ("financials", ROWS)
    assert "ticker [BFA|KRO]" in labels and "fiscal_year 2022-2024" in labels
    assert "metric [eps_diluted|gross_profit|net_income|rd_expense|revenue|total_assets]" in labels
    assert (
        "value" in labels and "company [Bluefin Analytics Inc.|Kestrel Ridge Outdoor Co.]" in labels
    )


def test_result_text_formats_floats_compactly(sample_db):
    r = run_sql(
        "SELECT fiscal_year, ROUND(value / 1e6, 1) AS revenue_musd FROM financials "
        "WHERE ticker = 'KRO' AND metric = 'revenue' ORDER BY fiscal_year",
        sample_db,
    )
    expected = [
        f"{year} | {FACTS[('KRO', year)]['revenue'] / 1e6:.1f}" for year in (2022, 2023, 2024)
    ]
    assert r.to_text().splitlines() == [
        "fiscal_year | revenue_musd",
        *expected,
    ]  # "412.6", not "412.60"
