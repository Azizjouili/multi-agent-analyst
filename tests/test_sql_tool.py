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
    validate_sql,
)


def _count(db):
    return run_sql("SELECT COUNT(*) FROM quarterly_sales", db).rows[0][0]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_loader_creates_typed_table(sample_db):
    with sqlite3.connect(sample_db) as conn:
        cols = {
            name: typ for _, name, typ, *_ in conn.execute("PRAGMA table_info(quarterly_sales)")
        }
    assert cols["year"] == "INTEGER" and cols["units_sold"] == "INTEGER"
    assert cols["revenue_usd"] == "REAL" and cols["region"] == "TEXT"
    assert _count(sample_db) == 288


def test_loaded_totals_match_the_generator(sample_db, sample_data):
    _, summaries = sample_data
    result = run_sql(
        "SELECT year, SUM(revenue_usd) FROM quarterly_sales GROUP BY year ORDER BY year", sample_db
    )
    for year, revenue in result.rows:
        assert revenue == pytest.approx(summaries[year].revenue)


def test_loader_without_csvs_explains_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="make_sample_data"):
        load_datasets(tmp_path, tmp_path / "x.db")


# ---------------------------------------------------------------------------
# Layer 1: static validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO quarterly_sales (year) VALUES (1999)",
        "UPDATE quarterly_sales SET revenue_usd = 0",
        "DELETE FROM quarterly_sales",
        "DROP TABLE quarterly_sales",
        "CREATE TABLE evil (x)",
        "ALTER TABLE quarterly_sales ADD COLUMN x",
        "REPLACE INTO quarterly_sales (year) VALUES (1)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA writable_schema = ON",
        "VACUUM",
        "SELECT 1; DROP TABLE quarterly_sales",
        "WITH x AS (SELECT 1) DELETE FROM quarterly_sales",
        "SELECT * FROM quarterly_sales WHERE 1 = 1; INSERT INTO quarterly_sales (year) VALUES (1)",
        "",
        "   ;  ",
    ],
)
def test_guard_rejects_anything_but_a_single_select(sql, sample_db):
    with pytest.raises(SQLGuardError):
        run_sql(sql, sample_db)
    assert _count(sample_db) == 288  # nothing changed


def test_guard_ignores_keywords_inside_literals_and_comments(sample_db):
    sql = "SELECT COUNT(*) FROM quarterly_sales WHERE region = 'DROP TABLE; DELETE' -- ; DROP"
    assert run_sql(sql, sample_db).rows == [(0,)]


def test_guard_allows_cte_trailing_semicolon_and_replace_function(sample_db):
    sql = (
        "WITH t AS (SELECT REPLACE(region, '-', ' ') AS r FROM quarterly_sales) "
        "SELECT COUNT(DISTINCT r) FROM t;"
    )
    assert run_sql(sql, sample_db).rows == [(3,)]
    assert validate_sql("SELECT 1;  ") == "SELECT 1"


# ---------------------------------------------------------------------------
# Layers 2 + 3: the engine blocks writes even if validation is bypassed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM quarterly_sales",
        "UPDATE quarterly_sales SET revenue_usd = 0",
        "CREATE TABLE evil (x)",
        "ATTACH DATABASE ':memory:' AS other",
    ],
)
def test_readonly_connection_blocks_writes_without_the_regex(sql, sample_db):
    # Call the engine directly, skipping validate_sql(), to prove the deeper
    # layers hold on their own.
    with open_readonly(sample_db) as conn, pytest.raises(sqlite3.DatabaseError):
        conn.execute(sql)
    assert _count(sample_db) == 288


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
        conn.execute("DELETE FROM quarterly_sales")
    conn.close()
    assert _count(copy) == 288


# ---------------------------------------------------------------------------
# Layer 4: row cap and timeout
# ---------------------------------------------------------------------------
def test_row_cap_truncates_and_flags(sample_db):
    capped = run_sql("SELECT * FROM quarterly_sales", sample_db, max_rows=50)
    assert len(capped.rows) == 50 and capped.truncated
    assert "more than 50 rows" in capped.to_text()

    full = run_sql("SELECT * FROM quarterly_sales", sample_db, max_rows=500)
    assert len(full.rows) == 288 and not full.truncated


def test_runaway_query_times_out(sample_db):
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    with pytest.raises(SQLTimeoutError, match="stopped"):
        run_sql(endless, sample_db, timeout_seconds=0.2)


def test_execution_errors_are_reported_for_repair(sample_db):
    with pytest.raises(SQLExecutionError, match="no such column: revnue"):
        run_sql("SELECT revnue FROM quarterly_sales", sample_db)


def test_missing_database_is_a_setup_error(tmp_path):
    with pytest.raises(DataNotReadyError, match="analyst.data.load"):
        run_sql("SELECT 1", tmp_path / "missing.db")


# ---------------------------------------------------------------------------
# Schema description and formatting
# ---------------------------------------------------------------------------
def test_schema_lists_exact_category_values_and_ranges(sample_db):
    schema = describe_schema(sample_db)
    assert "Table quarterly_sales (288 rows)" in schema
    assert "region TEXT: values: 'Asia-Pacific', 'Europe', 'North America'" in schema
    assert "year INTEGER: range 2022 to 2024" in schema


def test_result_text_formats_floats_compactly(sample_db, sample_data):
    _, summaries = sample_data
    r = run_sql(
        "SELECT year, ROUND(SUM(revenue_usd) / 1e6, 1) AS revenue_musd "
        "FROM quarterly_sales GROUP BY year ORDER BY year",
        sample_db,
    )
    expected = [f"{year} | {s.revenue / 1e6:.1f}" for year, s in sorted(summaries.items())]
    assert r.to_text().splitlines() == ["year | revenue_musd", *expected]  # "412.6", not "412.60"
