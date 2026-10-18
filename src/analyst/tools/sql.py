"""Read-only SQL tool for LLM-written queries.

Running SQL an LLM wrote is running untrusted code, so this uses defense in
depth: four independent layers, any one of which blocks a write.

1. Static validation (regex): one statement, starting with SELECT/WITH, no
   write keywords. Easy to bypass in theory, so it isn't the security
   boundary. Its job is to give the LLM a clear, fixable error message.
2. Read-only connection (`mode=ro` + `PRAGMA query_only`): the SQLite
   *engine* refuses to modify the file, whatever the SQL says.
3. Authorizer callback: SQLite's own parser reports every action a statement
   will take (read a column, call a function, write, attach...), and we allow
   only reads. That covers what a regex can miss, without parsing SQL
   ourselves.
4. Resource limits: a row cap (fetchmany) and a timeout (a progress handler
   that aborts long queries).

Each call opens and closes its own connection. That's cheap for SQLite, and
it avoids sharing connections across the threads LangGraph uses for parallel
branches (sqlite3 connections aren't thread-safe by default).
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class SQLToolError(Exception):
    """Base class for errors that are reported back to the LLM so it can fix its query."""


class SQLGuardError(SQLToolError):
    """The query was rejected before execution (not a single read-only SELECT)."""


class SQLExecutionError(SQLToolError):
    """SQLite rejected or failed the query (syntax error, unknown column, not authorized...)."""


class SQLTimeoutError(SQLExecutionError):
    """The query ran longer than the timeout and was aborted."""


class DataNotReadyError(RuntimeError):
    """The database file doesn't exist yet: a setup problem, not something the LLM can fix."""


# ---------------------------------------------------------------------------
# Layer 1: static validation
# ---------------------------------------------------------------------------
# One pass, leftmost match wins, so a "--" inside a string literal is not
# mistaken for a comment, and a quote inside a comment doesn't open a string.
_LITERALS_AND_COMMENTS = re.compile(
    r"'(?:[^']|'')*'"  # 'string literal'
    r'|"(?:[^"]|"")*"'  # "quoted identifier"
    r"|--[^\n]*"  # -- line comment
    r"|/\*.*?\*/",  # /* block comment */
    re.DOTALL,
)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|upsert|merge|drop|create|alter|truncate|attach|detach|pragma"
    r"|vacuum|reindex|analyze|begin|commit|rollback|savepoint|release|load_extension)\b"
    # REPLACE() is a harmless string function, so block only the statement form.
    r"|\breplace\s+into\b",
    re.IGNORECASE,
)


def _mask(sql: str) -> str:
    """Blank out literals and comments so keywords inside them are ignored."""

    def repl(m: re.Match) -> str:
        token = m.group()
        return "''" if token[0] in "'\"" else " "

    return _LITERALS_AND_COMMENTS.sub(repl, sql)


def validate_sql(sql: str) -> str:
    """Return the query (without a trailing ';') or raise SQLGuardError."""
    masked = _mask(sql).strip()
    statements = [s for s in masked.split(";") if s.strip()]
    if not statements:
        raise SQLGuardError("Empty query.")
    if len(statements) > 1:
        raise SQLGuardError(
            "Only one statement is allowed; remove the extra ';'-separated statements."
        )
    first_word = statements[0].split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise SQLGuardError(
            f"Only SELECT queries are allowed (the query starts with {first_word.upper()})."
        )
    if m := _FORBIDDEN.search(masked):
        raise SQLGuardError(
            f"Only read-only SELECT queries are allowed (found {m.group().upper()})."
        )
    return sql.strip().rstrip(";").strip()


# ---------------------------------------------------------------------------
# Layers 2 + 3: read-only connection with an authorizer
# ---------------------------------------------------------------------------
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,  # run a SELECT
    sqlite3.SQLITE_READ,  # read a column
    sqlite3.SQLITE_FUNCTION,  # call a function (SUM, ROUND, ...)
    sqlite3.SQLITE_RECURSIVE,  # WITH RECURSIVE
}


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None
) -> int:
    if action == sqlite3.SQLITE_FUNCTION and arg2 and arg2.lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@contextmanager
def open_readonly(db_path: Path, guarded: bool = True) -> Iterator[sqlite3.Connection]:
    """Open the database read-only. `guarded=False` skips the authorizer and is
    only for our own fixed schema-inspection queries, never for LLM SQL."""
    if not db_path.exists():
        raise DataNotReadyError(
            f"No database at {db_path}. Build it with: uv run python -m analyst.data.load"
        )
    # mode=ro in the URI: SQLite opens the file read-only at the OS level.
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")  # set before the authorizer, which denies PRAGMA
        if guarded:
            conn.set_authorizer(_authorizer)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Layer 4 + the public tool
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool  # True if more rows existed than the cap allowed
    elapsed_ms: float

    def to_text(self, max_rows: int | None = None) -> str:
        """A compact pipe-separated table: the format the LLM reads results in."""
        shown = self.rows if max_rows is None else self.rows[:max_rows]
        lines = [" | ".join(self.columns)]
        lines += [" | ".join(_fmt(v) for v in row) for row in shown]
        hidden = len(self.rows) - len(shown)
        if hidden > 0 or self.truncated:
            lines.append(
                f"... ({'more than ' if self.truncated else ''}{len(self.rows)} rows total)"
            )
        return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        # Two decimals at most, no trailing zeros: 412.60 -> "412.6", 3.0 -> "3"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def run_sql(
    sql: str, db_path: Path, max_rows: int = 500, timeout_seconds: float = 5.0
) -> SQLResult:
    """Validate and run one read-only query. Raises SQLToolError subclasses on failure."""
    query = validate_sql(sql)
    with open_readonly(db_path) as conn:
        deadline = time.monotonic() + timeout_seconds
        # SQLite calls this handler every 1,000 virtual-machine steps; a
        # non-zero return aborts the query. This interrupts a runaway query
        # mid-execution, in-process, without killing a thread.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        started = time.perf_counter()
        try:
            cursor = conn.execute(query)
            # Fetch one row past the cap: that's how we know the result was truncated.
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise SQLTimeoutError(
                    f"Query exceeded {timeout_seconds:g}s and was stopped. Aggregate more, or "
                    "avoid joins that multiply rows."
                ) from exc
            raise SQLExecutionError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:  # e.g. "not authorized" from the authorizer
            raise SQLExecutionError(str(exc)) from exc
        columns = [d[0] for d in cursor.description or []]

    return SQLResult(
        sql=query,
        columns=columns,
        rows=rows[:max_rows],
        truncated=len(rows) > max_rows,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Schema description for the LLM
# ---------------------------------------------------------------------------
def table_catalog(db_path: Path) -> list[tuple[str, int, list[str]]]:
    """[(table, row count, column labels)]: a compact overview for the supervisor.

    The data agent gets the full profile (describe_schema). The planner only
    needs to know which tables and dimensions exist to write a sensible task.
    A label is the column name, plus its values when there are few of them
    ("ticker [AAPL|AMZN|...]") and its range for a year column
    ("fiscal_year 2023-2026"). Why: in a long-format table such as financials,
    the *rows* say what data exists (which companies, which metrics), and a
    planner that sees only "metric, value" can't tell that revenue is there
    but gross profit is missing for Amazon.
    """
    with open_readonly(db_path, guarded=False) as conn:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        return [
            (
                t,
                conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0],
                [
                    _column_label(conn, t, row[1], row[2])
                    for row in conn.execute(f'PRAGMA table_info("{t}")')
                ],
            )
            for t in tables
        ]


CATALOG_MAX_VALUES = 12
CATALOG_MAX_CHARS = 200  # longer value lists (e.g. 12 full company names) are left out


def _column_label(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> str:
    t, c = f'"{table}"', f'"{col}"'
    if col_type == "TEXT":
        values = [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL ORDER BY {c} LIMIT ?",
                (CATALOG_MAX_VALUES + 1,),
            )
        ]
        listed = "|".join(map(str, values))
        if values and len(values) <= CATALOG_MAX_VALUES and len(listed) <= CATALOG_MAX_CHARS:
            return f"{col} [{listed}]"
    elif col_type == "INTEGER" and "year" in col.lower():
        lo, hi = conn.execute(f"SELECT MIN({c}), MAX({c}) FROM {t}").fetchone()
        if lo is not None:
            return f"{col} {lo}-{hi}" if lo != hi else f"{col} {lo}"
    return col


def describe_schema(db_path: Path, max_values: int = 12) -> str:
    """Tables, columns, types and a profile of each column's values.

    Why profile values and not just list columns: to write
    `WHERE region = 'Asia-Pacific'` the model must know the exact spelling
    (not 'APAC'), and to filter on years it should know which years exist.
    Showing distinct values and ranges prevents a whole class of
    "query ran but returned nothing" errors.

    Long-format tables (one row per company, year and metric, like the XBRL
    financials) get two more profiles; see _long_format_notes.
    """
    out: list[str] = []
    with open_readonly(db_path, guarded=False) as conn:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        for table in tables:
            t = f'"{table}"'
            n_rows = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            out.append(f"Table {table} ({n_rows} rows)")
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({t})")]
            for _, col, col_type, *_ in conn.execute(f"PRAGMA table_info({t})"):
                c = f'"{col}"'
                if col_type == "TEXT":
                    values = [
                        r[0]
                        for r in conn.execute(
                            f"SELECT DISTINCT {c} FROM {t} ORDER BY {c} LIMIT ?", (max_values + 1,)
                        )
                    ]
                    if len(values) <= max_values:
                        profile = "values: " + ", ".join(f"'{v}'" for v in values)
                    else:
                        profile = "e.g. " + ", ".join(f"'{v}'" for v in values[:3])
                else:
                    lo, hi = conn.execute(f"SELECT MIN({c}), MAX({c}) FROM {t}").fetchone()
                    profile = f"range {_fmt(lo)} to {_fmt(hi)}"
                out.append(f"  - {col} {col_type}: {profile}")
            out.extend(_long_format_notes(conn, t, columns))
    return "\n".join(out)


def _long_format_notes(conn: sqlite3.Connection, t: str, columns: list[str]) -> list[str]:
    """Which metrics and which fiscal years each company actually has.

    Why: in a long table, "SELECT value WHERE metric = 'gross_profit' AND
    ticker = 'AMZN'" is valid SQL that silently returns nothing (Amazon
    reports no gross profit line). And fiscal years differ by company:
    Microsoft's latest is FY2026 (ended June 2026), Apple's FY2025. Listing
    both up front lets the model write a query that works the first time,
    and tells it the unit of each metric (EPS is USD per share, not USD).
    """
    notes: list[str] = []
    if {"metric", "unit", "ticker"} <= set(columns):
        notes.append("  Metrics (unit; companies reporting it):")
        for metric, unit, n in conn.execute(
            f"SELECT metric, unit, COUNT(DISTINCT ticker) FROM {t} "
            "GROUP BY metric, unit ORDER BY metric"
        ):
            notes.append(f"    {metric} ({unit}; {n})")
    if {"ticker", "fiscal_year"} <= set(columns):
        has_metric, has_company = "metric" in columns, "company" in columns
        all_metrics = (
            {r[0] for r in conn.execute(f"SELECT DISTINCT metric FROM {t}")}
            if has_metric
            else set()
        )
        notes.append("  Companies (fiscal years present; metrics they don't report):")
        for (ticker,) in conn.execute(f"SELECT DISTINCT ticker FROM {t} ORDER BY ticker"):
            years = [
                str(r[0])
                for r in conn.execute(
                    f"SELECT DISTINCT fiscal_year FROM {t} WHERE ticker = ? ORDER BY 1", (ticker,)
                )
            ]
            line = f"    {ticker}"
            if has_company:
                line += (
                    " = "
                    + conn.execute(
                        f"SELECT company FROM {t} WHERE ticker = ? LIMIT 1", (ticker,)
                    ).fetchone()[0]
                )
            line += f": FY {', '.join(years)}"
            if has_metric:
                present = {
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT metric FROM {t} WHERE ticker = ?", (ticker,)
                    )
                }
                if missing := sorted(all_metrics - present):
                    line += f"; no {', '.join(missing)}"
            notes.append(line)
    return notes
