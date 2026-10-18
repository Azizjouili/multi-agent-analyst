"""Load every CSV in data/datasets/ into a SQLite database (one table per file).

The main one is financials.csv, the XBRL line items written by
scripts/fetch_edgar.py: one row per (company, fiscal year, metric).

    uv run python -m analyst.data.load [--datasets-dir DIR] [--db PATH]

Why SQLite:
- it's in the Python standard library: no server and no extra dependency;
- the database is a single file, easy to ship in the Docker image;
- SQL is the query language LLMs write most reliably;
- it can open a database in read-only mode, so the *engine* refuses writes,
  which is the strongest of our SQL guardrails (see tools/sql.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path

from analyst.config import get_settings


def _infer_type(values: list[str]) -> str:
    """Pick the narrowest SQLite type that fits every non-empty value.

    Why bother: CSV has no types. Stored as TEXT, "value" would sort as
    text ("100" < "99") and SUM() would behave unexpectedly. Real numeric
    columns make the SQL the LLM writes behave as intended.
    """
    present = [v for v in values if v != ""]
    for sql_type, cast in (("INTEGER", int), ("REAL", float)):
        try:
            for v in present:
                cast(v)
            return sql_type
        except ValueError:
            continue
    return "TEXT"


def _identifier(name: str) -> str:
    # Table and column names come from file names and CSV headers. Restrict
    # them to safe characters and quote them, so an odd header can't break
    # (or inject into) the CREATE TABLE statement.
    clean = re.sub(r"\W+", "_", name.strip()).strip("_").lower() or "col"
    return f'"{clean}"'


def load_csv(conn: sqlite3.Connection, path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    columns = list(zip(*rows, strict=True)) if rows else [[] for _ in header]
    types = [_infer_type(list(col)) for col in columns]
    casts = {"INTEGER": int, "REAL": float, "TEXT": str}

    table = _identifier(path.stem)
    col_defs = ", ".join(f"{_identifier(h)} {t}" for h, t in zip(header, types, strict=True))
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} ({col_defs})")

    placeholders = ", ".join("?" for _ in header)
    typed_rows = [
        tuple(None if v == "" else casts[t](v) for v, t in zip(row, types, strict=True))
        for row in rows
    ]
    conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", typed_rows)
    return len(typed_rows)


def load_datasets(datasets_dir: Path, db_path: Path) -> dict[str, int]:
    """Build the database from scratch and return {table: row count}."""
    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(
            f"No CSV files in {datasets_dir}. Fetch the SEC data first: "
            "uv run python scripts/fetch_edgar.py"
        )

    # Build into a temporary file, then swap it in with one atomic rename. A
    # query running during a reload sees either the old database or the new
    # one, never a half-loaded table.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)
    counts: dict[str, int] = {}
    conn = sqlite3.connect(tmp_path)
    try:
        with conn:  # one transaction: all tables load, or none do
            for path in csv_files:
                counts[_identifier(path.stem).strip('"')] = load_csv(conn, path)
    finally:
        conn.close()
    os.replace(tmp_path, db_path)
    return counts


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Load CSV datasets into SQLite.")
    parser.add_argument("--datasets-dir", type=Path, default=s.datasets_dir)
    parser.add_argument("--db", type=Path, default=s.db_path)
    args = parser.parse_args(argv)

    counts = load_datasets(args.datasets_dir, args.db)
    print(f"Loaded {len(counts)} table(s) into {args.db}")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")


if __name__ == "__main__":
    main()
