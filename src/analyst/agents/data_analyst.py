"""Data-analysis agent: answers one sub-task by querying the local SQLite dataset.

Inside the node:
  1. schema    - profile the database (tables, columns, actual values); no LLM call
  2. plan      - the LLM writes 1-3 SELECT queries (structured output)
  3. execute   - run each through the guarded SQL tool; on an error, send the
                 error back to the LLM to repair the query (at most MAX_REPAIRS)
  4. interpret - the LLM answers ONLY from the result tables, citing [Q1], [Q2], ...
  5. resolve   - labels -> source ids; each source *is* the SQL query, so any
                 number in the report can be re-checked by re-running the query

Why fixed steps instead of a free-form tool-calling loop: the control flow is
explicit, the number of LLM calls is bounded (which matters on a free-tier rate
limit), and every step can be tested with a scripted fake LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from analyst.agents.citations import resolve_citations
from analyst.agents.common import short_error
from analyst.config import Settings, get_settings
from analyst.llm import get_llm
from analyst.prompts import DATA_ANSWER_SYSTEM, DATA_SQL_FIX_SYSTEM, DATA_SQL_SYSTEM
from analyst.state import Finding, Source, SpecialistInput, make_source_id
from analyst.tools.sql import SQLResult, SQLToolError, describe_schema, run_sql

log = logging.getLogger("analyst.agents.data")

MAX_REPAIRS = 2  # repair attempts per failed query, after the first try
PROMPT_ROWS = 60  # result rows shown to the LLM; the tool's cap (500) is a safety net
SNIPPET_CHARS = 1500


class SQLQuery(BaseModel):
    purpose: str = Field(description="What the query computes, e.g. 'Net revenue by year'")
    sql: str = Field(description="One SQLite SELECT statement")


class SQLPlan(BaseModel):
    queries: list[SQLQuery] = Field(min_length=1, max_length=3)


@dataclass
class QueryOutcome:
    label: str  # "Q1"
    query: SQLQuery  # the final version, after any repairs
    result: SQLResult | None
    error: str | None
    attempts: int

    @property
    def source(self) -> Source:
        assert self.result is not None
        # The id hashes the whitespace-normalized SQL, so the same query
        # written on different lines gets the same id.
        normalized = re.sub(r"\s+", " ", self.result.sql).strip()
        return {
            "id": make_source_id("data", normalized),
            "kind": "data",
            "title": f"SQL: {self.query.purpose}",
            "locator": self.result.sql,  # the citation is the query itself
            "snippet": self.result.to_text(PROMPT_ROWS)[:SNIPPET_CHARS],
        }


class DataAnalystAgent:
    """A LangGraph node: `DataAnalystAgent()(SpecialistInput) -> {"findings", "sources"}`."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        db_path: Path | None = None,
    ):
        self.settings = settings or get_settings()
        self.db_path = db_path or self.settings.db_path
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm(
                self.settings, tier="strong", temperature=0
            )  # SQL is not a light step
        return self._llm

    # -- the node ------------------------------------------------------------
    def __call__(self, inp: SpecialistInput) -> dict:
        task = inp["subtask"]
        # A missing database raises DataNotReadyError here, on purpose: that's
        # a setup mistake to fix, not something to hide in a finding.
        schema = describe_schema(self.db_path)

        try:
            plan = self._plan(schema, inp["question"], task["instruction"])
        except Exception as exc:  # noqa: BLE001 - LLM/parse failure: report it, don't crash the run
            log.exception("data[%s]: query planning failed", task["id"])
            return _result(
                task["id"], f"ERROR: could not plan SQL queries ({short_error(exc)}).", []
            )

        outcomes = [
            self._execute_with_repair(f"Q{i}", q, schema)
            for i, q in enumerate(plan.queries, start=1)
        ]
        succeeded = [o for o in outcomes if o.result is not None]
        for o in outcomes:
            log.info(
                "data[%s]: %s %s after %d attempt(s): %s",
                task["id"], o.label, "ok" if o.result else "FAILED", o.attempts, o.query.purpose,
            )  # fmt: skip
        if not succeeded:
            errors = "; ".join(f"{o.label}: {o.error}" for o in outcomes)
            return _result(task["id"], f"ERROR: every SQL query failed ({errors}).", [])

        try:
            answer = self._interpret(inp["question"], task["instruction"], succeeded)
        except Exception as exc:  # noqa: BLE001
            log.exception("data[%s]: interpretation failed", task["id"])
            return _result(
                task["id"],
                f"ERROR: could not interpret the query results ({short_error(exc)}).",
                [],
            )

        labels = {o.label: o.source["id"] for o in succeeded}
        summary, used_ids = resolve_citations(answer, labels)
        if not used_ids:
            log.warning("data[%s]: answer has no valid citations", task["id"])
        sources = [o.source for o in succeeded if o.source["id"] in used_ids]
        return _result(task["id"], summary, sources)

    # -- step 2: plan ----------------------------------------------------------
    def _plan(self, schema: str, question: str, instruction: str) -> SQLPlan:
        plan = self.llm.with_structured_output(SQLPlan).invoke(
            [
                SystemMessage(DATA_SQL_SYSTEM.format(schema=schema)),
                HumanMessage(f"Overall research question: {question}\nSub-task: {instruction}"),
            ]
        )
        if plan is None:
            raise ValueError("empty structured output")
        return plan

    # -- step 3: execute, repairing failures -----------------------------------
    def _execute_with_repair(self, label: str, query: SQLQuery, schema: str) -> QueryOutcome:
        error = None
        for attempt in range(1, MAX_REPAIRS + 2):
            try:
                result = run_sql(
                    query.sql,
                    self.db_path,
                    max_rows=self.settings.sql_max_rows,
                    timeout_seconds=self.settings.sql_timeout_seconds,
                )
                return QueryOutcome(label, query, result, None, attempt)
            except SQLToolError as exc:
                error = str(exc)
                log.info("data: %s attempt %d failed: %s", label, attempt, error)
                if attempt > MAX_REPAIRS:
                    break
                # Reflection on tool use: show the model its own query and the
                # exact error ("no such column: revnue"). Errors like that are
                # usually fixed in one try.
                try:
                    query = self._repair(schema, query, error)
                except Exception as repair_exc:  # noqa: BLE001
                    error = f"{error} (repair failed: {repair_exc})"
                    break
        return QueryOutcome(label, query, None, error, attempt)

    def _repair(self, schema: str, query: SQLQuery, error: str) -> SQLQuery:
        fixed = self.llm.with_structured_output(SQLQuery).invoke(
            [
                SystemMessage(DATA_SQL_FIX_SYSTEM.format(schema=schema)),
                HumanMessage(
                    f"Purpose: {query.purpose}\nFailed query:\n{query.sql}\n\nError: {error}"
                ),
            ]
        )
        if fixed is None:
            raise ValueError("empty structured output")
        return fixed

    # -- step 4: interpret -------------------------------------------------------
    def _interpret(self, question: str, instruction: str, outcomes: list[QueryOutcome]) -> str:
        blocks = "\n\n".join(
            f"[{o.label}] {o.query.purpose}\nSQL: {o.result.sql}\n"
            f"Result:\n{o.result.to_text(PROMPT_ROWS)}"
            for o in outcomes
            if o.result is not None
        )
        messages = [
            SystemMessage(DATA_ANSWER_SYSTEM),
            HumanMessage(
                f"Overall research question: {question}\nSub-task: {instruction}\n\n"
                f"Query results:\n\n{blocks}"
            ),
        ]
        return self.llm.invoke(messages).text.strip()


def _result(subtask_id: str, summary: str, sources: list[Source]) -> dict:
    finding: Finding = {
        "subtask_id": subtask_id,
        "agent": "data",
        "summary": summary,
        "source_ids": [s["id"] for s in sources],
    }
    return {"findings": [finding], "sources": {s["id"]: s for s in sources}}
