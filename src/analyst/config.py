"""Central, typed configuration.

Why pydantic-settings: every tunable (provider, model, budgets, keys) is
declared once with a type and a default. Values come from environment
variables or a local `.env` file, and bad values fail at startup with a clear
error instead of deep inside an agent run.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root = two levels above src/analyst/. Paths are anchored here so the
# app works no matter which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated env vars on the machine shouldn't crash startup
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["gemini", "fake"] = "gemini"
    # SecretStr keeps the key out of logs, reprs and tracebacks.
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"  # "strong" tier: answers, SQL, writing, critique
    # "fast" tier: light steps (query planning, page selection). See llm.py.
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    # Retry strong-tier calls on the fast model when the strong one hits its
    # quota (429) or is overloaded (503). See llm.get_llm.
    llm_fallback_to_fast: bool = True
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)
    # Free-tier Gemini allows only a few requests per minute. Throttling on the
    # client is cheaper than handling 429 errors and retrying.
    llm_requests_per_minute: float = Field(8, gt=0)
    llm_max_retries: int = Field(3, ge=0)

    # --- RAG ---------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    # Chunk size is in characters. all-MiniLM-L6-v2 reads at most 256 tokens
    # and silently drops the rest. About 800 chars is ~200 tokens, so a whole
    # chunk is embedded, and it's still big enough to hold a complete thought.
    chunk_size: int = Field(800, gt=0)
    # Overlap repeats the end of one chunk at the start of the next, so a fact
    # that falls on a chunk boundary is complete in at least one chunk.
    chunk_overlap: int = Field(120, ge=0)
    retrieval_k: int = Field(5, gt=0)
    # Which 10-K Items to index (see rag/loader.py for why Item 8 is left out
    # by default). Env: EDGAR_ITEMS=1,1A,3,7,7A,8
    # NoDecode: read the env var as plain "1,1A,7" rather than as JSON.
    edgar_items: Annotated[tuple[str, ...], NoDecode] = ("1", "1A", "3", "7", "7A")

    # --- Data analysis -----------------------------------------------------
    datasets_dir: Path = (
        PROJECT_ROOT / "data" / "datasets"
    )  # financials.csv (scripts/fetch_edgar.py)
    db_path: Path = PROJECT_ROOT / "data" / "analyst.db"
    # Guardrails for LLM-written SQL. The row cap keeps one careless
    # `SELECT *` from flooding the prompt. The timeout stops a runaway query
    # (e.g. an accidental cross join) from hanging the whole agent run.
    sql_max_rows: int = Field(500, gt=0)
    sql_timeout_seconds: float = Field(5.0, gt=0)

    # --- Web research --------------------------------------------------------
    web_max_results: int = Field(5, gt=0)  # search results per query
    web_max_pages: int = Field(3, gt=0)  # pages actually fetched and read per sub-task
    web_timeout_seconds: float = Field(10.0, gt=0)
    # Stop downloading after this many bytes. A malicious or huge page can't
    # exhaust memory or bandwidth.
    web_max_bytes: int = Field(2_000_000, gt=0)
    # Characters of extracted text kept per page, and how many of them reach
    # the prompt. Pages are long, so the most relevant passages are selected
    # to fit (see agents/web_research.py).
    web_max_page_chars: int = Field(30_000, gt=0)
    web_prompt_chars_per_page: int = Field(3_000, gt=0)
    web_respect_robots: bool = True
    web_user_agent: str = "MultiAgentAnalyst/0.1 (research demo; +https://github.com/)"

    # --- Loop budgets ------------------------------------------------------
    # Hard limits so the critic loops always end, even if the critic never
    # becomes satisfied.
    max_revisions: int = Field(2, ge=0)
    max_research_rounds: int = Field(1, ge=0)
    # Upper bound on sub-tasks per plan. Each one costs ~2-3 LLM calls, so this
    # is also a cost and quota control.
    max_subtasks: int = Field(6, ge=1)

    # --- Human-in-the-loop -------------------------------------------------
    # On: the graph pauses at the human gate until someone approves or rejects.
    # Off: the gate auto-approves, for batch jobs such as evals.
    human_review: bool = True
    # Where paused runs are saved. A file, not memory, so a run can wait across
    # process restarts and be resumed later by its thread id.
    checkpoint_db: Path = PROJECT_ROOT / "data" / "checkpoints.db"

    # --- API -----------------------------------------------------------------
    # Runs executing at the same time; extra runs wait their turn. Each run
    # makes dozens of LLM calls, and free-tier quotas are small, so starting
    # ten at once would only make all ten hit 429s.
    api_max_concurrent_runs: int = Field(2, ge=1)

    # --- Langfuse (optional) -----------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @field_validator("edgar_items", mode="before")
    @classmethod
    def _split_items(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(i.strip().upper() for i in value.split(",") if i.strip())
        return value

    @property
    def langfuse_enabled(self) -> bool:
        # Tracing is on only when both keys are set. Otherwise it's a silent
        # no-op, so the project runs without a Langfuse account.
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Why a cached function instead of a module-level global: tests can call
    `get_settings.cache_clear()` after changing env vars, and FastAPI can inject
    it as a dependency later.
    """
    return Settings()
