"""Langfuse tracing, injected once per graph run through the RunnableConfig.

Why the config and not each agent: LangChain passes a run's `callbacks` down
to every child runnable (each graph node, and every LLM call made inside a
node, including the parallel specialists started by `Send`). So one handler
added at `graph.invoke(..., config)` sees the whole run as a single nested
trace: run -> node -> LLM call, with prompts, outputs, latency and tokens.
Agents stay unaware of tracing, and none of them can forget to report.

Optional by design: tracing turns on only when both Langfuse keys are set.
Otherwise `tracing_config()` returns {} and the run behaves exactly as before,
so the project works without a Langfuse account and tests never send data.

What a run looks like in the Langfuse UI:
  - trace name = the research question (what you'd search for)
  - tag        = the thread_id (the id the CLI prints, for exact lookup)
  - session    = the thread_id too. A run paused at the human gate makes one
                 trace up to the pause and another per resume; the session
                 shows them together, in order.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from analyst.config import Settings, get_settings

log = logging.getLogger("analyst.tracing")

TRACE_NAME_CHARS = 200  # a long question shouldn't become an unreadable trace title


@lru_cache(maxsize=4)
def _client(public_key: str, secret_key: str, base_url: str) -> Any:
    """One Langfuse client per key set, reused by every run.

    Why cache it: the client owns a background exporter thread that batches
    spans and sends them asynchronously. Creating one per run would start a
    new thread each time and lose the batching.
    """
    # Imported lazily so that code paths with tracing off never load the SDK.
    from langfuse import Langfuse

    # Keys are passed explicitly, from our Settings (which read .env), instead
    # of relying on the SDK finding them in os.environ.
    return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)


def get_callback_handler(settings: Settings | None = None) -> BaseCallbackHandler | None:
    """A fresh Langfuse callback handler, or None when tracing is off.

    A new handler per run, sharing the cached client: the handler keeps
    per-run bookkeeping (open spans by run id), so concurrent runs, such as
    two API requests, shouldn't share one.
    """
    settings = settings or get_settings()
    if not settings.langfuse_enabled:
        return None
    from langfuse.langchain import CallbackHandler

    _client(
        settings.langfuse_public_key,
        settings.langfuse_secret_key.get_secret_value(),
        settings.langfuse_host,
    )
    # The handler finds the client registered above through its public key.
    return CallbackHandler(public_key=settings.langfuse_public_key)


def tracing_config(
    question: str, thread_id: str, settings: Settings | None = None
) -> dict[str, Any]:
    """The tracing part of a run's RunnableConfig: {} when tracing is off."""
    handler = get_callback_handler(settings)
    if handler is None:
        return {}
    name = " ".join(question.split())[:TRACE_NAME_CHARS]
    return {
        "callbacks": [handler],
        # Root span name, the fallback the trace name is taken from.
        "run_name": name,
        # The Langfuse handler reads these special metadata keys on the root
        # run and applies them to the whole trace.
        "metadata": {
            "langfuse_trace_name": name,
            "langfuse_tags": [thread_id],
            "langfuse_session_id": thread_id,
        },
    }


def flush(settings: Settings | None = None) -> None:
    """Send buffered spans now. Call before a short-lived process exits.

    The exporter sends in the background, in batches. The SDK also flushes at
    exit, but an explicit flush makes sure a CLI run's last spans (finalize,
    the human-gate pause) arrive.
    """
    settings = settings or get_settings()
    if not settings.langfuse_enabled:
        return
    try:
        _client(
            settings.langfuse_public_key,
            settings.langfuse_secret_key.get_secret_value(),
            settings.langfuse_host,
        ).flush()
    except Exception:  # noqa: BLE001 - tracing must never break a run
        log.warning("tracing: flush to Langfuse failed", exc_info=True)
