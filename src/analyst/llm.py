"""LLM provider factory.

Why a factory: agents call `get_llm()` and receive a langchain-core
`BaseChatModel`. They never import a provider SDK directly, so switching
providers (or using the offline fake in tests) is a config change, not a code
change.

Model routing: each call site asks for a *tier*, not a model name.
  - "fast"   (GEMINI_FAST_MODEL):  light, high-volume steps where a small model
                                   is enough: search-query planning, page selection.
  - "strong" (GEMINI_MODEL):       steps whose quality shows in the report:
                                   supervisor planning, SQL writing, grounded
                                   answers, writing, critiquing.
Why: free-tier quotas are counted per model per day, so spreading calls over
two models roughly doubles the daily budget, and the light steps get faster.

Run `uv run python -m analyst.llm` to check that your provider and API key work.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr

from analyst.config import Settings, get_settings


class LLMConfigError(RuntimeError):
    """Raised when the configured provider can't be built, e.g. a missing API key."""


# ---------------------------------------------------------------------------
# Fake provider (offline, deterministic)
# ---------------------------------------------------------------------------
class FakeChatModel(BaseChatModel):
    """A scripted chat model for tests and offline development.

    Why write our own instead of langchain's FakeListChatModel: our agents
    will use `.with_structured_output(Schema)` for plans and critiques, and the
    built-in fakes don't support it. This fake returns the scripted `responses`
    in order (wrapping around at the end):

    - a `str`  → returned as the message text
    - a `dict` → used as the structured output when called through
                 `with_structured_output`, or serialized to text otherwise

    Every input it receives is recorded in `history`, so tests can check what
    an agent actually sent (e.g. which excerpts reached the prompt).
    """

    responses: list[str | dict[str, Any]] = ["(fake response)"]
    _cursor: int = PrivateAttr(default=0)
    _history: list[Any] = PrivateAttr(default_factory=list)

    @property
    def history(self) -> list[Any]:
        return self._history

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _next(self) -> str | dict[str, Any]:
        item = self.responses[self._cursor % len(self.responses)]
        self._cursor += 1
        return item

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._history.append(messages)
        item = self._next()
        text = item if isinstance(item, str) else _to_json(item)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(
        self, schema: type[BaseModel] | dict[str, Any], **kwargs: Any
    ) -> Runnable:
        # Real models produce structured output through tool calling or JSON
        # mode. Here we just validate the next scripted dict against the
        # schema, which is enough to test the routing logic that depends on it.
        def _invoke(_input: Any) -> Any:
            self._history.append(_input)
            item = self._next()
            if not isinstance(item, dict):
                raise ValueError("FakeChatModel: structured output needs a dict response")
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(item)
            return item

        return RunnableLambda(_invoke)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        # Tool-using agents call bind_tools(). The fake ignores the tools; tests
        # script the responses directly.
        return self


def _to_json(item: dict[str, Any]) -> str:
    import json

    return json.dumps(item)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
Tier = Literal["fast", "strong"]

# One rate limiter per *model*, shared by every instance of that model in the
# process. Per model, because Gemini quotas are per model: a single shared
# limiter would slow the fast model down with the strong model's traffic.
# Shared across instances, because throttling each agent separately would let
# parallel agents exceed the model's limit together.
_LIMITERS: dict[str, InMemoryRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _rate_limiter(model: str, settings: Settings) -> InMemoryRateLimiter:
    with _LIMITERS_LOCK:
        if model not in _LIMITERS:
            # Token bucket: refills at RPM/60 tokens per second. max_bucket_size=1
            # spreads calls evenly instead of allowing a burst. That matters
            # when the supervisor fans out to three agents at the same moment.
            _LIMITERS[model] = InMemoryRateLimiter(
                requests_per_second=settings.llm_requests_per_minute / 60,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        return _LIMITERS[model]


def model_for(tier: Tier, settings: Settings) -> str:
    return settings.gemini_fast_model if tier == "fast" else settings.gemini_model


def get_llm(
    settings: Settings | None = None,
    *,
    tier: Tier = "strong",
    temperature: float | None = None,
    fake_responses: Iterable[str | dict[str, Any]] | None = None,
) -> BaseChatModel:
    """Build the chat model selected by `LLM_PROVIDER` for the given tier.

    `temperature` lets a single agent override the default. For example, the
    critic can use 0 so its judgments are as repeatable as possible.
    """
    settings = settings or get_settings()
    temp = settings.llm_temperature if temperature is None else temperature

    if settings.llm_provider == "fake":
        return FakeChatModel(responses=list(fake_responses or ["(fake response)"]))

    if settings.llm_provider == "gemini":
        if settings.google_api_key is None or not settings.google_api_key.get_secret_value():
            raise LLMConfigError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set. "
                "Create a free key at https://aistudio.google.com/apikey and put it in .env, "
                "or set LLM_PROVIDER=fake to run offline."
            )
        # Imported here, not at module top, so tests and the fake provider never
        # load the Google SDK.
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

        def build(model: str) -> ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.google_api_key.get_secret_value(),
                temperature=temp,
                max_retries=settings.llm_max_retries,
                rate_limiter=_rate_limiter(model, settings),
            )

        primary = build(model_for(tier, settings))
        fast_model = model_for("fast", settings)
        if tier == "strong" and settings.llm_fallback_to_fast and fast_model != primary.model:
            # Strong -> fast fallback. When the strong model's daily quota is used
            # up (429) or it's overloaded (503), the same call is retried on the
            # fast model: a slightly weaker answer instead of an ERROR finding.
            # Only these two errors, on purpose: a wrong model name (404) should
            # fail loudly, and a context overflow wouldn't fit the smaller model.
            # Attribute access (.model) still reaches the primary model, and
            # .with_structured_output() is applied to both models.
            return primary.with_fallbacks(
                [build(fast_model)], exceptions_to_handle=(GoogleRateLimitError, GoogleAPIError)
            )
        return primary

    raise LLMConfigError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


if __name__ == "__main__":
    # Quick live check of both tiers: `uv run python -m analyst.llm`
    s = get_settings()
    for t in ("fast", "strong"):
        name = model_for(t, s) if s.llm_provider == "gemini" else "fake"
        print(f"provider={s.llm_provider} tier={t} model={name}")
        # `.text`, not `.content`: Gemini 3 returns content as a list of blocks.
        print("  reply:", get_llm(s, tier=t).invoke("Reply with exactly: pong").text)
