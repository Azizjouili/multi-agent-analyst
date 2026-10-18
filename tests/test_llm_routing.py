"""Tests for per-step model routing (fast vs strong tier)."""

from langchain_core.runnables import Runnable
from pydantic import SecretStr

from analyst.agents.rag import RagAgent
from analyst.agents.web_research import WebResearchAgent
from analyst.config import Settings
from analyst.llm import FakeChatModel, get_llm, model_for
from analyst.rag.retriever import Retriever
from analyst.rag.vector_store import VectorStore
from analyst.tools.web_search import SearchResult
from tests.fakes import BagOfWordsEmbedder, FakeSearch, FakeSite, html_page


def _gemini_settings(**kw):
    return Settings(
        llm_provider="gemini",
        google_api_key=SecretStr("test-key"),
        gemini_model="gemini-strong-test",
        gemini_fast_model="gemini-fast-test",
        _env_file=None,
        **kw,
    )


def test_tiers_map_to_configured_models():
    s = _gemini_settings()
    assert model_for("fast", s) == "gemini-fast-test"
    assert model_for("strong", s) == "gemini-strong-test"
    # Building the clients makes no network call (the network is blocked in tests).
    assert get_llm(s, tier="fast").model.endswith("gemini-fast-test")
    assert get_llm(s, tier="strong").model.endswith("gemini-strong-test")
    assert get_llm(s).model.endswith("gemini-strong-test")  # strong is the default


def test_each_model_gets_its_own_shared_rate_limiter():
    s = _gemini_settings()
    fast_a, fast_b = get_llm(s, tier="fast"), get_llm(s, tier="fast")
    strong = get_llm(s, tier="strong")
    assert fast_a.rate_limiter is fast_b.rate_limiter  # shared across instances of a model
    assert fast_a.rate_limiter is not strong.rate_limiter  # separate per model (separate quotas)


def test_rag_agent_plans_on_fast_and_answers_on_strong(sample_index):
    fast = FakeChatModel(responses=[{"queries": ["risk factors"], "years": [2024]}])
    strong = FakeChatModel(responses=["Tariffs are a risk [D1]."])
    retriever = Retriever(VectorStore.load(sample_index), BagOfWordsEmbedder())
    agent = RagAgent(
        retriever=retriever,
        llm=strong,
        fast_llm=fast,
        settings=Settings(llm_provider="fake", _env_file=None),
    )

    agent(
        {
            "question": "Q",
            "subtask": {"id": "t", "agent": "rag", "instruction": "Key risks in 2024"},
        }
    )
    assert len(fast.history) == 1  # query rewriting
    assert len(strong.history) == 1  # grounded answer


def test_web_agent_plans_and_selects_on_fast_and_answers_on_strong():
    urls = [f"https://site{i}.example.com/a" for i in range(4)]
    site = FakeSite({u: html_page(f"Page {i}", f"<p>Fact {i}.</p>") for i, u in enumerate(urls)})
    fast = FakeChatModel(responses=[{"queries": ["q"]}, {"selected": ["R1"]}])
    strong = FakeChatModel(responses=["Fact 0 [W1]."])
    agent = WebResearchAgent(
        search=FakeSearch([SearchResult(f"r{i}", u, "s") for i, u in enumerate(urls)]),
        fetcher=site.fetcher(),
        llm=strong,
        fast_llm=fast,
        settings=Settings(llm_provider="fake", _env_file=None, web_max_pages=1),
    )
    agent(
        {"question": "Q", "subtask": {"id": "t", "agent": "web", "instruction": "Industry context"}}
    )
    assert len(fast.history) == 2  # search queries + page selection
    assert len(strong.history) == 1  # grounded answer


# ---------------------------------------------------------------------------
# Strong -> fast fallback
# ---------------------------------------------------------------------------
def test_strong_tier_falls_back_to_fast_on_quota_errors():
    from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

    llm = get_llm(_gemini_settings(), tier="strong")
    assert llm.model.endswith("gemini-strong-test")  # attributes still reach the primary model
    assert llm.fallbacks[0].model.endswith("gemini-fast-test")
    assert set(llm.exceptions_to_handle) == {GoogleRateLimitError, GoogleAPIError}


def test_fast_tier_and_disabled_setting_have_no_fallback():
    assert not hasattr(get_llm(_gemini_settings(), tier="fast"), "fallbacks")
    off = get_llm(_gemini_settings(llm_fallback_to_fast=False), tier="strong")
    assert not hasattr(off, "fallbacks")


class _QuotaExhausted(FakeChatModel):
    """Fails like a strong model whose daily quota is used up."""

    def _generate(self, *args, **kwargs):
        from langchain_google_genai.chat_models import GoogleRateLimitError

        raise GoogleRateLimitError("429 RESOURCE_EXHAUSTED")

    # The return annotation is load-bearing: RunnableWithFallbacks only routes a
    # method through the fallback when its type hints say it returns a Runnable.
    def with_structured_output(self, schema, **kwargs) -> Runnable:
        from langchain_core.runnables import RunnableLambda
        from langchain_google_genai.chat_models import GoogleRateLimitError

        def fail(_):
            raise GoogleRateLimitError("429 RESOURCE_EXHAUSTED")

        return RunnableLambda(fail)


def test_fallback_retries_plain_and_structured_calls_on_the_fast_model():
    from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: str

    fast = FakeChatModel(responses=["from the fast model", {"verdict": "accept"}])
    llm = _QuotaExhausted().with_fallbacks(
        [fast], exceptions_to_handle=(GoogleRateLimitError, GoogleAPIError)
    )
    assert llm.invoke("hi").text == "from the fast model"
    # with_structured_output goes through the fallback too (applied to both models).
    assert llm.with_structured_output(Verdict).invoke("judge") == Verdict(verdict="accept")
