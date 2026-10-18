"""Smoke tests for the project skeleton: config loading and the LLM factory."""

import os

import pytest
from pydantic import BaseModel

from analyst.config import Settings, get_settings
from analyst.llm import FakeChatModel, LLMConfigError, get_llm


def test_settings_defaults():
    s = get_settings()
    assert s.llm_provider == "fake"  # forced by conftest
    assert s.max_revisions >= 0
    assert s.docs_dir.name == "docs"


def test_langfuse_disabled_without_keys():
    assert get_settings().langfuse_enabled is False


def test_fake_llm_returns_scripted_text_in_order():
    llm = get_llm(fake_responses=["first", "second"])
    assert isinstance(llm, FakeChatModel)
    assert llm.invoke("hi").content == "first"
    assert llm.invoke("hi").content == "second"
    assert llm.invoke("hi").content == "first"  # wraps around


class Verdict(BaseModel):
    verdict: str
    score: int


def test_fake_llm_structured_output():
    llm = get_llm(fake_responses=[{"verdict": "accept", "score": 5}])
    result = llm.with_structured_output(Verdict).invoke("judge this")
    assert result == Verdict(verdict="accept", score=5)


def test_gemini_without_key_gives_clear_error():
    s = Settings(llm_provider="gemini", google_api_key=None, _env_file=None)
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        get_llm(s)


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs GOOGLE_API_KEY")
def test_gemini_live_ping():
    s = Settings(llm_provider="gemini")
    reply = get_llm(s).invoke("Reply with exactly: pong")
    assert "pong" in str(reply.content).lower()
