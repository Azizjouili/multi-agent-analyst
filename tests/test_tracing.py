"""Offline tests for Langfuse tracing: on/off by config, and propagation through the graph."""

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from analyst import tracing
from analyst.agents import placeholders
from analyst.agents.critic import CriticAgent
from analyst.agents.writer import WriterAgent
from analyst.config import Settings
from analyst.graph import CRITIC, DATA, RAG, SUPERVISOR, WEB, WRITER, build_graph, run_config
from analyst.llm import FakeChatModel

QUESTION = "Summarize the company's revenue trend and key risks"


def _settings(**kw):
    return Settings(llm_provider="fake", _env_file=None, **kw)


@pytest.fixture
def langfuse_settings():
    """Settings with (fake) Langfuse keys. Nothing is sent: no span is ever
    created in these tests, and the network is blocked anyway.

    No shutdown() in teardown, on purpose: Langfuse keeps one resource manager
    per public key for the whole process. Shutting it down and creating a new
    client with the same key reuses the stopped manager, and its flush() then
    waits forever. The cached client simply lives until the test session ends.
    """
    return _settings(langfuse_public_key="pk-lf-test", langfuse_secret_key="sk-lf-test")


# ---------------------------------------------------------------------------
# On / off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"langfuse_public_key": "pk-lf-test"},  # one key alone isn't enough
        {"langfuse_secret_key": "sk-lf-test"},
    ],
)
def test_no_callback_without_both_keys(keys):
    settings = _settings(**keys)
    assert tracing.get_callback_handler(settings) is None
    assert tracing.tracing_config(QUESTION, "t1", settings) == {}
    assert "callbacks" not in run_config("t1", QUESTION, settings)


def test_callback_present_with_both_keys(langfuse_settings):
    from langfuse.langchain import CallbackHandler

    config = run_config("t1", QUESTION, langfuse_settings)

    (handler,) = config["callbacks"]
    assert isinstance(handler, CallbackHandler)
    assert config["configurable"] == {"thread_id": "t1"}
    # Findable in the UI: named after the question, tagged and grouped by thread id.
    assert config["run_name"] == QUESTION
    assert config["metadata"] == {
        "langfuse_trace_name": QUESTION,
        "langfuse_tags": ["t1"],
        "langfuse_session_id": "t1",
    }
    # A new handler per run (per-run bookkeeping), one shared client.
    assert run_config("t2", QUESTION, langfuse_settings)["callbacks"][0] is not handler
    assert tracing._client.cache_info().currsize == 1


def test_lookups_without_a_question_carry_no_callbacks(langfuse_settings):
    assert run_config("t1", settings=langfuse_settings) == {"configurable": {"thread_id": "t1"}}


def test_long_questions_are_shortened_for_the_trace_name(langfuse_settings):
    config = tracing.tracing_config("why " * 200, "t1", langfuse_settings)
    assert len(config["metadata"]["langfuse_trace_name"]) == tracing.TRACE_NAME_CHARS


# ---------------------------------------------------------------------------
# Propagation: one handler in the run config reaches every node and LLM call
# ---------------------------------------------------------------------------
class RecordingHandler(BaseCallbackHandler):
    """Stands in for the Langfuse handler and records what it would trace."""

    def __init__(self):
        self.runs = {}  # run_id -> name
        self.chat_parents = []  # name of the node that made each chat-model call
        self.root_metadata = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.runs[run_id] = kwargs.get("name") or (serialized or {}).get("name")
        if parent_run_id is None:
            self.root_metadata = kwargs.get("metadata")

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self.chat_parents.append(self.runs.get(parent_run_id))


def test_one_handler_in_the_run_config_traces_every_node_and_llm_call(monkeypatch):
    recorder = RecordingHandler()
    monkeypatch.setattr(tracing, "get_callback_handler", lambda settings=None: recorder)

    settings = _settings(human_review=False)
    writer = WriterAgent(llm=FakeChatModel(responses=["# R\n\n- Point [S1].\n"]), settings=settings)
    judgment = {
        "scores": {"groundedness": 5, "coverage": 5, "clarity": 5},
        "verdict": "accept",
    }
    critic = CriticAgent(llm=FakeChatModel(responses=[judgment]), settings=settings)
    graph = build_graph(
        {
            SUPERVISOR: placeholders.supervisor,
            WEB: placeholders.web_research,
            RAG: placeholders.rag,
            DATA: placeholders.data_analysis,
            WRITER: writer,  # agents know nothing about tracing...
            CRITIC: critic,
        },
        settings=settings,
    )
    graph.invoke({"question": QUESTION}, run_config("t1", QUESTION, settings))

    names = set(recorder.runs.values())
    # ...yet every node was seen, including the specialists running in parallel via Send,
    assert {SUPERVISOR, WEB, RAG, DATA, WRITER, CRITIC} <= names
    # the root run carries the trace name and tags,
    assert QUESTION in names
    assert recorder.root_metadata["langfuse_tags"] == ["t1"]
    # and the writer's LLM call is nested under the writer node.
    assert recorder.chat_parents == [WRITER]
