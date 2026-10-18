import socket

import pytest

from analyst.config import get_settings
from analyst.rag.index import build_index
from tests.fakes import BagOfWordsEmbedder


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Keep every test independent of the developer's real .env.

    Why: tests must pass on a fresh clone or in CI with no API keys, and must
    never call a paid or rate-limited API by accident. We force the fake
    provider and clear the cached settings around each test. Human review is
    off too: no one is there to answer, so the gate auto-approves. Tests of the
    gate itself (test_human_gate.py) turn it back on explicitly.
    """
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("HUMAN_REVIEW", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Fail any test that tries to reach the real network.

    Why: "the tests don't use the network" should be guaranteed, not assumed.
    A forgotten mock would otherwise make the suite slow, flaky, dependent on
    DuckDuckGo's mood, and impossible to run offline or in CI. Loopback stays
    allowed: Python on Windows uses a local socket pair internally. Tests
    marked `live` or `model` opt out.
    """
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("model"):
        return
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise RuntimeError(f"network access is blocked in tests (connect to {address!r})")
        return real_connect(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _LOOPBACK and host is not None:
            raise RuntimeError(f"network access is blocked in tests (DNS lookup of {host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory):
    """The synthetic company data, generated once per test session into a temp dir."""
    from scripts import make_sample_data as gen

    out = tmp_path_factory.mktemp("data")
    return out, gen.generate(out)


@pytest.fixture(scope="session")
def sample_db(sample_data, tmp_path_factory):
    """The sample CSV loaded into a temporary SQLite database."""
    from analyst.data.load import load_datasets

    out, _ = sample_data
    db_path = tmp_path_factory.mktemp("db") / "analyst.db"
    load_datasets(out / "datasets", db_path)
    return db_path


@pytest.fixture(scope="session")
def sample_index(sample_data, tmp_path_factory):
    """A vector index over the sample PDFs, built with the offline fake embedder."""
    out, _ = sample_data
    index_dir = tmp_path_factory.mktemp("index")
    build_index(out / "docs", index_dir, BagOfWordsEmbedder(), chunk_size=800, chunk_overlap=120)
    return index_dir
