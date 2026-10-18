# syntax=docker/dockerfile:1.7
#
# Two stages:
#   builder  installs the dependencies with uv and downloads the embedding model
#   runtime  copies only the virtualenv, the model and the source: no uv, no
#            build tools, no caches. Smaller image, smaller attack surface.
#
#   docker compose build          (or: docker build -t multi-agent-analyst .)
#
# No secrets are baked in: .env is excluded by .dockerignore, and keys come
# from the environment at `docker run` / `docker compose up` time.

ARG PYTHON_VERSION=3.12

# uv ships as a static binary in its own image; pinned so builds are repeatable.
FROM ghcr.io/astral-sh/uv:0.12.22 AS uv

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv

# UV_COMPILE_BYTECODE: compile .pyc now, so the first start is faster.
# UV_LINK_MODE=copy: the uv cache is a separate mount, so hard links can't work.
# UV_PYTHON_DOWNLOADS=0: use the image's Python; the runtime stage has the same one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, project second. This layer only rebuilds when
# pyproject.toml or uv.lock change, not on every source edit, and it's the slow
# one (torch). --frozen: install exactly what uv.lock says, which includes the
# CPU-only torch wheel (pyproject pins torch to the PyTorch CPU index), so no
# 2 GB of CUDA libraries. --no-dev skips pytest/ruff; the sample-data group
# (reportlab) lets the indexer generate the demo PDFs.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group sample-data --no-install-project

# Then the project itself. Installed editable, pointing at /app/src. That
# matters: config.py finds data/ relative to its own file, so the source must
# live at /app/src for data/ to resolve to /app/data (the compose volume).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group sample-data

# Download the embedding model at build time, into a known folder that the
# runtime stage copies. With HF_HUB_OFFLINE=1 at runtime, the container never
# reaches Hugging Face: it starts fast, works offline, and can't silently pick
# up a changed model. Must match EMBEDDING_MODEL at runtime.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Non-root user: if anything in the app were exploited (it fetches untrusted web
# pages), the attacker gets an unprivileged account, not root in the container.
# UID 1000 matches the usual first user on a Linux host, so files it writes to
# the data/ bind mount stay owned by you.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Same paths as in the builder: the virtualenv's scripts and the editable
# install hard-code /app/.venv and /app/src.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /opt/hf /opt/hf
COPY scripts ./scripts
RUN mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 8000

# Lets `docker ps` and compose show whether the API actually answers, not just
# whether the process exists. Uses Python: the slim image has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# 0.0.0.0 inside the container: Docker's port mapping must be able to reach it.
# Exposure to the outside is decided by the port mapping (see docker-compose.yml).
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
