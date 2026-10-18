"""Start the API server.

    uv run python scripts/serve.py                 # http://127.0.0.1:8000/docs
    uv run python scripts/serve.py --port 9000
    uv run python scripts/serve.py --host 0.0.0.0  # reachable from other machines

Paused runs live in the checkpoint database (CHECKPOINT_DB), so stopping and
restarting the server loses nothing: poll GET /runs/{thread_id} afterwards.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Multi-Agent Analyst API.")
    # 127.0.0.1 by default: the API has no authentication, so it shouldn't be
    # reachable from the network unless you ask for it explicitly.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes (dev)")
    args = parser.parse_args()

    # Agent progress logs ("writer: first draft ...") next to uvicorn's access log.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(name)s: %(message)s")
    # factory=True: uvicorn calls create_app() itself. With --reload it must
    # import the app by string, and the factory keeps imports side-effect free.
    uvicorn.run(
        "analyst.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
