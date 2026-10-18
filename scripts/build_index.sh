#!/bin/sh
# Prepare data/ for the API: SEC corpus (if missing) -> SQLite -> vector index.
# Run by the `indexer` compose service:  docker compose run --rm indexer
set -eu

# Fetch the pinned 10-K filings and XBRL financials only when there are none,
# so documents you added to data/docs are never touched. Needs network access
# to sec.gov; set SEC_USER_AGENT to your name and email (see the README).
if [ -z "$(ls data/docs/*.htm* data/docs/*.pdf 2>/dev/null)" ]; then
    echo "data/docs is empty: fetching the SEC EDGAR corpus (data/corpus/manifest.yaml)"
    python scripts/fetch_edgar.py
fi

python -m analyst.data.load   # CSVs in data/datasets -> data/analyst.db
python -m analyst.rag.index   # 10-Ks (and PDFs) in data/docs -> data/index (baked-in model)
echo "Done: data/analyst.db and data/index are ready."
