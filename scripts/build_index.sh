#!/bin/sh
# Prepare data/ for the API: sample data (if missing) -> SQLite -> vector index.
# Run by the `indexer` compose service:  docker compose run --rm indexer
set -eu

# Generate the fictional company's PDFs and CSV only when there are none, so
# your own documents in data/docs are never overwritten.
if [ -z "$(ls -A data/docs 2>/dev/null)" ]; then
    echo "data/docs is empty: generating the Kestrel Ridge sample data"
    python scripts/make_sample_data.py
fi

python -m analyst.data.load   # CSVs in data/datasets -> data/analyst.db
python -m analyst.rag.index   # PDFs in data/docs -> data/index (embedded with the baked-in model)
echo "Done: data/analyst.db and data/index are ready."
