#!/bin/sh
# First start: build the database, train/register models and index the knowledge base.
set -e

if [ ! -f "${ATLAS_ARTIFACTS_DIR:-artifacts}/models/category/production" ]; then
  echo "[atlas] No production models found — running bootstrap (data → train → index → evaluate)…"
  atlas bootstrap --no-gates
fi

exec "$@"
