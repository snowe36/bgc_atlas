#!/usr/bin/env bash
# Reproduce the biosynthetic novelty atlas end-to-end (CPU-only).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -e ".[dev]"

np-download
np-featurize
np-sanity
np-atlas
np-novelty
np-validate
np-apply

echo "Done. See reports/novelty_ranking.csv and reports/figures/"
