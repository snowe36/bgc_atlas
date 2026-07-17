#!/usr/bin/env bash
# Reproduce the biosynthetic architecture atlas end-to-end (CPU-only).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -e ".[dev]"

bgc-download
bgc-featurize
bgc-sanity
bgc-atlas
bgc-novelty
bgc-validate
bgc-apply

echo "Done. See reports/novelty_ranking.csv and reports/figures/"
