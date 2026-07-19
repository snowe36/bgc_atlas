#!/usr/bin/env bash
# CPU pipeline: download → temporal.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv sync --extra dev

uv run bgc-download
uv run bgc-featurize
uv run bgc-sanity
uv run bgc-atlas
uv run bgc-novelty
uv run python scripts/run_case_studies.py
uv run bgc-validate
uv run bgc-apply
uv run bgc-temporal

echo "Done. See reports/ and reports/figures/"
echo "Optional GPU: docs/esm.md (then bgc-ablation && bgc-novelty-compare)"
