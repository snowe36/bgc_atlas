#!/usr/bin/env bash
# Reproduce the biosynthetic architecture atlas end-to-end (CPU-only).
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
uv run bgc-validate
uv run bgc-apply
uv run bgc-temporal

echo "Done. See reports/novelty_ranking.csv and reports/figures/"
echo "(GPU embeddings are optional and not run here — see 'GPU / protein language model embeddings' in README.md;"
echo " once data/processed/esm_embeddings.npy exists, also run: uv run bgc-ablation && uv run bgc-novelty-compare)"
