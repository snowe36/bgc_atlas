#!/usr/bin/env bash
# 5-minute showcase — curated predicted BGCs; no fresh MIBiG download required
# Does not run `uv sync` (avoids dropping optional extras like torch/embed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg

run_py() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run --no-sync python "$@"
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 "$@"
  else
    python3 "$@"
  fi
}

# Lightweight editable install if import fails
if ! run_py -c "import bgcatlas" >/dev/null 2>&1; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" -m pip install -q -e ".[dev]"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install -q -e ".[dev]"
  else
    run_py -m pip install -q -e ".[dev]"
  fi
fi

run_py -c "from bgcatlas.demo import run_demo; run_demo()"

echo ""
echo "Open:"
echo "  demo/outputs/biological_case_studies.png"
echo "  demo/outputs/predicted_novelty_ranking.csv"
