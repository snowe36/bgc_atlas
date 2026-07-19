#!/usr/bin/env bash
# Pod-side ESM embed entrypoint (uploaded by launch_gpu_job.py).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/bgc_atlas}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
cd "$REPO_DIR"

MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}" # 2h default
if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
  (
    sleep "$MAX_RUNTIME_SECONDS"
    echo "[watchdog] Max runtime (${MAX_RUNTIME_SECONDS}s) exceeded — self-terminating pod ${RUNPOD_POD_ID}"
    curl -s --request DELETE \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      >> /workspace/watchdog.log 2>&1
  ) & disown
  echo "[bootstrap] Watchdog armed: self-terminate after ${MAX_RUNTIME_SECONDS}s (pod ${RUNPOD_POD_ID})"
else
  echo "[bootstrap] WARNING: RUNPOD_API_KEY/RUNPOD_POD_ID not set — no self-terminate watchdog!" >&2
fi

echo "[bootstrap] GPU info:"
nvidia-smi || true

echo "[bootstrap] Creating / activating venv at ${VENV_DIR}..."
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip -q

echo "[bootstrap] Installing bgc-atlas with embed extra (transformers)..."
# Image already has torch; pull project + transformers/accelerate stack
pip install -e ".[embed]" -q

ESM_MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
ESM_POOLING="${ESM_POOLING:-length_weighted}"
ESM_MAX_AA="${ESM_MAX_AA:-1024}"
ESM_MAX_PROTEINS="${ESM_MAX_PROTEINS:-80}"
ESM_BATCH_TOKENS="${ESM_BATCH_TOKENS:-6000}"

mkdir -p data/processed reports
echo "[bootstrap] Running ESM embed: model=${ESM_MODEL} pooling=${ESM_POOLING}"
set +e
python scripts/run_esm_embed.py \
  --input data/processed/mibig_proteins.parquet \
  --outdir data/processed \
  --model "$ESM_MODEL" \
  --pooling "$ESM_POOLING" \
  --max-aa "$ESM_MAX_AA" \
  --max-proteins-per-bgc "$ESM_MAX_PROTEINS" \
  --batch-tokens "$ESM_BATCH_TOKENS" \
  2>&1 | tee /workspace/esm_embed_run.log
STATUS="${PIPESTATUS[0]}"
set -e

# Copy log into reports for rsync-down
cp -f /workspace/esm_embed_run.log reports/esm_embed_run.log 2>/dev/null || true

echo "$STATUS" > /workspace/DONE
echo "[bootstrap] esm embed exited with status $STATUS; wrote /workspace/DONE"
exit "$STATUS"
