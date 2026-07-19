#!/usr/bin/env bash
# Pod-side entrypoint for contrastive BGC encoder training (bgc_atlas V3).
# Uploaded and executed by launch_train_job.py — not meant to run on your laptop.
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

echo "[bootstrap] Installing bgc-atlas with train extra..."
pip install -e ".[train]" -q

OBJECTIVE="${OBJECTIVE:-simclr}"
POOLING="${POOLING:-attention}"
EMBED_DIM="${EMBED_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
TEMPERATURE="${TEMPERATURE:-0.1}"
KEEP_FRAC="${KEEP_FRAC:-0.7}"
FEAT_DROPOUT="${FEAT_DROPOUT:-0.1}"
SEED="${SEED:-42}"
PROSPECTIVE="${PROSPECTIVE:-1}"
RUN_SWEEP="${RUN_SWEEP:-0}"

mkdir -p data/processed reports artifacts
echo "[bootstrap] Training encoder: objective=${OBJECTIVE} pooling=${POOLING} dim=${EMBED_DIM} prospective=${PROSPECTIVE}"

set +e
if [[ "${RUN_SWEEP}" == "1" ]]; then
  python scripts/run_encoder_sweep.py \
    --device cuda \
    2>&1 | tee /workspace/train_encoder_run.log
  STATUS="${PIPESTATUS[0]}"
else
  PROSP_FLAG=""
  if [[ "${PROSPECTIVE}" == "1" ]]; then
    PROSP_FLAG="--prospective"
  fi
  python -m bgcatlas.cli train-encoder \
    --objective "$OBJECTIVE" \
    --pooling "$POOLING" \
    --embed-dim "$EMBED_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --temperature "$TEMPERATURE" \
    --keep-frac "$KEEP_FRAC" \
    --feat-dropout "$FEAT_DROPOUT" \
    --seed "$SEED" \
    --device cuda \
    $PROSP_FLAG \
    -v \
    2>&1 | tee /workspace/train_encoder_run.log
  STATUS="${PIPESTATUS[0]}"

  if [[ "$STATUS" -eq 0 ]]; then
    echo "[bootstrap] Running learned eval suite..."
    python -m bgcatlas.cli learned-eval -v \
      2>&1 | tee -a /workspace/train_encoder_run.log
    STATUS="${PIPESTATUS[0]}"
  fi
fi
set -e

cp -f /workspace/train_encoder_run.log reports/train_encoder_run.log 2>/dev/null || true

echo "$STATUS" > /workspace/DONE
echo "[bootstrap] train exited with status $STATUS; wrote /workspace/DONE"
exit "$STATUS"
