#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT}/experiments/agentdojo/logs"
mkdir -p "${LOG_DIR}"

MODEL="${FIREWORKS_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"
PIPELINE_NAME="${SED_PIPELINE_NAME:-${MODEL##*/}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sed_full_agentdojo_${PIPELINE_NAME}_${STAMP}.log"

PYTHON="${PYTHON:-python}"
if [[ ! -x "${PYTHON}" ]]; then
  exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY in .env or environment}"

SUITES="${SED_SUITES:-slack,banking,travel,workspace}"

exec >> "${LOG_FILE}" 2>&1
echo "---- $(date) ----"

export FIREWORKS_MODEL="${MODEL}"
export SED_PIPELINE_NAME="${PIPELINE_NAME}"

echo "SED full AgentDojo run"
echo "  model: ${MODEL}"
echo "  pipeline: ${PIPELINE_NAME}"
echo "  python: ${PYTHON}"
echo "  suites: ${SUITES}"
echo "  runs:   experiments/agentdojo/runs/${PIPELINE_NAME}/"
echo "  log:    ${LOG_FILE}"
echo "  started: $(date)"

cd "${ROOT}"
"${PYTHON}" experiments/agentdojo/run_full_sed.py --suites "${SUITES}"

echo ""
echo "========== Per-suite SED metrics =========="
for SUITE in $(echo "${SUITES}" | tr ',' ' '); do
  echo "--- ${SUITE} ---"
  "${PYTHON}" experiments/agentdojo/compute_metrics.py \
    --pipeline "${PIPELINE_NAME}" \
    --suite "${SUITE}" \
    --attack important_instructions_sed || true
done

echo ""
echo "========== DRIFT vs SED (full attack matrix) =========="

echo ""
echo "Finished: $(date)"
echo "Log: ${LOG_FILE}"
