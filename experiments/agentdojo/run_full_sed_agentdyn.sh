#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT}/experiments/agentdojo/logs"
mkdir -p "${LOG_DIR}"

MODEL="${FIREWORKS_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"
PIPELINE_NAME="${SED_PIPELINE_NAME:-${MODEL##*/}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sed_full_agentdyn_${PIPELINE_NAME}_${STAMP}.log"

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
: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY}"

export FIREWORKS_MODEL="${MODEL}"
export SED_PIPELINE_NAME="${PIPELINE_NAME}"
export SED_RUNS_DIR="experiments/agentdyn/runs"
export SED_MEMORY_ROOT="experiments/agentdyn/sed_memory"
export SED_BENCHMARK_VERSION="${SED_BENCHMARK_VERSION:-v1.2.2}"

exec >> "${LOG_FILE}" 2>&1
echo "---- $(date) ----"
echo "SED full AgentDyn run"
echo "  model: ${MODEL}"
echo "  pipeline: ${PIPELINE_NAME}"
echo "  log: ${LOG_FILE}"

cd "${ROOT}"
"${PYTHON}" experiments/agentdojo/run_full_sed.py --suites shopping,github,dailylife

echo ""
echo "========== Per-suite SED metrics =========="
for SUITE in shopping github dailylife; do
  echo "--- ${SUITE} ---"
  "${PYTHON}" experiments/agentdojo/compute_metrics.py \
    --runs-dir experiments/agentdyn/runs \
    --pipeline "${PIPELINE_NAME}" \
    --suite "${SUITE}" \
    --attack important_instructions_sed || true
done

echo "Finished: $(date)"
