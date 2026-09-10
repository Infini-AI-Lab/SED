#!/usr/bin/env bash
# DTap smoke with SED enabled: one benign CRM task end-to-end.
#
# Prerequisites:
#   bash setup_benchmarks.sh dtap
#   conda activate minisweagent
#   FIREWORKS_API_KEY in .env
#
# Usage:
#   bash experiments/dtap/run_smoke_sed.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DTAP_DIR="${DTAP_ROOT:-${ROOT}/third_party/DecodingTrust-Agent}"
LOG="${ROOT}/experiments/dtap/smoke_sed.log"
mkdir -p "$(dirname "${LOG}")"

if [[ ! -d "${DTAP_DIR}" ]]; then
  echo "DTap not found at ${DTAP_DIR}. Run: bash setup_benchmarks.sh dtap" >&2
  exit 1
fi

if [[ -n "${MINISWE_PYTHON:-}" ]]; then
  PYTHON="${MINISWE_PYTHON}"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate minisweagent
  PYTHON="$(which python)"
else
  PYTHON="$(which python3)"
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY in .env}"

export OPENAI_API_KEY="${FIREWORKS_API_KEY}"
export OPENAI_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}"
export FIREWORKS_BASE_URL="${OPENAI_BASE_URL}"
export SED_ROOT="${ROOT}"
export SED_ENABLE=1
export DTAP_DEFENSE=sed
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODEL="${FIREWORKS_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"

"${PYTHON}" "${ROOT}/experiments/dtap/setup_dtap_sed.py" "${DTAP_DIR}"
"${PYTHON}" "${ROOT}/experiments/dtap/patch_fireworks.py" "${DTAP_DIR}"

cd "${DTAP_DIR}"
pip install -q -e ".[openai]" 2>&1 | tail -3 || true

SMOKE="${DTAP_DIR}/smoke_sed_one_task.jsonl"
echo '{"domain": "crm", "type": "benign", "task_id": "1"}' > "${SMOKE}"

rm -rf "${DTAP_DIR}/results/benchmark/openaisdk"/*/crm/benign/1 2>/dev/null || true

echo "=== $(date) DTap SED smoke model=${MODEL} ===" | tee "${LOG}"
"${PYTHON}" eval/evaluation.py \
  --task-list "${SMOKE}" \
  --agent-type openaisdk \
  --model "${MODEL}" \
  --max-parallel 1 \
  2>&1 | tee -a "${LOG}"
echo "=== $(date) DTap SED smoke done ===" | tee -a "${LOG}"
