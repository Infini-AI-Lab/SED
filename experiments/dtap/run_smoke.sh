#!/usr/bin/env bash
# DTap smoke: one benign CRM task end-to-end (Docker + MCP + judge).
#
# Usage (on Lovelace):
#   bash experiments/dtap/run_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DTAP_DIR="${HOME}/nisargagondi/DecodingTrust-Agent"
LOG="${HOME}/nisargagondi/logs/dtap_smoke.log"
PATCH="${ROOT}/experiments/dtap/patch_fireworks.py"
mkdir -p "$(dirname "${LOG}")"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate minisweagent
set -a; source "${ROOT}/.env"; set +a

# Route OpenAI Agents SDK through Fireworks OpenAI-compatible API.
export OPENAI_API_KEY="${FIREWORKS_API_KEY}"
export OPENAI_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}"
export FIREWORKS_BASE_URL="${OPENAI_BASE_URL}"
MODEL="${FIREWORKS_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"

if [[ ! -d "${DTAP_DIR}" ]]; then
  echo "Cloning DecodingTrust-Agent..."
  git clone --depth 1 https://github.com/AI-secure/DecodingTrust-Agent.git "${DTAP_DIR}"
fi

python "${PATCH}" "${DTAP_DIR}"

cd "${DTAP_DIR}"
pip install -q -e ".[openai]" 2>&1 | tail -3

SMOKE="${DTAP_DIR}/smoke_one_task.jsonl"
echo '{"domain": "crm", "type": "benign", "task_id": "1"}' > "${SMOKE}"

# Clean prior failed smoke artifacts so --skip-existing does not hide a re-run.
rm -rf "${DTAP_DIR}/results/benchmark/openaisdk"/*/crm/benign/1 2>/dev/null || true

echo "=== $(date) DTap smoke start model=${MODEL} base=${OPENAI_BASE_URL} ===" | tee "${LOG}"
python eval/evaluation.py \
  --task-list "${SMOKE}" \
  --agent-type openaisdk \
  --model "${MODEL}" \
  --max-parallel 1 \
  2>&1 | tee -a "${LOG}"
echo "=== $(date) DTap smoke done ===" | tee -a "${LOG}"
