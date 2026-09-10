#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FCV_ROOT="${FCV_ROOT:-${ROOT}/third_party/FCV}"
FCV_MS="${FCV_ROOT}/mini-swe-agent"
if [[ ! -d "${FCV_MS}" ]]; then
  echo "FCV not found at ${FCV_ROOT}. Run: bash setup_benchmarks.sh fcv" >&2
  exit 1
fi
CONFIG="${FCV_MS}/model_configs/fireworks_cwe538.yaml"
OUT_DIR="${FCV_MS}/fcv_cwe538_sed_smoke"
FILTER='(django__django-10914|astropy__astropy-13236)'

if [[ -n "${MINISWE_PYTHON:-}" ]]; then
  PYTHON="${MINISWE_PYTHON}"
elif [[ -x "${HOME}/Applications/anaconda3/envs/minisweagent/bin/python" ]]; then
  PYTHON="${HOME}/Applications/anaconda3/envs/minisweagent/bin/python"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate minisweagent
  PYTHON="$(which python)"
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
: "${FIREWORKS_API_KEY:?Set FIREWORKS_API_KEY}"

export OPENAI_API_KEY="${FIREWORKS_API_KEY}"

bash "${ROOT}/experiments/fcv/check_docker.sh"

cd "${ROOT}"

SMOKE_ARGS=(
  --config "${CONFIG}"
  -o "${OUT_DIR}"
  --subset verified
  --split test
  --filter "${FILTER}"
  --workers 1
)
if [[ "${SMOKE_REDO:-}" == "1" ]]; then
  SMOKE_ARGS+=(--redo-existing)
fi

"${PYTHON}" evaluation/FCV/run_swebench_sed.py "${SMOKE_ARGS[@]}"

"${PYTHON}" "${ROOT}/experiments/fcv/check_smoke_passed.py" "${OUT_DIR}/preds.json" 2

echo "Smoke preds: ${OUT_DIR}/preds.json"
echo "Memory: ${ROOT}/experiments/fcv/sed_memory/deepseek-v4-pro/cwe_538/"
