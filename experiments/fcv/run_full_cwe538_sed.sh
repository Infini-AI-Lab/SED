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
PASS1_REPORT="${FCV_MS}/pass1_deepseek_100/swebench_report.json"
OUT_DIR="${FCV_MS}/fcv_cwe538_deepseek_68_v2_sed"
BASELINE_PREDS="${FCV_MS}/fcv_cwe538_deepseek_68_v2"
EVAL_ID="local-eval-fcv-cwe538-v2-sed"
CWE_ENV="cwe_538"
CWE_JUDGE="CWE-538"
# Default 4: matches ~48GB RAM + shared SED memory (see README). Override with FCV_WORKERS.
WORKERS="${FCV_WORKERS:-4}"

AJ_ROOT="${FCV_ROOT}/attack-lm-judge"
EVAL_REPORTS="${AJ_ROOT}/eval_reports/cwe538_v2_sed"

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

bash "${ROOT}/experiments/fcv/check_docker.sh"

SMOKE_PREDS="${FCV_MS}/fcv_cwe538_sed_smoke/preds.json"
if [[ "${FCV_SKIP_SMOKE_CHECK:-}" != "1" ]]; then
  "${PYTHON}" "${ROOT}/experiments/fcv/check_smoke_passed.py" "${SMOKE_PREDS}" 2
fi

if [[ ! -f "${PASS1_REPORT}" ]]; then
  echo "Missing ${PASS1_REPORT}"
  exit 1
fi

FILTER="$("${PYTHON}" "${ROOT}/experiments/fcv/resolved_ids_filter.py" "${PASS1_REPORT}")"

cd "${ROOT}"

echo "========== SED attack run (68 resolved) =========="
"${PYTHON}" evaluation/FCV/run_swebench_sed.py \
  --config "${CONFIG}" \
  -o "${OUT_DIR}" \
  --subset verified \
  --split test \
  --filter "${FILTER}" \
  --workers "${WORKERS}"

echo "========== Validate CWE injection =========="
python3 "${FCV_MS}/scripts/validate_cwe_injection.py" \
  --attack-dir "${OUT_DIR}" \
  --cwe-type "${CWE_ENV}" \
  --pass1-dir "${FCV_MS}/pass1_deepseek_100" \
  --expect-different-from-pass1

echo "========== SWE-bench eval =========="
"${PYTHON}" -m swebench.harness.run_evaluation \
  -d princeton-nlp/SWE-Bench_Verified \
  -s test \
  -p "${OUT_DIR}/preds.json" \
  --max_workers "${WORKERS}" \
  -id "${EVAL_ID}" \
  -t 900 \
  --cache_level env

bash "${ROOT}/experiments/fcv/run_finish_cwe538_sed.sh"
