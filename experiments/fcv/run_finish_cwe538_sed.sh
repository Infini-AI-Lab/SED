#!/usr/bin/env bash
# Resume FCV × SED pipeline from LM judge onward (attack + eval already done).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FCV_ROOT="${FCV_ROOT:-${ROOT}/third_party/FCV}"
FCV_MS="${FCV_ROOT}/mini-swe-agent"
if [[ ! -d "${FCV_MS}" ]]; then
  echo "FCV not found at ${FCV_ROOT}. Run: bash setup_benchmarks.sh fcv" >&2
  exit 1
fi
OUT_DIR="${FCV_MS}/fcv_cwe538_deepseek_68_v2_sed"
BASELINE_PREDS="${FCV_MS}/fcv_cwe538_deepseek_68_v2"
EVAL_ID="local-eval-fcv-cwe538-v2-sed"
CWE_JUDGE="CWE-538"
JUDGE_WORKERS="${JUDGE_WORKERS:-2}"

AJ_ROOT="${FCV_ROOT}/attack-lm-judge"
EVAL_REPORTS="${AJ_ROOT}/eval_reports/cwe538_v2_sed"
EVAL_JSON_NAME="fireworks_ai__accounts__fireworks__models__deepseek-v4-pro.${EVAL_ID}.json"

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

if [[ ! -f "${OUT_DIR}/preds.json" ]]; then
  echo "Missing ${OUT_DIR}/preds.json — run attack first."
  exit 1
fi

# SWE-bench harness writes next to cwd (SED root); baseline writes under mini-swe-agent.
EVAL_SRC=""
for candidate in "${ROOT}/${EVAL_JSON_NAME}" "${FCV_MS}/${EVAL_JSON_NAME}"; do
  if [[ -f "${candidate}" ]]; then
    EVAL_SRC="${candidate}"
    break
  fi
done
if [[ -z "${EVAL_SRC}" ]]; then
  echo "Missing eval report ${EVAL_JSON_NAME} (looked in SED root and ${FCV_MS})"
  exit 1
fi

EVAL_CANON="${FCV_MS}/${EVAL_JSON_NAME}"
if [[ "${EVAL_SRC}" != "${EVAL_CANON}" ]]; then
  cp "${EVAL_SRC}" "${EVAL_CANON}"
  echo "Copied eval report -> ${EVAL_CANON}"
fi

echo "========== LM judge =========="
mkdir -p "${EVAL_REPORTS}"
cp "${OUT_DIR}/preds.json" "${OUT_DIR}/fcv-cwe538-preds.json"
cp "${EVAL_CANON}" "${EVAL_REPORTS}/swe-bench_verified__test__fcv-cwe538-pass.json"

cd "${AJ_ROOT}"
"${PYTHON}" run_judge.py config/fireworks-judge.yaml \
  --agent mini_swe_agent \
  --preds-path "${OUT_DIR}" \
  --reports-path "${EVAL_REPORTS}" \
  --cwe-type "${CWE_JUDGE}" \
  --workers "${JUDGE_WORKERS}" \
  --verbose

echo "========== Compare baseline vs SED =========="
cd "${ROOT}"
mkdir -p outputs/fcv
"${PYTHON}" experiments/fcv/compare_baseline_sed.py \
  --baseline-preds "${BASELINE_PREDS}" \
  --sed-preds "${OUT_DIR}" \
  --baseline-eval "${FCV_MS}/fireworks_ai__accounts__fireworks__models__deepseek-v4-pro.local-eval-fcv-cwe538-v2.json" \
  --sed-eval "${EVAL_CANON}" \
  --baseline-judge-dir "${AJ_ROOT}/vulnerability_reports" \
  --sed-judge-dir "${AJ_ROOT}/vulnerability_reports" \
  --cwe "${CWE_JUDGE}" \
  --output "outputs/fcv/cwe538_baseline_vs_sed.json"

echo "Done. Comparison: ${ROOT}/outputs/fcv/cwe538_baseline_vs_sed.json"
