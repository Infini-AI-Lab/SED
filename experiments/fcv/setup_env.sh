#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FCV_ROOT="${FCV_ROOT:-${ROOT}/third_party/FCV}"
FCV_MS="${FCV_ROOT}/mini-swe-agent"
if [[ ! -d "${FCV_MS}" ]]; then
  echo "FCV not found at ${FCV_ROOT}. Run: bash setup_benchmarks.sh fcv" >&2
  exit 1
fi

if [[ ! -d "${FCV_MS}/src/minisweagent" ]]; then
  echo "Missing FCV mini-swe-agent at ${FCV_MS}"
  exit 1
fi

PYTHON="${MINISWE_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate minisweagent
    PYTHON="$(which python)"
  else
    echo "Set MINISWE_PYTHON to minisweagent env python, or install conda env minisweagent"
    exit 1
  fi
fi

"${PYTHON}" "${ROOT}/experiments/fcv/verify_imports.py"

echo "Use: conda activate minisweagent"
echo "SED runner: python evaluation/FCV/run_swebench_sed.py"
