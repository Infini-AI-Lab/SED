#!/usr/bin/env bash
set -euo pipefail

# Resume SED full eval from a given suite (skips cached JSON automatically).
# Example:
#   START_SUITE=banking bash experiments/agentdojo/run_resume_sed.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALL_SUITES=(slack banking travel workspace)
START_SUITE="${START_SUITE:-slack}"

SUITES=()
seen=0
for suite in "${ALL_SUITES[@]}"; do
  if [[ "${suite}" == "${START_SUITE}" ]]; then
    seen=1
  fi
  if [[ "${seen}" -eq 1 ]]; then
    SUITES+=("${suite}")
  fi
done

if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "Unknown START_SUITE=${START_SUITE}"
  exit 1
fi

export SED_SUITES="$(IFS=,; echo "${SUITES[*]}")"
LOG_DIR="${ROOT}/experiments/agentdojo/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
export LOG_SUFFIX="resume_${STAMP}"

exec bash "${ROOT}/experiments/agentdojo/run_full_sed.sh"
