#!/usr/bin/env bash
# Tiny AgentDojo × SED smoke (one user task × one injection).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export USER_TASKS="${USER_TASKS:-user_task_0}"
export INJECTION_TASKS="${INJECTION_TASKS:-injection_task_1}"
export SUITE_NAME="${SUITE_NAME:-slack}"

echo "==> AgentDojo SED smoke: suite=${SUITE_NAME} users=${USER_TASKS} injections=${INJECTION_TASKS}"
python experiments/agentdojo/run_sed.py
