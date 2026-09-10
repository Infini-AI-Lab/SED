#!/usr/bin/env bash
# Fetch the upstream benchmarks SED evaluates against.
#
# These are third-party projects and are NOT redistributed in this repository.
# They are cloned here at the exact commits used in the paper.
#
#   bash setup_benchmarks.sh            # everything
#   bash setup_benchmarks.sh redcode    # one target
#
# Override a location with REDCODE_ROOT / OPENRT_ROOT / FCV_ROOT / DTAP_ROOT if you already have a
# checkout elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${ROOT}/third_party"

# ── pinned upstream commits ───────────────────────────────────────────────────
REDCODE_URL="https://github.com/AI-secure/RedCode.git"
REDCODE_SHA="dbbf08281c56669d88502feb38b2dd901a69333c"

OPENRT_URL="https://github.com/AI45Lab/OpenRT.git"
OPENRT_SHA="81efa876f15c22dcff624894eb5d649e135ac0ec"

# FCV ships both mini-swe-agent and attack-lm-judge, which the CWE-538 runs need.
FCV_URL="https://github.com/Infini-AI-Lab/FCV.git"
FCV_SHA="ea132bd0067ed57e6829b39f4ee16c17a7eb68f7"

# DecodingTrust-Agent (DTap) — CRM / workflow / code tool-agent benchmark.
DTAP_URL="https://github.com/AI-secure/DecodingTrust-Agent.git"
DTAP_SHA="e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
# ──────────────────────────────────────────────────────────────────────────────

clone_at() {
  local name="$1" url="$2" sha="$3" dest="${VENDOR}/$1"
  # OpenRT's importable package lives at <dest>/OpenRT; other repos vary.
  if [[ -d "${dest}/.git" ]]; then
    if [[ "${name}" == "OpenRT" && ! -d "${dest}/OpenRT" ]]; then
      echo "==> ${name}: present but missing package tree — re-cloning"
      rm -rf "${dest}"
    elif ! git -C "${dest}" rev-parse --verify --quiet "${sha}^{commit}" >/dev/null 2>&1; then
      echo "==> ${name}: present but pinned commit missing — fetching / recovering"
      if ! git -C "${dest}" fetch --quiet origin "${sha}" 2>/dev/null; then
        echo "    fetch failed — re-cloning from ${url}"
        rm -rf "${dest}"
      fi
    fi
  fi
  if [[ -d "${dest}/.git" ]]; then
    echo "==> ${name}: already present at ${dest}"
    if [[ "$(git -C "${dest}" rev-parse HEAD)" != "${sha}" ]]; then
      echo "    checking out pinned ${sha:0:12}"
      git -C "${dest}" fetch --quiet origin "${sha}"
      git -C "${dest}" checkout --quiet "${sha}"
    fi
    return
  fi
  echo "==> ${name}: cloning ${url}"
  mkdir -p "${VENDOR}"
  # If a broken non-git directory is in the way, remove it.
  if [[ -e "${dest}" && ! -d "${dest}/.git" ]]; then
    echo "    removing broken checkout at ${dest}"
    rm -rf "${dest}"
  fi
  git clone --quiet "${url}" "${dest}"
  git -C "${dest}" checkout --quiet "${sha}"
  echo "    pinned at ${sha:0:12}"
}

setup_redcode() {
  clone_at RedCode "${REDCODE_URL}" "${REDCODE_SHA}"
  cat <<'EOF'
    RedCode also needs its Docker image before evaluation can run:
      cd third_party/RedCode/environment && docker build -t redcode .
    See third_party/RedCode/dataset/README.md for the RedCode-Exec dataset.
EOF
}

setup_openrt() {
  clone_at OpenRT "${OPENRT_URL}" "${OPENRT_SHA}"
  echo "    OpenRT supplies the AutoDAN-Turbo / PAIR / TAP / X-Teaming attackers."
  echo "    Note: OpenRT is licensed AGPL-3.0."
  # SED imports OpenRT from third_party/ via sys.path (no editable install).
  # OpenRT.models eagerly imports PIL; install the light deps the text path needs.
  if command -v python >/dev/null 2>&1; then
    echo "    installing OpenRT text-path deps (Pillow, nest-asyncio, aiofiles)…"
    python -m pip install --quiet "Pillow>=10" "nest-asyncio>=1.6" "aiofiles>=23" || {
      echo "    WARNING: pip install of OpenRT deps failed — run:" >&2
      echo "      pip install 'Pillow>=10' nest-asyncio aiofiles" >&2
    }
  else
    echo "    WARNING: python not on PATH; later run: pip install 'Pillow>=10'" >&2
  fi
}

# Pin matches the version exercised in the release Mac smokes.
AGENTDOJO_VERSION="${AGENTDOJO_VERSION:-0.1.35}"

setup_agentdojo() {
  echo "==> AgentDojo: pip install agentdojo==${AGENTDOJO_VERSION}"
  if command -v python >/dev/null 2>&1; then
    python -m pip install --quiet "agentdojo==${AGENTDOJO_VERSION}" || {
      echo "    WARNING: pip install agentdojo==${AGENTDOJO_VERSION} failed" >&2
      echo "      run: pip install 'agentdojo==${AGENTDOJO_VERSION}'" >&2
      return 0
    }
    echo "    agentdojo==${AGENTDOJO_VERSION} installed"
  else
    echo "    WARNING: python not on PATH; run: pip install 'agentdojo==${AGENTDOJO_VERSION}'" >&2
  fi
}
setup_agentharm() {
  echo "==> AgentHarm: uses inspect_ai (already in requirements.txt)."
  echo "    Dataset downloads from Hugging Face on first run."
  echo "    See experiments/agentharm/README.md for smoke commands."
}
setup_fcv() {
  clone_at FCV "${FCV_URL}" "${FCV_SHA}"
  echo "    supplies mini-swe-agent/ and attack-lm-judge/ for the CWE-538 runs."
  local cfg="${ROOT}/evaluation/FCV/model_configs/fireworks_cwe538.yaml"
  local dest="${VENDOR}/FCV/mini-swe-agent/model_configs/fireworks_cwe538.yaml"
  if [[ -f "${cfg}" ]]; then
    cp "${cfg}" "${dest}"
    echo "    installed model_configs/fireworks_cwe538.yaml"
  else
    echo "    WARNING: ${cfg} is missing — the CWE-538 runs need it." >&2
  fi
}

setup_dtap() {
  clone_at DecodingTrust-Agent "${DTAP_URL}" "${DTAP_SHA}"
  echo "    DTap adapter: python experiments/dtap/setup_dtap_sed.py ${VENDOR}/DecodingTrust-Agent"
  echo "    Smoke: bash experiments/dtap/run_smoke_sed.sh"
  python "${ROOT}/experiments/dtap/setup_dtap_sed.py" "${VENDOR}/DecodingTrust-Agent" || {
    echo "    WARNING: setup_dtap_sed.py failed — run manually after fixing errors." >&2
  }
  python "${ROOT}/experiments/dtap/patch_fireworks.py" "${VENDOR}/DecodingTrust-Agent" || true
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(redcode openrt agentdojo agentharm fcv dtap)

for t in "${targets[@]}"; do
  case "$(printf '%s' "${t}" | tr '[:upper:]' '[:lower:]')" in
    redcode)   setup_redcode   ;;
    openrt)    setup_openrt    ;;
    agentdojo) setup_agentdojo ;;
    agentharm) setup_agentharm ;;
    fcv)       setup_fcv       ;;
    dtap)      setup_dtap      ;;
    *) echo "unknown target: ${t}" >&2; exit 1 ;;
  esac
done

echo
echo "Done. Clones live in ${VENDOR} (git-ignored)."
