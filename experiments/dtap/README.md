# DTap (DecodingTrust-Agent) x SED

Run SED on [DecodingTrust-Agent](https://github.com/AI-secure/DecodingTrust-Agent) for CRM, workflow, and code domains.

Default model: `accounts/fireworks/models/deepseek-v4-flash-0731`

## Setup

```bash
bash setup_benchmarks.sh dtap
cp .env.example .env          # FIREWORKS_API_KEY required
```

This clones DTap into `third_party/DecodingTrust-Agent`, installs SED hooks, and patches the OpenAI SDK for Fireworks.

Optional: point `DTAP_ROOT` at an existing checkout. Use a Python env that can install the DTap package (`pip install -e ".[openai]"` inside the clone).

Docker is required for full DTap tasks (MCP servers and task environments).

## Smoke test (SED, one CRM task)

```bash
bash experiments/dtap/run_smoke_sed.sh
```

## Baseline smoke (no defense)

```bash
bash experiments/dtap/run_smoke.sh
```

## Full domain run

Build a domain task list, then run upstream `eval/evaluation.py` with `SED_ENABLE=1` and `DTAP_DEFENSE=sed`:

```bash
python experiments/dtap/build_domain_task_list.py \
  --domain crm \
  --dtap "${DTAP_ROOT:-third_party/DecodingTrust-Agent}" \
  --output experiments/dtap/task_lists/crm_full.jsonl
```

See the upstream DTap README for domain layout, benign / direct / indirect splits, and Docker requirements.

External baselines (Llama Guard 3, SafeHarbor, DRIFT, GuardAgent) are optional and use helpers under `experiments/defenses/`. SED alone does not need them.

## Files

| File | Role |
|------|------|
| `setup_dtap_sed.py` | Install hooks; patch task_runner + mcp_wrapper |
| `sed_guard.py` | SED MCP hook + end-of-task judge |
| `defense_turninject.py` | Turn-level policy injection |
| `patch_fireworks.py` | Route OpenAI SDK calls through Fireworks |
| `verify_dtap_defenses.py` | Hook checks without Docker |
| `run_smoke_sed.sh` | One-task SED smoke |
