# FCV x SED

Evaluate SED on [FCV](https://github.com/Infini-AI-Lab/FCV) CWE-injected SWE-bench instances (CWE-538 by default). No attack generation step: instances are pre-built upstream.

## Prerequisites

- Docker running
- `FIREWORKS_API_KEY` in the SED `.env`
- FCV clone via `bash setup_benchmarks.sh fcv` (or set `FCV_ROOT`)
- A Python env with mini-swe-agent deps (`conda activate minisweagent` if you use that env)

`setup_benchmarks.sh fcv` also installs `evaluation/FCV/model_configs/fireworks_cwe538.yaml` into the FCV clone.

## Setup

```bash
bash setup_benchmarks.sh fcv
bash experiments/fcv/setup_env.sh
bash experiments/fcv/check_docker.sh
```

## Smoke (2 instances, CWE-538)

```bash
bash experiments/fcv/run_smoke_cwe538_sed.sh
```

Do not start the full run until smoke produces real patches (not Docker/`CalledProcessError` failures). `run_full_cwe538_sed.sh` checks this via `check_smoke_passed.py` (override with `FCV_SKIP_SMOKE_CHECK=1`).

Failed Docker instances are retried on the next launch. Use `SMOKE_REDO=1` to wipe and rerun smoke from scratch.

## Full run (shared SED memory)

```bash
FCV_WORKERS=4 bash experiments/fcv/run_full_cwe538_sed.sh
```

Pipeline: `run` -> `validate` -> `eval` -> `judge` -> `compare`

| `FCV_WORKERS` | Notes |
|---|---|
| `1` | Slowest, simplest |
| `4` (default) | Shared SED memory across instances |
| `6-8` | Faster if Docker and API keep up; watch RAM |

## Judge mode

By default SED judges once per instance (end of task), matching AgentDojo:

| Env var | Default | Meaning |
|---|---|---|
| `SED_FCV_JUDGE_MODE` | `end` | Judge + memory update once when the instance finishes |
| `SED_FCV_JUDGE_MODE` | `step` | Judge after every bash step (much slower) |
| `SED_FCV_DEFER_L2_PERSIST` | `1` | Persist L2 embeds on flush, not every step |
| `SED_FCV_JUDGE_PARSE_FAIL` | `benign` | Treat unparseable judge JSON as benign |

Policies are still injected before every step.

## Outputs

| Path | Content |
|------|---------|
| `third_party/FCV/mini-swe-agent/fcv_cwe538_*_sed/` | preds + trajectories |
| `experiments/fcv/sed_memory/` | L1/L2 memory |
| `outputs/fcv/` | comparison JSON |

## Metrics

- **FCV ASR**: LM-judge vulnerable rate on functionally resolved patches
- **Resolve rate**: SWE-bench functional pass rate
- **Delta ASR**: baseline minus SED
