# AgentDojo → SED

Runs SED on [AgentDojo](https://github.com/ethz-spylab/agentdojo) suites with the
`important_instructions` prompt-injection attack.

## Setup

```bash
# from repo root, with venv active
bash setup_benchmarks.sh agentdojo   # pip install agentdojo==0.1.35
cp .env.example .env                 # then set FIREWORKS_API_KEY
```

`run_sed.py` exits with a clear message if `FIREWORKS_API_KEY` is missing/empty
(instead of a generic OpenAI credentials error).

## Smoke (Mac-friendly)

```bash
USER_TASKS=user_task_0 INJECTION_TASKS=injection_task_1 \
  bash experiments/agentdojo/run_smoke_sed.sh
```

Or directly:

```bash
USER_TASKS=user_task_0 INJECTION_TASKS=injection_task_1 \
  python experiments/agentdojo/run_sed.py
```

### Reading smoke output

Cold memory (first episode) typically shows `policies=0`. On each
`user_task × injection_task` pair:

| Field | Meaning |
|---|---|
| `utility` | user task still succeeds under injection |
| `security` | `True` = injection blocked / did not achieve attacker goal |
| `policies` | how many L2 policies were injected into the system prompt |

Artifacts:

- runs: `experiments/agentdojo/runs/` (JSON per task pair; **cached by default**)
- SED memory: `experiments/agentdojo/sed_memory/<pipeline>/<suite>/<attack>/`

### Cache / force rerun

Failed or partial runs still write JSON under `runs/`. The next invocation may
print `[SED] cached …` and skip the API. To re-run for real:

```bash
# option A
SED_FORCE_RERUN=1 USER_TASKS=user_task_0 INJECTION_TASKS=injection_task_1 \
  python experiments/agentdojo/run_sed.py --force-rerun

# option B
rm -rf experiments/agentdojo/runs
```

## Env knobs

| Env | Default | Role |
|---|---|---|
| `SUITE_NAME` | `slack` | AgentDojo suite |
| `USER_TASKS` | all (smoke: set `user_task_0`) | comma-separated user tasks |
| `INJECTION_TASKS` | `injection_task_1,2,3` | comma-separated injections |
| `FIREWORKS_MODEL` | from `.env` | chat model |
| `SED_PIPELINE_NAME` | `local` | AgentDojo attack registry name (keep `local`) |
| `SED_FORCE_RERUN` | unset/false | ignore cached runs |
| `SED_RUNS_DIR` | `experiments/agentdojo/runs` | cache / log directory |

## Full suite

```bash
bash experiments/agentdojo/run_full_sed.sh
# or
python experiments/agentdojo/run_full_sed.py
```

## Citation

```bibtex
@inproceedings{debenedetti2024agentdojo,
  title={AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents},
  author={Debenedetti, Edoardo and others},
  booktitle={NeurIPS},
  year={2024}
}
```
