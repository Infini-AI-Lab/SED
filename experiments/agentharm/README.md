# AgentHarm

Run SED on AgentHarm harmful and benign splits with Inspect-AI:

```bash
python experiments/agentharm/run_sed.py \
  --task both \
  --split test_public \
  --target-model accounts/fireworks/models/deepseek-v4-flash-0731 \
  --judge-model accounts/fireworks/models/deepseek-v4-flash-0731
```

Smoke test a single validation sample without writing memory:

```bash
python experiments/agentharm/run_sed.py \
  --task harmful \
  --split val \
  --limit 1 \
  --no-update-memory
```

Memory-updating runs default to `--max-connections 1` so SED writes to its
JSONL memory sequentially. For frozen-memory evals with `--no-update-memory`,
you can raise `--max-connections` for speed.

The reusable benchmark adapter lives in `evaluation/AgentHarm/`, including the
AgentHarm benchmark tools and grading functions. Runtime logs and SED memory are
written under `experiments/agentharm/runs/` and
`experiments/agentharm/sed_memory/`, both ignored by Git.


```bash
inspect view start --log-dir experiments/agentharm/runs
```
