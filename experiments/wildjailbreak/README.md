# WildJailbreak → SED

Runs SED on the WildJailbreak evaluation split shipped under
`evaluation/wildjailbreak/`.

## Smoke

```bash
# from repo root, with .env configured
python experiments/wildjailbreak/run_sed_wjb.py --limit 1 -v
```

## Eval

```bash
python experiments/wildjailbreak/eval_wjb_results.py --help
```

Both use `FIREWORKS_API_KEY` / `FIREWORKS_MODEL` / `JUDGE_MODEL` from `.env`
(repo `.env` overrides a stale shell `FIREWORKS_MODEL` if one is exported).

Results append to `experiments/wildjailbreak/sed_wjb_results.jsonl`. A failed
attempt leaves an `error` row; filter those out (or delete the file) before
reading ASR. If every example fails, the runner exits non-zero.
