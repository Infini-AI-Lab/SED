# HarmBench → SED

Runs [HarmBench](https://github.com/centerforaisafety/HarmBench) adversarial
prompts through SED and scores them with the HarmBench classifier prompt.

| File | Role |
|------|------|
| `run_sed_hb.py` | Per example: retrieve → run target model → judge → memorize |
| `eval_hb_results.py` | Score generations with the HarmBench classifier prompt; ASR is the fraction of "yes" labels |

## Data

Shipped under `evaluation/Harmbench/`:

| File | Contents |
|------|----------|
| `adversarial_1400.jsonl` | 1,400 precomputed adversarial prompts — 14 attack methods × 100 behaviors (GCG, EnsembleGCG, PAIR, TAP, AutoDAN, AutoPrompt, PAP, PEZ, GBDA, UAT, FewShot, HumanJailbreaks, IntentMasking, ZeroShot) |
| `harmbench_behaviors_text_all.csv` | 400 behaviors — the full text set, used to resolve behavior metadata |
| `harmbench_behaviors_text_test.csv` | 320 behaviors — the official test split, used by the adaptive-attack experiment |

## Smoke

```bash
# from repo root; needs FIREWORKS_API_KEY + FIREWORKS_MODEL in .env
python experiments/harmbench/run_sed_hb.py --limit 2 -v
python experiments/harmbench/eval_hb_results.py --limit 2 -v
```

Without `--limit`, `run_sed_hb.py` walks the full 1,400-prompt file.

Both use the Fireworks models and judge configured in `.env`
(`FIREWORKS_MODEL` / `JUDGE_MODEL`).

## Citation

```bibtex
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and others},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
