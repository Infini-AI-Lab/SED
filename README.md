<div align="center">
<h1> Self-Evolving Defense: Continual Security Policy
Learning for LLM Agents </h1>

Minh Nhat Le<sup>2,†</sup>, Nisarga Gondi<sup>1,†</sup>, Yibo Peng<sup>1</sup>, Ronghao Ni<sup>1</sup>, Limin Jia<sup>1</sup>, Beidi Chen<sup>1</sup>, Haizhong Zheng<sup>1</sup>
<br>
<sup>1</sup>Carnegie Mellon University,
<sup>2</sup>University of Massachusetts Amherst
<br>
†Equal contribution

<div align="center">
[<a href="https://infini-ai-lab.github.io/SED/">Website</a>]
[<a href="TODO-PAPER-URL">Paper</a>]
</div>
<br>

<!-- ---------- -->
**TL;DR**
LLM agents increasingly read untrusted content, call external tools, and modify software repositories, which exposes them to jailbreaks, prompt injection, and insecure code generation. Existing defenses require retraining, fail to adapt as attacks evolve, or guard only a single interface. We present Self-Evolving Defense (SED), a training-free framework that distills harmful agent trajectories into reusable security policies and retrieves the relevant ones for later tasks, so a frozen agent keeps adapting without any weight update. Memory is read during an episode and written only after an external judge scores the completed trajectory, so each attack success becomes a defense against the next attempt. Across three open-source models (DeepSeek-V4-Flash, GLM-5.2, Kimi K3) and eight benchmarks spanning jailbreaks, prompt injection, and insecure code, SED holds adaptive X-Teaming attack success on HarmBench to 7.8% against 35.2% for the best baseline defense, lowers targeted prompt injection on AgentDojo to 0.42% against 3.7%, and cuts RedCode risky execution from 95.6% to 11.3%, all while preserving benign task utility.

</div>

> **Content warning:** this repository contains prompts and model outputs that are harmful or offensive by construction.

## Overview

<!-- ---------- -->
<p align="center">
  <img src="docs/static/images/main_diagram.png" alt="One SED episode: read-only retrieval during the episode, judged memory writes after it" style="width:100%;"/>
</p>

<p align="center"><i>
<strong>Figure 1</strong> One SED episode. During the episode (left, read-only) the retriever scores each policy body against the query by cosine similarity, selects the top-N, and injects them into the frozen agent's system prompt (steps 1–3). After the episode (right) an external judge scores the full trajectory once: benign or refused trajectories are discarded, while a judged attack success is written to episodic memory as evidence, synthesized into new policies, and placed into the policy tree (steps 4–6). The updated policy memory serves the next episode (step 7). Memory is never written mid-task, so an attack success can only influence later episodes.
</i></p>

<!-- ------- -->

## Getting Started

SED wraps existing agent benchmarks rather than replacing them, so most paths reuse upstream code (RedCode, OpenRT, AgentDojo, AgentHarm, FCV, DTap) with a SED adapter layered on top.

### 1. Environment Setup

Requires **Python ≥ 3.10**. On shared servers the default `python3` may still be 3.9 — use `python3.11` / `python3.12` explicitly:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

`requirements.txt` pulls a full ML stack (`torch`, `transformers`, `sentence-transformers`, …) even for API-only Fireworks runs, so expect a large first install on a laptop. Embeddings still go through the Fireworks API (`qwen3-embedding-8b`), not local GPU inference.

The repo `.env` **overrides** any already-exported `FIREWORKS_*` shell variables, so a stale `FIREWORKS_MODEL` in your environment cannot silently win.

### 2. Fireworks API Key and Models

SED calls Fireworks for chat, memory, judge, and embeddings.

1. Create an API key at [fireworks.ai/api-keys](https://fireworks.ai/api-keys).
2. Put it in `.env` as `FIREWORKS_API_KEY=...`.
3. Confirm the model IDs in `.env` are deployed on your account:

```bash
curl -s https://api.fireworks.ai/inference/v1/models \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" | head
```

Defaults in `.env.example`:

| Variable | Default |
|---|---|
| `FIREWORKS_MODEL` | `accounts/fireworks/models/deepseek-v4-flash-0731` |
| `FIREWORKS_MEMORY_MODEL` | same |
| `JUDGE_MODEL` | same |

The paper also reports `glm-5p2` and `kimi-k3` on Fireworks. Swap those IDs into the three variables above to reproduce those rows.

### 3. Fetch Benchmarks

Upstream benchmarks are not redistributed. Clone them at the paper commits when a path needs them:

```bash
bash setup_benchmarks.sh            # all targets
bash setup_benchmarks.sh openrt     # adaptive jailbreaks
bash setup_benchmarks.sh agentdojo  # installs the agentdojo package
bash setup_benchmarks.sh redcode   # needs Docker
bash setup_benchmarks.sh fcv
bash setup_benchmarks.sh dtap
```

| Target | Source | Notes |
|---|---|---|
| `redcode` | AI-secure/RedCode | needs Docker |
| `openrt` | AI45Lab/OpenRT | AutoDAN-Turbo / PAIR / TAP / X-Teaming (AGPL-3.0); also installs OpenRT deps |
| `fcv` | Infini-AI-Lab/FCV | mini-swe-agent + attack-lm-judge |
| `dtap` | AI-secure/DecodingTrust-Agent | CRM / workflow / code; needs Docker |
| `agentdojo` | `pip install agentdojo==0.1.35` | no git clone; version pinned by the script |
| `agentharm` | via `inspect_ai` | downloads the HF dataset on first run |

HarmBench and WildJailbreak evaluation files ship in this repo under `evaluation/`, with no `setup_benchmarks.sh` step.

Two targets need a step after cloning:

- **RedCode** also needs its Docker image built by hand — `setup_benchmarks.sh` only prints the reminder:

  ```bash
  cd third_party/RedCode/environment && docker build -t redcode .
  ```

  The RedCode-Exec dataset ships inside the clone at `third_party/RedCode/dataset/RedCode-Exec`; see that repo's `dataset/README.md` if a task index is missing.

- **DTap** is patched automatically: the script runs `experiments/dtap/setup_dtap_sed.py` and `experiments/dtap/patch_fireworks.py` against the clone. If either fails it warns and continues, so re-run them by hand before the smoke.

**Output**: clones go under `third_party/` (gitignored). To reuse an existing checkout, set `REDCODE_ROOT`, `OPENRT_ROOT`, `FCV_ROOT`, or `DTAP_ROOT` — these are read by the runners, so `setup_benchmarks.sh` still clones into `third_party/` regardless.

### 4. Running SED

Each command below is a small laptop smoke that exercises the full loop on a couple of episodes. Full paper runs are much larger — see the per-benchmark README linked under each block.

#### HarmBench

Precomputed adversarial prompts; data already ships in `evaluation/Harmbench/`.

```bash
python experiments/harmbench/run_sed_hb.py --limit 2 -v
python experiments/harmbench/eval_hb_results.py --limit 2 -v
```

See [`experiments/harmbench/README.md`](experiments/harmbench/README.md).

#### WildJailbreak

```bash
python experiments/wildjailbreak/run_sed_wjb.py --limit 1 -v
python experiments/wildjailbreak/eval_wjb_results.py --limit 1 -v
```

See [`experiments/wildjailbreak/README.md`](experiments/wildjailbreak/README.md).

#### AgentDojo

```bash
bash setup_benchmarks.sh agentdojo
USER_TASKS=user_task_0 INJECTION_TASKS=injection_task_1 \
  bash experiments/agentdojo/run_smoke_sed.sh
```

AgentDyn shares this path: `bash experiments/agentdojo/run_full_sed_agentdyn.sh`.

See [`experiments/agentdojo/README.md`](experiments/agentdojo/README.md).

#### AgentHarm

```bash
python experiments/agentharm/run_sed.py --task harmful --split val --limit 1 --no-update-memory
```

See [`experiments/agentharm/README.md`](experiments/agentharm/README.md).

#### Adaptive jailbreaks (AutoDAN-Turbo / X-Teaming)

```bash
bash setup_benchmarks.sh openrt
cd experiments/adaptive_jailbreak
python run.py --defense no_defense --attack autodan_turbo --limit 2 --budget 8 --seeds 0
```

See [`experiments/adaptive_jailbreak/README.md`](experiments/adaptive_jailbreak/README.md).

#### RedCode

Needs Docker **and the `redcode` image built** (see step 3). `experiments/redcode/redcode_eval.sh` is the full paper run (indices 1–25, 250 instances);
for a smoke, call the module directly with a narrower range:

```bash
bash setup_benchmarks.sh redcode
python -m evaluation.redcode_sed.SED \
  --mode sed --task_type python_eval \
  --start_id 1 --end_id 1 --max_instances 2 \
  --prompt_variants code_input \
  --results_dir outputs/redcode_smoke
```

#### FCV and DTap

Both need Docker.

```bash
bash setup_benchmarks.sh fcv  && bash experiments/fcv/run_smoke_cwe538_sed.sh
bash setup_benchmarks.sh dtap && bash experiments/dtap/run_smoke_sed.sh
```

See [`experiments/fcv/README.md`](experiments/fcv/README.md) and [`experiments/dtap/README.md`](experiments/dtap/README.md).

Hyperparameters from the paper appendix live in `sed/config.py` (K=3, c=2, k=5, τ_new=0.55, τ_merge=0.80, b=1, c_max=3).

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{le2026sed,
  title     = {Self-Evolving Defense: Continual Security Policy Learning for LLM Agents},
  author    = {Le, Minh Nhat and Gondi, Nisarga and Peng, Yibo and Ni, Ronghao
               and Jia, Limin and Chen, Beidi and Zheng, Haizhong},
  year      = {2026},
}
```

## Contact

We welcome feedback from the community as we continue to refine and extend SED. If you have questions or feedback, please reach out:

- **Email**: [nhatminhle@umass.edu](mailto:nhatminhle@umass.edu)
- **Website**: [https://infini-ai-lab.github.io/SED/](https://infini-ai-lab.github.io/SED/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
