#!/usr/bin/env bash
# redcode_eval.sh — Run the RedCode benchmark against the Self-Evolving Defense Agent
# Usage: bash experiments/redcode/redcode_eval.sh
#
# Comment out any flags you don't need.

set -euo pipefail

# ── Load .env ──────────────────────────────────────────────────────────────
# Exports FIREWORKS_API_KEY, FIREWORKS_MODEL, FIREWORKS_MEMORY_MODEL, HF_TOKEN
set -a
source "$(dirname "$0")/../../.env"
set +a

# ── Clean up any leftover redcode container from a previous interrupted run ─
docker rm -f redcode 2>/dev/null || true

# ── Run ────────────────────────────────────────────────────────────────────
python -m evaluation.redcode_sed.SED \
  \
  `# ── Run mode ──────────────────────────────────────────────────────────` \
  `# sed        — full Self-Evolving Defense (L2 policies + memory writes)` \
  `# baseline   — bare ReAct agent, no safety prompt, no memory` \
  `# static     — RedCode's SAFETY_SYS_SUFFIX4, no memory` \
  `# --mode baseline` \
  --mode sed \
  \
  `# ── Guard model post-filter (optional, works with any mode) ──────────` \
  `# Apply Guard as a post-filter to catch unsafe outputs.` \
  `# Works with any mode: sed, baseline, or static.` \
  \
  `# ── Task type ─────────────────────────────────────────────────────────` \
  `# python_eval — RedCode Python benchmark (task_type 1-25)` \
  `# bash_eval   — RedCode Bash benchmark (task_type 1-25)` \
  `# mixed_eval  — Unified RedCode + AutoDAN interleaved (read from --mixed_file)` \
  --task_type mixed_eval \
  \
  `# ── Task index range (inclusive) ──────────────────────────────────────` \
  `# Omit both to run all indices for the chosen task_type.` \
  `# Python indices: 1-9, 10-25 (see _PYTHON_IDS in SED.py)` \
  `# Bash   indices: 1-9, 12-16, 18, 20-23, 25` \
  `# --shuffle_ids` \
  `# --shuffle_seed 42` \
  --start_id 1 \
  --end_id 25 \
  \
  `# ── Instances per index  ────────────────────────────` \
  --max_instances 250 \
  \
  `# ── Prompt variants ───────────────────────────────────────────────────` \
  `# Default: all four variants per instance.` \
  `# Choices: code_input  code_input_jailbreaking  text_summary  text_details` \
  `# --prompt_variants code_input` \
   --prompt_variants code_input \
  \
  `# ── Mixed eval file ──────────────────────────────────────────────────` \
  --mixed_file evaluation/redcode_sed/data/mixed_eval_dataset_2_python_test.jsonl \
  \
  `# ── Retrieval mode (SED mode only) ───────────────────────────────────` \
  `# --retrieval semantic` \
  `# --retrieval llm` \
  `# --retrieval oracle` \
  `# --retrieval reranker` \
  --retrieval semantic \
  \
  `# ── Flat-memory ablation (SED mode only) ─────────────────────────────` \
  `# Synthesis writes policies to a flat store instead of the L2 tree.` \
  `# --flat_memory` \
  \
  `# ── Models ────────────────────────────────────────────────────────────` \
  `# All default to env vars; override here if needed.` \
  `# Judge/memory are pinned to deepseek so only the target model changes —` \
  `# JUDGE_MODEL is unset in .env, so an unpinned judge would follow the target.` \
  `# --fireworks_model accounts/fireworks/models/deepseek-v4-flash` \
  --fireworks_model accounts/fireworks/models/deepseek-v4-flash \
  --fireworks_memory_model accounts/fireworks/models/deepseek-v4-flash \
  --judge_model accounts/fireworks/models/deepseek-v4-flash \
  \
  `# ── Output directory ──────────────────────────────────────────────────` \
  `# Use separate dirs for each mode so plots can compare them:` \
  `#   --results_dir results/SED          # for --mode sed` \
  --results_dir experiments/redcode/runs/sed_dsv4flash \
  \
  `# ── Resume a previous mixed_eval run (mixed_eval only) ───────────────` \
  `# Pass the log file from a prior interrupted run to skip already-processed` \
  `# entries and append new results to the same log/res files.` \
  `# --results_dir must still point to the same run directory (for attack_memory).` \
  `# --log_file results/mixed_eval/log/mixed_log_2026-04-13_23-31-06.json` \
  \
  `# Skip the judge LLM entirely. Guard interceptions drive policy synthesis` \
  `# (harm metadata fields left empty; policy synth works from conv text).` \
  `# --guard_as_judge` \
  \
  `# ── RedCode ground-truth judge context (SED mode only) ───────────────` \
  `# Keep the judge LLM, but include RedCode score labels in {goal_line}` \
  `# so policy synthesis metadata is enriched by benchmark ground truth.` \
  `# --redcode_judge_goal` \
  --redcode_judge_goal \
  `# ── Dry run (no Docker, no LLM calls — just checks plumbing) ──────────` \
  `# --dry_run` \
  \
  `# ── Dataset version tag (written into result JSON) ────────────────────` \
  --version v1 \
  2>&1 | grep -v "Starting\|All processes started.\|Address already\|in use\|Press CTRL\|Serving Flask\|Debug mode\|Debugger\|Restarting\|Warning\|WSGI"
