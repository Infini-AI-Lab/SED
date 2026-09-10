#!/usr/bin/env bash
# Edit the flags below, then: ./run.sh
set -e
cd "$(dirname "$0")"

# ─── edit these ───────────────────────────────────────────────
DEFENSE=sed        # no_defense | static | sed
ATTACK=xteaming             # direct | autodan_turbo | pair | tap | xteaming
SEEDS="0 1 2"             # e.g. "0 1 2"
LIMIT=                   # how many behaviors; empty = all 320
BUDGET=50                 # target queries per behavior (ignored for direct)
WORKERS=1                 # per-behavior parallel workers (non-lifelong only; auto-forced to 1 for lifelong/shards)
SHARDS=1                 # parallel lifelong streams; N-x faster, each learns only from its shard (use this for lifelong)
ATTACKER_LIFELONG=yes     # yes | no  — keep strategies across behaviors
DEFENSE_LIFELONG=yes      # yes | no  — let SED memory evolve (SED only)
FLAT_MEMORY=no             # yes | no  — SED: flat policy store (no tree manager); ignores RETRIEVAL_CAP/PARENT_PULL
RETRIEVAL_CAP=2           # SED retrieval: max leaves per shared parent (0 = off)
PARENT_PULL=yes           # yes | no  — SED retrieval: also inject each leaf's nearest retrievable ancestor
JUDGE_GOAL_HINT=yes        # yes | no  — let SED's memory judge see the attack goal as a hint (no = content-only)
FROZEN_MEMORY=            # path to an attack_memory dir to load READ-ONLY (forces non-lifelong); empty = off
LOAD_STRATEGIES=data/warm_up_strategy_library.json          # path to a strategy library to hot-start (skips warm-up); empty = off
RESUME=no                 # yes | no  — skip behaviors already in the output file
DEFENSE_HINT=no           # yes | no  — tell the attacker what the defense screens for (gray-box)
OUT=results2               # output dir
BEHAVIORS=../../evaluation/Harmbench/harmbench_behaviors_text_test.csv   # behaviors CSV; empty = full test set
# ──────────────────────────────────────────────────────────────

[[ -n "$FROZEN_MEMORY" ]] && DEFENSE_LIFELONG=no   # frozen memory is read-only → non-lifelong (parallel-safe)
[[ "$ATTACKER_LIFELONG" == yes || "$DEFENSE_LIFELONG" == yes ]] && WORKERS=1
[[ "$SHARDS" -gt 1 ]] && WORKERS=1   # shards and per-behavior workers are mutually exclusive

args=(--defense "$DEFENSE" --attack "$ATTACK" --seeds $SEEDS --budget "$BUDGET" --max-workers "$WORKERS" --shards "$SHARDS" --out "$OUT")
[[ -n "$LIMIT" ]]            && args+=(--limit "$LIMIT")
[[ "$ATTACKER_LIFELONG" == yes ]] && args+=(--attacker-lifelong)
[[ "$DEFENSE_LIFELONG" == yes ]]  && args+=(--defense-lifelong)
[[ "$RESUME" == yes ]]      && args+=(--resume)
[[ "$DEFENSE_HINT" == yes ]] && args+=(--defense-hint)
[[ "$FLAT_MEMORY" == yes ]]  && args+=(--flat-memory)
[[ "$FLAT_MEMORY" != yes && "$RETRIEVAL_CAP" -gt 0 ]] && args+=(--retrieval-cap "$RETRIEVAL_CAP")
[[ "$FLAT_MEMORY" != yes && "$PARENT_PULL" == yes ]]  && args+=(--parent-pull)
[[ "$JUDGE_GOAL_HINT" == yes ]] && args+=(--judge-goal-hint)
[[ -n "$FROZEN_MEMORY" ]]    && args+=(--frozen-memory "$FROZEN_MEMORY")
[[ -n "$LOAD_STRATEGIES" ]]  && args+=(--load-strategies "$LOAD_STRATEGIES")

python run.py "${args[@]}"
