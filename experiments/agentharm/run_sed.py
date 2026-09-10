"""CLI wrapper for running AgentHarm with SED via Inspect-AI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = REPO_ROOT / "evaluation" / "AgentHarm" / "tasks.py"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
MIN_OPENAI_VERSION = (2, 40, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentHarm / AgentHarmBenign with SED.")
    parser.add_argument("--task", choices=["harmful", "benign", "both"], default="both")
    parser.add_argument("--defense", choices=["sed", "none"], default="sed",
                        help="'sed' runs the full defense; 'none' runs the no-defense baseline.")
    parser.add_argument("--split", choices=["val", "test_public", "test_private"], default="test_public")
    parser.add_argument("--target-model", default=os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"))
    parser.add_argument("--memory-model", default=os.environ.get("FIREWORKS_MEMORY_MODEL"))
    parser.add_argument("--synth-model", default=None)
    parser.add_argument("--memory-dir", default="experiments/agentharm/sed_memory/default")
    parser.add_argument("--log-dir", default="experiments/agentharm/runs")
    parser.add_argument("--behavior-id", action="append", dest="behavior_ids", default=[])
    parser.add_argument("--detailed-behaviors", choices=["true", "false"], default=None)
    parser.add_argument("--hint-included", choices=["true", "false"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-irrelevant-tools", type=int, default=0)
    parser.add_argument("--tool-choice", choices=["auto", "none", "forced_first"], default="auto")
    parser.add_argument("--policy-mode", choices=["auto", "retrieve", "all"], default="auto")
    parser.add_argument("--max-connections", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--no-update-memory", action="store_true")
    parser.add_argument("--flat-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print inspect commands without running them.")
    return parser.parse_args()


def check_openai_version() -> None:
    try:
        import openai
    except ImportError:
        print(
            "Missing dependency: openai. Install dependencies with `pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    version = tuple(int(part) for part in openai.__version__.split(".")[:3])
    if version < MIN_OPENAI_VERSION:
        print(
            "AgentHarm Inspect runs require openai>=2.40.0 "
            f"(found {openai.__version__}). Upgrade with:\n"
            "  pip install --upgrade 'openai>=2.40.0'",
            file=sys.stderr,
        )
        raise SystemExit(1)


def fireworks_model(model: str) -> str:
    if model.startswith(("fireworks/", "openai/", "openai-api/")):
        return model
    return f"fireworks/{model}"


def policy_mode(mode: str, task_name: str, no_update_memory: bool) -> str:
    if mode != "auto":
        return mode
    return "all" if task_name == "benign" and no_update_memory else "retrieve"


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=True)
    args = parse_args()
    if not args.dry_run:
        check_openai_version()
    tasks = ["harmful", "benign"] if args.task == "both" else [args.task]
    args.memory_dir = os.path.abspath(args.memory_dir)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("FIREWORKS_BASE_URL", FIREWORKS_BASE_URL)
    if not args.dry_run and not env.get("FIREWORKS_API_KEY"):
        raise SystemExit("FIREWORKS_API_KEY is required. Set it in .env or the shell.")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    for task_name in tasks:
        suffix = {"none": "_baseline", "sed": "_sed"}[args.defense]
        inspect_task = f"agentharm{suffix}" if task_name == "harmful" else f"agentharm_benign{suffix}"
        command = [
            "inspect",
            "eval",
            f"{TASK_FILE}@{inspect_task}",
            "--model",
            fireworks_model(args.target_model),
            "--log-dir",
            args.log_dir,
            "--max-connections",
            str(args.max_connections),
            "--max-tokens",
            str(args.max_tokens),
            "-T",
            f"split={args.split}",
            "-T",
            f"refusal_judge={fireworks_model(args.judge_model)}",
            "-T",
            f"semantic_judge={fireworks_model(args.judge_model)}",
            "-T",
            f"n_irrelevant_tools={args.n_irrelevant_tools}",
            "-T",
            f"tool_choice={args.tool_choice}",
        ]
        if args.defense == "sed":
            command.extend([
                "-T", f"target_model={args.target_model}",
                "-T", f"judge_model={args.judge_model}",
                "-T", f"memory_dir={args.memory_dir}",
                "-T", f"policy_mode={policy_mode(args.policy_mode, task_name, args.no_update_memory)}",
                "-T", f"update_memory={str(not args.no_update_memory).lower()}",
            ])
            if args.memory_model:
                command.extend(["-T", f"memory_model={args.memory_model}"])
            if args.synth_model:
                command.extend(["-T", f"synth_model={args.synth_model}"])
            if args.flat_memory:
                command.extend(["-T", "flat_memory=true"])
        if args.detailed_behaviors is not None:
            command.extend(["-T", f"detailed_behaviors={args.detailed_behaviors}"])
        if args.hint_included is not None:
            command.extend(["-T", f"hint_included={args.hint_included}"])
        if args.limit is not None:
            command.extend(["-T", f"limit={args.limit}"])
        if args.behavior_ids:
            command.extend(["-T", f"behavior_ids={','.join(args.behavior_ids)}"])

        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
