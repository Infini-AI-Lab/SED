#!/usr/bin/env python3
"""
Run WildJailbreak prompts through the SED pipeline.

Loop per example:
  retrieve -> run target model -> judge -> memorize

The target model sees the adversarial WildJailbreak prompt. The judge receives
the vanilla prompt as the goal so it can decide whether the response complied
with the underlying harmful/benign request instead of the wrapper text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from dotenv import load_dotenv
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sed.judge import Judge  # noqa: E402
from sed.synthesizer import PolicySynthesizer  # noqa: E402
from sed.agent import DefenseAgent  # noqa: E402

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "wildjailbreak"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "sed_wjb_results.jsonl"
DEFAULT_DATA_DIR = EXPERIMENT_DIR
DEFAULT_DATASET_DIR = REPO_ROOT / "evaluation" / "wildjailbreak"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WildJailbreak through SED.")
    parser.add_argument(
        "--split",
        choices=["harmful", "benign", "all"],
        default="harmful",
        help="WildJailbreak split to run.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional JSONL path. Defaults to evaluation/wildjailbreak/<split>.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSONL path for run traces.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory for this run's SED attack memory files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to run.")
    parser.add_argument("--start", type=int, default=0, help="Start offset after loading examples.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose result_id already appears in the output file.",
    )
    parser.add_argument(
        "--no-update-memory",
        action="store_true",
        help="Judge but do not write harmful failures into SED memory.",
    )
    parser.add_argument(
        "--chat-model",
        default=os.environ.get("FIREWORKS_MODEL"),
        help="Fireworks target model. Defaults to FIREWORKS_MODEL.",
    )
    parser.add_argument(
        "--memory-model",
        default=os.environ.get("FIREWORKS_MEMORY_MODEL") or os.environ.get("FIREWORKS_MODEL"),
        help="Fireworks model used by the policy synthesizer.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("FIREWORKS_JUDGE_MODEL")
        or os.environ.get("JUDGE_MODEL")
        or os.environ.get("FIREWORKS_MODEL"),
        help="OpenAI-compatible judge model. Defaults to FIREWORKS_JUDGE_MODEL/JUDGE_MODEL/FIREWORKS_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        help="OpenAI-compatible API base URL for all clients.",
    )
    parser.add_argument(
        "--fireworks-api-key",
        default=os.environ.get("FIREWORKS_API_KEY"),
        help="Fireworks API key. Defaults to FIREWORKS_API_KEY.",
    )
    parser.add_argument("--l1-top-k", type=int, default=5)
    parser.add_argument("--l2-top-k", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between examples.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def loaded_result_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("result_id"):
                ids.add(str(item["result_id"]))
    return ids


def resolve_input_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.input:
        split = args.split if args.split != "all" else "custom"
        return [(split, Path(args.input).expanduser().resolve())]
    if args.split == "all":
        return [
            ("harmful", DEFAULT_DATASET_DIR / "harmful.jsonl"),
            ("benign", DEFAULT_DATASET_DIR / "benign.jsonl"),
        ]
    return [(args.split, DEFAULT_DATASET_DIR / f"{args.split}.jsonl")]


def build_agent(args: argparse.Namespace) -> DefenseAgent:
    if not args.fireworks_api_key:
        raise ValueError("Set FIREWORKS_API_KEY or pass --fireworks-api-key.")
    if not args.chat_model:
        raise ValueError("Set FIREWORKS_MODEL or pass --chat-model.")
    if not args.memory_model:
        raise ValueError("Set FIREWORKS_MEMORY_MODEL/FIREWORKS_MODEL or pass --memory-model.")

    client = OpenAI(
        api_key=args.fireworks_api_key,
        base_url=args.base_url,
    )
    policy_synthesizer = PolicySynthesizer(llm_client=client, model=args.memory_model)

    data_dir = Path(args.data_dir).expanduser().resolve()
    attack_memory_dir = data_dir / "attack_memory"
    attack_memory_dir.mkdir(parents=True, exist_ok=True)

    return DefenseAgent(
        fireworks_api_key=args.fireworks_api_key,
        chat_model=args.chat_model,
        memory_model=args.memory_model,
        attack_memory_jsonl_l1=str(attack_memory_dir / "layer1.jsonl"),
        attack_memory_npy_l1=str(attack_memory_dir / "layer1.npy"),
        attack_memory_jsonl_l2=str(attack_memory_dir / "layer2.jsonl"),
        attack_memory_npy_l2=str(attack_memory_dir / "layer2.npy"),
        policy_synthesizer=policy_synthesizer,
        l2_top_k=args.l2_top_k,
    )


def build_judge(args: argparse.Namespace) -> Judge:
    if not args.fireworks_api_key:
        raise ValueError("Set FIREWORKS_API_KEY or pass --fireworks-api-key.")
    if not args.judge_model:
        raise ValueError("Set FIREWORKS_JUDGE_MODEL/JUDGE_MODEL or pass --judge-model.")
    client = OpenAI(api_key=args.fireworks_api_key, base_url=args.base_url)
    return Judge(llm_client=client, judge_model=args.judge_model)


def make_result_id(split: str, index: int) -> str:
    return f"{split}/{index}"


def run_one(
    *,
    agent: DefenseAgent,
    judge: Judge,
    split: str,
    index: int,
    row: dict[str, Any],
    update_memory: bool,
) -> dict[str, Any]:
    adversarial = str(row.get("adversarial") or row.get("instruction") or row.get("prompt") or "")
    vanilla = str(row.get("vanilla") or row.get("goal") or adversarial)
    if not adversarial.strip():
        raise ValueError(f"Row {split}/{index} has no adversarial/instruction/prompt field.")

    conversation_id = f"wjb-{split}-{index}-{uuid.uuid4().hex[:8]}"
    agent.reset_history()
    agent._current_conversation_id = conversation_id

    started = time.time()
    response = agent.chat(adversarial)
    conversation = list(agent.history)

    judge_result = judge.judge(
        conversation_history=conversation,
        goal=vanilla,
    )
    judge_dict = asdict(judge_result)

    retrieved_policy_ids = list(
        agent.memory_manager._retrieval_buffer.get(conversation_id, [])
    )

    memory_updated = False
    if update_memory and judge_result.should_store:
        agent.update_attack_memory(
            conversation_id=conversation_id,
            conversation_history=conversation,
            judge_result=judge_result,
            full_conversation=conversation,
        )
        memory_updated = True

    return {
        "result_id": make_result_id(split, index),
        "split": split,
        "index": index,
        "conversation_id": conversation_id,
        "vanilla": vanilla,
        "adversarial": adversarial,
        "response": response,
        "judge": judge_dict,
        "retrieved_policy_ids": retrieved_policy_ids,
        "memory_updated": memory_updated,
        "duration_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)
    args = parse_args()
    setup_logging(args.verbose)

    output_path = Path(args.output).expanduser().resolve()
    completed = loaded_result_ids(output_path) if args.resume else set()

    paths = resolve_input_paths(args)
    examples: list[tuple[str, int, dict[str, Any]]] = []
    for split, path in paths:
        rows = load_jsonl(path)
        for index, row in enumerate(rows):
            if index < args.start:
                continue
            examples.append((split, index, row))

    if args.limit is not None:
        examples = examples[: args.limit]

    agent = build_agent(args)
    judge = build_judge(args)

    logging.info(
        "Running %d WJB example(s) | target=%s | judge=%s | output=%s",
        len(examples),
        agent.chat_model,
        judge.model,
        output_path,
    )

    progress = tqdm(examples, desc="Running WJB SED", unit="example")
    for split, index, row in progress:
        result_id = make_result_id(split, index)
        progress.set_postfix_str(result_id)
        if result_id in completed:
            continue

        try:
            result = run_one(
                agent=agent,
                judge=judge,
                split=split,
                index=index,
                row=row,
                update_memory=not args.no_update_memory,
            )
        except Exception as exc:
            logging.exception("Failed on %s", result_id)
            adversarial = str(row.get("adversarial") or row.get("instruction") or row.get("prompt") or "")
            vanilla = str(row.get("vanilla") or row.get("goal") or adversarial)
            result = {
                "result_id": result_id,
                "split": split,
                "index": index,
                "vanilla": vanilla,
                "adversarial": adversarial,
                "error": str(exc),
            }

        append_jsonl(output_path, [result])
        if args.sleep:
            time.sleep(args.sleep)

    logging.info("Done. Results written to %s", output_path)

    if examples and output_path.exists():
        wanted = {make_result_id(split, index) for split, index, _ in examples}
        rows = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        batch = [r for r in rows if r.get("result_id") in wanted]
        n_err = sum(1 for r in batch if r.get("error"))
        if batch and n_err == len(batch):
            raise SystemExit(
                f"All {n_err} WildJailbreak example(s) failed. "
                "Check FIREWORKS_API_KEY and FIREWORKS_MODEL (see .env.example)."
            )


if __name__ == "__main__":
    main()
