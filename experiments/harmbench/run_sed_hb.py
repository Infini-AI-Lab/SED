#!/usr/bin/env python3
"""
Run HarmBench adversarial prompts through the SED pipeline.

Loop per example:
  retrieve -> run target model -> judge -> memorize

The target model sees the HarmBench `test_case` verbatim. The judge receives the
behavior text (the HarmBench goal) so it scores whether the response actually
carried out the underlying harmful behavior rather than the wrapper text.

python experiments/harmbench/run_sed_hb.py \
  --chat-model accounts/fireworks/models/deepseek-v4-flash-0731 \
  --output experiments/harmbench/sed_hb_results.jsonl

Smoke test:
python experiments/harmbench/run_sed_hb.py --limit 20 \
  --chat-model accounts/fireworks/models/deepseek-v4-flash-0731
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "harmbench"
DATASET_DIR = REPO_ROOT / "evaluation" / "Harmbench"
DEFAULT_INPUT = DATASET_DIR / "adversarial_1400.jsonl"
DEFAULT_BEHAVIORS = DATASET_DIR / "harmbench_behaviors_text_all.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "sed_hb_results.jsonl"
DEFAULT_DATA_DIR = EXPERIMENT_DIR


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
    parser = argparse.ArgumentParser(description="Run HarmBench adversarial prompts through SED.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Adversarial JSONL.")
    parser.add_argument("--behaviors", default=str(DEFAULT_BEHAVIORS), help="HarmBench behaviors CSV (for goal text).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL path for run traces.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory for this run's SED attack memory files.")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to run.")
    parser.add_argument("--start", type=int, default=0, help="Start offset after loading examples.")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose result_id already appears in the output file.")
    parser.add_argument("--no-update-memory", action="store_true", help="Judge but do not write harmful failures into SED memory.")
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers. >1 requires --no-update-memory (read-only memory).")
    parser.add_argument("--flat", action="store_true",
                        help="Ablation: synthesis writes a flat store (no tree manager).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between examples.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_behaviors(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return {b["BehaviorID"]: b for b in csv.DictReader(f)}


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


def build_agent(args: argparse.Namespace) -> DefenseAgent:
    if not args.fireworks_api_key:
        raise ValueError("Set FIREWORKS_API_KEY or pass --fireworks-api-key.")
    if not args.chat_model:
        raise ValueError("Set FIREWORKS_MODEL or pass --chat-model.")
    if not args.memory_model:
        raise ValueError("Set FIREWORKS_MEMORY_MODEL/FIREWORKS_MODEL or pass --memory-model.")

    client = OpenAI(api_key=args.fireworks_api_key, base_url=args.base_url)
    policy_synthesizer = PolicySynthesizer(llm_client=client, model=args.memory_model)

    data_dir = Path(args.data_dir).expanduser().resolve()
    attack_memory_dir = data_dir / "attack_memory"
    attack_memory_dir.mkdir(parents=True, exist_ok=True)

    agent = DefenseAgent(
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
    agent.memory_manager.flat = args.flat
    return agent


def build_judge(args: argparse.Namespace) -> Judge:
    if not args.fireworks_api_key:
        raise ValueError("Set FIREWORKS_API_KEY or pass --fireworks-api-key.")
    if not args.judge_model:
        raise ValueError("Set FIREWORKS_JUDGE_MODEL/JUDGE_MODEL or pass --judge-model.")
    client = OpenAI(api_key=args.fireworks_api_key, base_url=args.base_url)
    return Judge(llm_client=client, judge_model=args.judge_model)


# One agent/judge per worker thread, so concurrent examples never share mutable
# chat state. Safe only with a read-only (frozen) memory — see the --workers guard.
_thread_local = threading.local()


def _worker_agent(args: argparse.Namespace) -> DefenseAgent:
    if not hasattr(_thread_local, "agent"):
        _thread_local.agent = build_agent(args)
    return _thread_local.agent


def _worker_judge(args: argparse.Namespace) -> Judge:
    if not hasattr(_thread_local, "judge"):
        _thread_local.judge = build_judge(args)
    return _thread_local.judge


def make_result_id(index: int) -> str:
    return str(index)


def run_one(
    *,
    agent: DefenseAgent,
    judge: Judge,
    index: int,
    row: dict[str, Any],
    behaviors: dict[str, dict[str, str]],
    update_memory: bool,
) -> dict[str, Any]:
    adversarial = str(row.get("test_case") or "")
    if not adversarial.strip():
        raise ValueError(f"Row {index} has no test_case field.")

    behavior_id = str(row.get("behavior") or "")
    meta = behaviors.get(behavior_id, {})
    goal = meta.get("Behavior", "") or adversarial

    conversation_id = f"hb-{index}-{uuid.uuid4().hex[:8]}"
    agent.reset_history()
    agent._current_conversation_id = conversation_id

    started = time.time()
    response = agent.chat(adversarial)
    conversation = list(agent.history)

    judge_result = judge.judge(conversation_history=conversation, goal=None)
    judge_dict = asdict(judge_result)

    retrieved_policy_ids = list(agent.last_policy_ids)

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
        "result_id": make_result_id(index),
        "index": index,
        "mode": "sed",
        "conversation_id": conversation_id,
        "method": row.get("method"),
        "behavior": behavior_id,
        "vanilla": goal,
        "context": meta.get("ContextString", ""),
        "contextual": "context" in meta.get("Tags", ""),
        "adversarial": adversarial,
        "source_models": row.get("source_models"),
        "weight": row.get("weight"),
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

    behaviors = load_behaviors(Path(args.behaviors).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()
    completed = loaded_result_ids(output_path) if args.resume else set()

    rows = load_jsonl(Path(args.input).expanduser().resolve())
    examples = [(index, row) for index, row in enumerate(rows) if index >= args.start]
    if args.limit is not None:
        examples = examples[: args.limit]

    if args.workers > 1 and not args.no_update_memory:
        raise SystemExit("--workers > 1 requires --no-update-memory (memory writes are not parallel-safe).")

    update_memory = not args.no_update_memory
    write_lock = threading.Lock()

    logging.info("Running %d HB example(s) | workers=%d | output=%s",
                 len(examples), args.workers, output_path)

    def work(index: int, row: dict[str, Any]) -> None:
        result_id = make_result_id(index)
        if result_id in completed:
            return
        try:
            result = run_one(
                agent=_worker_agent(args),
                judge=_worker_judge(args),
                index=index,
                row=row,
                behaviors=behaviors,
                update_memory=update_memory,
            )
        except Exception as exc:
            logging.exception("Failed on %s", result_id)
            result = {
                "result_id": result_id,
                "index": index,
                "mode": "sed",
                "method": row.get("method"),
                "behavior": row.get("behavior"),
                "adversarial": str(row.get("test_case") or ""),
                "error": str(exc),
            }
        with write_lock:
            append_jsonl(output_path, [result])
        if args.sleep:
            time.sleep(args.sleep)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, index, row) for index, row in examples]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Running HB SED", unit="example"):
            pass

    logging.info("Done. Results written to %s", output_path)

    if examples and output_path.exists():
        rows = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        # Only count rows from this invocation's result_ids.
        wanted = {make_result_id(index) for index, _ in examples}
        batch = [r for r in rows if r.get("result_id") in wanted]
        n_err = sum(1 for r in batch if r.get("error"))
        if batch and n_err == len(batch):
            raise SystemExit(
                f"All {n_err} HarmBench example(s) failed. "
                "Check FIREWORKS_API_KEY and FIREWORKS_MODEL (see .env.example)."
            )


if __name__ == "__main__":
    main()
