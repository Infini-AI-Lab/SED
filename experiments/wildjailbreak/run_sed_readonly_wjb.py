#!/usr/bin/env python3
"""Run WildJailbreak prompts with read-only SED memory injection.

This runner is for fixed-memory evaluation, especially benign over-refusal tests:
it retrieves and injects L2 policies, calls the target model, and writes JSONL
results. It does not judge, update L1/L2 memory, or change attribution counters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sed.memory import MemoryManager  # noqa: E402
from sed.store import MemoryStore  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "evaluation" / "wildjailbreak"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "wildjailbreak"
DEFAULT_DATA_DIR = EXPERIMENT_DIR
DEFAULT_OUTPUT = EXPERIMENT_DIR / "sed_readonly_wjb_results.jsonl"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant.\n"

_thread_local = threading.local()


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
    parser = argparse.ArgumentParser(description="Run WildJailbreak with read-only SED memory.")
    parser.add_argument(
        "--split",
        choices=["harmful", "benign", "all"],
        default="benign",
        help="WildJailbreak split to run.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional JSONL path. Defaults to evaluation/wildjailbreak/<split>.jsonl.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL path for run traces.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing attack_memory/layer{1,2}.{jsonl,npy}.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to run.")
    parser.add_argument("--start", type=int, default=0, help="Start offset within each loaded split.")
    parser.add_argument("--resume", action="store_true", help="Skip result_ids already in output.")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent API workers.")
    parser.add_argument(
        "--model",
        default=os.environ.get("FIREWORKS_MODEL"),
        help="Fireworks target model. Defaults to FIREWORKS_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FIREWORKS_API_KEY"),
        help="Fireworks API key. Defaults to FIREWORKS_API_KEY.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--l1-top-k", type=int, default=5)
    parser.add_argument("--l2-top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
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


def make_result_id(split: str, index: int) -> str:
    return f"{split}/{index}"


def load_examples(args: argparse.Namespace) -> list[tuple[str, int, dict[str, Any]]]:
    examples: list[tuple[str, int, dict[str, Any]]] = []
    for split, path in resolve_input_paths(args):
        rows = load_jsonl(path)
        for index, row in enumerate(rows):
            if index < args.start:
                continue
            examples.append((split, index, row))
    if args.limit is not None:
        examples = examples[: args.limit]
    return examples


def l2_content(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata", {})
    detection = metadata.get("detection", [])
    detection_text = " ".join(detection) if isinstance(detection, list) else ""
    return " ".join(filter(None, [
        metadata.get("description", ""),
        metadata.get("scope", ""),
        detection_text,
        metadata.get("harm", ""),
    ]))


def get_client(args: argparse.Namespace) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        _thread_local.client = client
    return client


def get_memory_manager(args: argparse.Namespace) -> MemoryManager:
    manager = getattr(_thread_local, "memory_manager", None)
    if manager is None:
        attack_memory_dir = Path(args.data_dir).expanduser().resolve() / "attack_memory"
        l1_store = MemoryStore(
            jsonl_path=str(attack_memory_dir / "layer1.jsonl"),
            embed_path=str(attack_memory_dir / "layer1.npy"),
            fireworks_api_key=args.api_key,
        )
        l2_store = MemoryStore(
            jsonl_path=str(attack_memory_dir / "layer2.jsonl"),
            embed_path=str(attack_memory_dir / "layer2.npy"),
            fireworks_api_key=args.api_key,
            content_fn=l2_content,
        )
        manager = MemoryManager(
            l1_store=l1_store,
            l2_store=l2_store,
            l2_top_k=args.l2_top_k,
        )
        _thread_local.memory_manager = manager
    return manager


def build_system_prompt(base_prompt: str, attack_context: str) -> str:
    if attack_context:
        return base_prompt.rstrip() + "\n\n" + attack_context
    return base_prompt


def get_policy_names(memory_manager: MemoryManager, policy_ids: list[str]) -> list[str]:
    names: list[str] = []
    for policy_id in policy_ids:
        entry = memory_manager.l2.get(policy_id) if memory_manager.l2 is not None else None
        metadata = entry.get("metadata", {}) if entry else {}
        names.append(str(metadata.get("policy_name") or metadata.get("name") or policy_id))
    return names


def call_model(args: argparse.Namespace, system_prompt: str, adversarial: str) -> str:
    client = get_client(args)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": adversarial},
        ],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content or ""


def run_one(args: argparse.Namespace, split: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    result_id = make_result_id(split, index)
    adversarial = str(row.get("adversarial") or row.get("instruction") or row.get("prompt") or "")
    vanilla = str(row.get("vanilla") or row.get("goal") or adversarial)
    if not adversarial.strip():
        raise ValueError(f"Row {result_id} has no adversarial/instruction/prompt field.")

    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            memory_manager = get_memory_manager(args)
            attack_context, policy_ids = memory_manager.retrieve_attack_context_readonly(adversarial)
            policy_names = get_policy_names(memory_manager, policy_ids)
            if args.verbose:
                logging.info(
                    "Retrieved for %s: %s",
                    result_id,
                    ", ".join(policy_names) if policy_names else "(none)",
                )
            response = call_model(args, build_system_prompt(args.system_prompt, attack_context), adversarial)
            return {
                "result_id": result_id,
                "split": split,
                "index": index,
                "mode": "sed_readonly",
                "model": args.model,
                "vanilla": vanilla,
                "adversarial": adversarial,
                "response": response,
                "retrieved_policy_ids": policy_ids,
                "retrieved_policy_names": policy_names,
                "memory_context_injected": bool(attack_context),
                "memory_updated": False,
                "duration_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - keep batch jobs moving.
            last_error = str(exc)
            if attempt >= args.retries:
                break
            delay = args.retry_base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            logging.warning(
                "Retrying %s after error on attempt %d/%d: %s",
                result_id,
                attempt + 1,
                args.retries + 1,
                exc,
            )
            time.sleep(delay)

    return {
        "result_id": result_id,
        "split": split,
        "index": index,
        "mode": "sed_readonly",
        "model": args.model,
        "vanilla": vanilla,
        "adversarial": adversarial,
        "error": last_error,
        "memory_updated": False,
        "duration_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        raise ValueError("Set FIREWORKS_API_KEY or pass --api-key.")
    if not args.model:
        raise ValueError("Set FIREWORKS_MODEL or pass --model.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    output_path = Path(args.output).expanduser().resolve()
    completed = loaded_result_ids(output_path) if args.resume else set()
    examples = [
        example for example in load_examples(args)
        if make_result_id(example[0], example[1]) not in completed
    ]

    logging.info(
        "Running %d read-only SED WJB example(s) | model=%s | workers=%d | output=%s",
        len(examples),
        args.model,
        args.workers,
        output_path,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, args, split, index, row): (split, index)
            for split, index, row in examples
        }
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running WJB SED read-only",
            unit="example",
        )
        for future in progress:
            split, index = futures[future]
            result_id = make_result_id(split, index)
            progress.set_postfix_str(result_id)
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - defensive fallback.
                logging.exception("Unhandled worker failure for %s", result_id)
                result = {
                    "result_id": result_id,
                    "split": split,
                    "index": index,
                    "mode": "sed_readonly",
                    "model": args.model,
                    "error": str(exc),
                    "memory_updated": False,
                }

            append_jsonl(output_path, [result])

    logging.info("Done. Results written to %s", output_path)


if __name__ == "__main__":
    main()
