#!/usr/bin/env python3
"""Run mini-SWE-agent on SWE-bench with SED policy injection (FCV Pass 2)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import traceback
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FCV_ROOT = Path(os.environ.get("FCV_ROOT", _REPO_ROOT / "third_party" / "FCV"))
_FCV_MINISWE_SRC = _FCV_ROOT / "mini-swe-agent" / "src"
if str(_FCV_MINISWE_SRC) not in sys.path:
    sys.path.insert(0, str(_FCV_MINISWE_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sed.judge import Judge  # noqa: E402
from sed.synthesizer import PolicySynthesizer  # noqa: E402
from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent  # noqa: E402
from evaluation.FCV.sed_miniswe_agent import build_sed_agent_class  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments import get_environment  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.extra.swebench import (  # noqa: E402
    DATASET_MAPPING,
    get_sb_environment,
    get_swebench_docker_image_name,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager  # noqa: E402
from minisweagent.run.utils.save import save_traj  # noqa: E402
from minisweagent.utils.log import add_file_handler, logger  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

_OUTPUT_FILE_LOCK = threading.Lock()
_SHARED_DEFENSE: DefenseAgent | None = None
_SHARED_JUDGE: Judge | None = None
_SHARED_LOCK = threading.Lock()
_SHARED_MEMORY_LOCK = threading.Lock()

_RETRY_MARKERS = (
    "returned non-zero exit status",
    "CalledProcessError",
    "timed out after",
    "Unknown environment type:",
    "Input/output error",
    "This model isn't mapped yet.",
)


def _apply_docker_platform(config: dict) -> None:
    """Use linux/amd64 SWE-bench images reliably on Apple Silicon."""
    env_cfg = config.setdefault("environment", {})
    run_args = list(env_cfg.get("run_args", ["--rm"]))
    if "--platform" not in run_args:
        env_cfg["run_args"] = ["--platform", "linux/amd64", *run_args]


def _filter_instances_for_resume(
    instances: list[dict],
    preds_path: Path,
    *,
    redo_existing: bool,
) -> list[dict]:
    if redo_existing or not preds_path.exists():
        return instances

    existing = json.loads(preds_path.read_text())
    completed: list[str] = []
    retryable: list[str] = []
    for instance_id, row in existing.items():
        patch = str(row.get("model_patch", ""))
        if any(marker in patch for marker in _RETRY_MARKERS):
            retryable.append(instance_id)
        else:
            completed.append(instance_id)

    if completed:
        logger.info(f"Skipping {len(completed)} completed instances in preds.json")
    if retryable:
        logger.info(f"Retrying {len(retryable)} failed instances from preds.json")

    return [i for i in instances if i["instance_id"] not in completed]


def _resolve_paths(cwe_type: str, pipeline_name: str) -> Path:
    root = _REPO_ROOT / "experiments" / "fcv" / "sed_memory" / pipeline_name / cwe_type
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_sed_stack(chat_model: str, memory_root: Path) -> tuple[DefenseAgent, Judge]:
    api_key = os.environ["FIREWORKS_API_KEY"]
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
        timeout=float(os.environ.get("FCV_REQUEST_TIMEOUT_SECONDS", "600")),
    )
    synthesizer = PolicySynthesizer(
        llm_client=client,
        model=os.environ.get("FIREWORKS_MEMORY_MODEL", chat_model),
        log_dir=str(memory_root / "synth_log"),
    )
    defense = DefenseAgent(
        fireworks_api_key=api_key,
        chat_model=chat_model,
        memory_model=os.environ.get("FIREWORKS_MEMORY_MODEL", chat_model),
        policy_synthesizer=synthesizer,
        attack_memory_jsonl_l1=str(memory_root / "layer1.jsonl"),
        attack_memory_npy_l1=str(memory_root / "layer1.npy"),
        attack_memory_jsonl_l2=str(memory_root / "layer2.jsonl"),
        attack_memory_npy_l2=str(memory_root / "layer2.npy"),
        system_prompt=DEFENSE_BASE_PROMPT,
    )
    parse_fail_harmful = os.environ.get("SED_FCV_JUDGE_PARSE_FAIL", "benign").lower() != "benign"
    judge = Judge(
        llm_client=client,
        judge_model=os.environ.get("JUDGE_MODEL", chat_model),
        parse_failure_assumes_harmful=parse_fail_harmful,
    )
    return defense, judge


def _get_shared_stack(chat_model: str, memory_root: Path) -> tuple[DefenseAgent, Judge]:
    global _SHARED_DEFENSE, _SHARED_JUDGE
    with _SHARED_LOCK:
        if _SHARED_DEFENSE is None or _SHARED_JUDGE is None:
            _SHARED_DEFENSE, _SHARED_JUDGE = _build_sed_stack(chat_model, memory_root)
        return _SHARED_DEFENSE, _SHARED_JUDGE


class ProgressTrackingSEDAgent(build_sed_agent_class(DefaultAgent)):
    def __init__(self, *args, progress_manager: RunBatchProgressManager, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})",
        )
        return super().step()


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    *,
    chat_model: str,
    memory_root: Path,
    workers: int,
) -> None:
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    defense, judge = _get_shared_stack(chat_model, memory_root)
    memory_lock = _SHARED_MEMORY_LOCK if workers > 1 else None
    judge_mode = os.environ.get("SED_FCV_JUDGE_MODE", "end").strip().lower()
    if judge_mode not in {"end", "step"}:
        logger.warning(f"Unknown SED_FCV_JUDGE_MODE={judge_mode!r}, using 'end'")
        judge_mode = "end"
    defer_l2_persist = os.environ.get("SED_FCV_DEFER_L2_PERSIST", "1") != "0"

    agent = None
    extra_info = None

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingSEDAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            defense_agent=defense,
            judge=judge,
            memory_lock=memory_lock,
            judge_mode=judge_mode,
            defer_l2_persist=defer_l2_persist,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)
        extra_info = {
            "sed_policy_ids": agent.last_policy_ids,
            "sed_judge_events": agent.judge_events,
            "l1_size": defense.memory_manager.l1_size(),
            "l2_size": defense.memory_manager.l2_size(),
        }
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict],
    *,
    filter_spec: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    if filter_spec:
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
        logger.info(f"Instance filter: {before_filter} -> {len(instances)} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench batch run with SED")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--filter", default="", dest="filter_spec")
    parser.add_argument("--slice", default="", dest="slice_spec")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--pipeline-name", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    _apply_docker_platform(config)
    model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})
    api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key and not model_kwargs.get("api_key"):
        model_kwargs["api_key"] = api_key
    chat_model = config.get("model", {}).get("model_name", os.environ.get("FIREWORKS_MODEL", ""))
    if chat_model.startswith("fireworks_ai/"):
        chat_model = chat_model.replace("fireworks_ai/", "", 1)
    pipeline_name = args.pipeline_name or chat_model.split("/")[-1]
    cwe_type = config.get("agent", {}).get("cwe_type") or os.environ.get("CWE_TYPE", "cwe_538")
    memory_root = _resolve_paths(cwe_type, pipeline_name)

    args.output.mkdir(parents=True, exist_ok=True)
    add_file_handler(args.output / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    logger.info(f"Loading dataset {dataset_path}, split {args.split}")
    instances = list(load_dataset(dataset_path, split=args.split))
    instances = filter_instances(
        instances,
        filter_spec=args.filter_spec,
        slice_spec=args.slice_spec,
    )

    instances = _filter_instances_for_resume(
        instances,
        args.output / "preds.json",
        redo_existing=args.redo_existing,
    )
    if not instances:
        logger.info("No instances left to run.")
        return

    progress_manager = RunBatchProgressManager(len(instances), args.output / "progress.json")
    judge_mode = os.environ.get("SED_FCV_JUDGE_MODE", "end")
    logger.info(
        f"Running {len(instances)} instances with SED "
        f"(workers={args.workers}, judge_mode={judge_mode})"
    )

    if args.workers <= 1:
        for instance in instances:
            process_instance(
                instance,
                args.output,
                config,
                progress_manager,
                chat_model=chat_model,
                memory_root=memory_root,
                workers=args.workers,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    process_instance,
                    instance,
                    args.output,
                    config,
                    progress_manager,
                    chat_model=chat_model,
                    memory_root=memory_root,
                    workers=args.workers,
                )
                for instance in instances
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()


if __name__ == "__main__":
    main()
