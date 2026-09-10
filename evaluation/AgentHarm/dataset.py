"""AgentHarm dataset loading helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from inspect_ai.dataset import Dataset, FieldSpec, json_dataset
from platformdirs import user_cache_dir


TaskName = Literal["harmful", "benign", "chat"]
SplitName = Literal["val", "test_public", "test_private"]


_SPLIT_FILES: dict[str, dict[str, str]] = {
    "harmful": {
        "val": "benchmark/harmful_behaviors_validation.json",
        "test_public": "benchmark/harmful_behaviors_test_public.json",
        "test_private": "benchmark/harmful_behaviors_test_private.json",
    },
    "benign": {
        "val": "benchmark/benign_behaviors_validation.json",
        "test_public": "benchmark/benign_behaviors_test_public.json",
        "test_private": "benchmark/benign_behaviors_test_private.json",
    },
    "chat": {
        "val": "benchmark/chat_validation.json",
        "test_public": "benchmark/chat_public_test.json",
        "test_private": "benchmark/chat_private_test.json",
    },
}


def load_agentharm_dataset(task_name: TaskName, split: SplitName) -> Dataset:
    cache_dir = Path(user_cache_dir("inspect_evals")) / "agentharm_dataset" / "AgentHarm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id="ai-safety-institute/AgentHarm",
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns="**/*.json",
            resume_download=True,
        )
    except Exception:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise

    input_path = cache_dir / _SPLIT_FILES[task_name][split]
    if not input_path.exists():
        raise FileNotFoundError(f"AgentHarm split file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    jsonl_path = input_path.with_suffix(".jsonl")
    try:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in data["behaviors"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return json_dataset(
            str(jsonl_path),
            FieldSpec(
                input="prompt",
                id="id",
                metadata=[
                    "id",
                    "id_original",
                    "detailed_prompt",
                    "hint_included",
                    "name",
                    "category",
                    "prompt",
                    "target_functions",
                    "grading_function",
                ],
            ),
        )
    finally:
        jsonl_path.unlink(missing_ok=True)


def filter_agentharm_dataset(
    dataset: Dataset,
    behavior_ids: list[str] | str | None = None,
    detailed_behaviors: bool | None = None,
    hint_included: bool | None = None,
    limit: int | None = None,
) -> Dataset:
    behavior_ids = _normalize_behavior_ids(behavior_ids)
    dataset_ids = [sample.id for sample in dataset]
    missing = [behavior_id for behavior_id in behavior_ids if behavior_id not in dataset_ids]
    if missing:
        raise ValueError(f"Behavior IDs not found in dataset: {missing}")

    if behavior_ids:
        dataset = dataset.filter(lambda sample: sample.id in behavior_ids)
    if detailed_behaviors is not None:
        dataset = dataset.filter(
            lambda sample: bool(sample.metadata)
            and sample.metadata["detailed_prompt"] == detailed_behaviors
        )
    if hint_included is not None:
        dataset = dataset.filter(
            lambda sample: bool(sample.metadata)
            and sample.metadata["hint_included"] == hint_included
        )
    if limit is not None:
        keep_ids = {sample.id for idx, sample in enumerate(dataset) if idx < limit}
        dataset = dataset.filter(lambda sample: sample.id in keep_ids)
    return dataset


def _normalize_behavior_ids(behavior_ids: list[str] | str | None) -> list[str]:
    if behavior_ids is None:
        return []
    if isinstance(behavior_ids, str):
        return [part.strip() for part in behavior_ids.split(",") if part.strip()]
    return behavior_ids
