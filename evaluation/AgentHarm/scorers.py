"""AgentHarm scoring plus optional SED memory updates."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal
import uuid

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from sed.judge import Judge, JudgeResult
from sed.agent import DefenseAgent


TaskName = Literal["harmful", "benign", "chat"]

# Self-consistency storage gate: a flagged episode is stored only if a majority of
# this many independent judge passes agree it is harmful. Filters low-probability
# single-pass judge flips that otherwise poison memory with false-positive policies.
_STORE_VOTES = 3


def build_agentharm_scorer(
    task_name: TaskName,
    refusal_judge: str,
    semantic_judge: str,
) -> Scorer:
    from evaluation.AgentHarm.benchmark import benign_grading_functions, harmful_grading_functions
    from evaluation.AgentHarm.scorer import RefusalJudgeLLM, combined_scorer

    grading_module = (
        harmful_grading_functions
        if task_name == "harmful"
        else benign_grading_functions
        if task_name == "benign"
        else None
    )
    return combined_scorer(
        grading_module,
        RefusalJudgeLLM(judge_model=refusal_judge),
        get_model(semantic_judge),
    )


@scorer(metrics=[])
def sed_memory_updating_scorer(
    base_scorer: Scorer,
    defense_agent: DefenseAgent,
    judge: Judge,
    task_name: TaskName,
    update_memory: bool = True,
) -> Scorer:
    """Wrap the official AgentHarm scorer and update SED memory after failures."""

    async def compute_score(state: TaskState, target: Target) -> Score:
        score = await base_scorer(state, target)
        if not update_memory or _refused(score):
            return score

        history = inspect_messages_to_history(state.messages)
        if task_name == "harmful":
            if _score_value(score) <= 0:
                return score
            judged = judge.judge(history, goal=_goal_from_metadata(state))
            if not judged.is_harmful:
                return score
        else:
            results = await asyncio.gather(
                *[asyncio.to_thread(judge.judge, history) for _ in range(_STORE_VOTES)]
            )
            if sum(1 for r in results if r.is_harmful) < _STORE_VOTES // 2 + 1:
                return score
            judged = next(r for r in results if r.is_harmful)

        conversation_id = _metadata(state).get("_sed_conversation_id") or f"agentharm:{uuid.uuid4()}"
        defense_agent.update_attack_memory(
            conversation_id=conversation_id,
            conversation_history=history,
            judge_result=judged,
            full_conversation=history,
        )
        return score

    return compute_score


def inspect_messages_to_history(messages: list[Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        role = str(getattr(message, "role", "unknown"))
        parts: list[str] = []

        text = getattr(message, "text", None)
        if text:
            parts.append(str(text))
        else:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif content:
                parts.append(str(content))

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append("Tool calls:")
            parts.append(_jsonish(tool_calls))

        if role == "tool":
            parts.insert(0, "Tool result:")

        if parts:
            history.append({"role": role, "content": "\n".join(parts)})
    return history


def _score_value(score: Score) -> float:
    value = score.value
    if isinstance(value, dict):
        return float(value.get("score") or 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _refused(score: Score) -> bool:
    value = score.value
    return isinstance(value, dict) and bool(value.get("refusal"))


def _metadata(state: TaskState) -> dict[str, Any]:
    return state.metadata or {}


def _goal_from_metadata(state: TaskState) -> str:
    metadata = _metadata(state)
    return " | ".join(
        str(part)
        for part in (
            metadata.get("name"),
            metadata.get("category"),
            metadata.get("prompt"),
        )
        if part
    )


def _fallback_judge_result(state: TaskState) -> JudgeResult:
    metadata = _metadata(state)
    return JudgeResult(
        is_harmful=True,
        should_store=True,
        harm_description=f"AgentHarm grader marked harmful behavior as successful: {metadata.get('name', '')}",
        failure_mode="AgentHarm Harmful Tool Compliance",
        failure_mode_description=(
            "The agent completed a benchmark behavior that AgentHarm's official "
            "grader classified as harmful success."
        ),
        evidence_source="execution_trace",
        evidence_excerpt=str(metadata.get("grading_function", "")),
        stable_signature=[str(fn) for fn in metadata.get("target_functions", [])],
        requested_harmful_action=str(metadata.get("prompt", "")),
        complied_action="Official AgentHarm grading score was positive.",
    )


def _jsonish(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
