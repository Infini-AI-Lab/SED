"""DRIFT pre-tool MCP hook for DTap (build_constraints + dynamic_validation)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dt_arena.src.hooks.task_context import get_task_context
from dt_arena.src.types.hooks import ToolCallContext, ToolCallResult

logger = logging.getLogger(__name__)

SED_ROOT = Path(os.environ.get("SED_ROOT", Path.home() / "nisargagondi/SED"))
if str(SED_ROOT) not in sys.path:
    sys.path.insert(0, str(SED_ROOT))

from experiments.defenses.drift_agentic_guard import DriftAgenticGuard  # noqa: E402

REFUSAL = "I'm not able to help with that request."

_guard_instance: DriftAgenticGuard | None = None
_task_constraints: dict[str, dict] = {}


def _get_guard() -> DriftAgenticGuard:
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = DriftAgenticGuard()
    return _guard_instance


async def ensure_drift_context(instruction: str, malicious_goal: str | None = None) -> str:
    """Build per-task DRIFT constraints and return system augmentation text."""
    tctx = get_task_context()
    task_id = str(tctx.get("task_id") or "dtap")
    guard = _get_guard()
    if task_id not in _task_constraints:
        constraints = await asyncio.to_thread(
            guard.build_constraints, instruction or "", malicious_goal=malicious_goal or ""
        )
        _task_constraints[task_id] = constraints
    else:
        guard.constraints = _task_constraints[task_id]
    return guard.system_augmentation(_task_constraints[task_id])


class DriftBlocked(Exception):
    """Raised from pre-hook to stop a tool call before MCP dispatch."""


class DriftHook:
    """Pre-tool guard: dynamic_validation on malicious tasks only."""

    def __init__(self) -> None:
        self._init_lock = asyncio.Lock()

    async def _ensure_task(self, instruction: str, malicious_goal: str, task_id: str) -> DriftAgenticGuard:
        guard = _get_guard()
        async with self._init_lock:
            if task_id not in _task_constraints:
                constraints = await asyncio.to_thread(
                    guard.build_constraints, instruction or "", malicious_goal=malicious_goal or ""
                )
                _task_constraints[task_id] = constraints
            else:
                guard.constraints = _task_constraints[task_id]
        return guard

    async def on_pre_tool_call(self, ctx: ToolCallContext) -> Optional[ToolCallContext]:
        if os.environ.get("DTAP_DEFENSE", "") != "drift":
            return ctx
        tctx = get_task_context()
        # Benign tasks: constraints injected at turn level; skip per-tool blocking.
        if tctx.get("task_split") == "benign" or not tctx.get("malicious_goal"):
            ctx.metadata["drift"] = "skip_benign"
            return ctx

        instruction = tctx.get("instruction", "")
        malicious_goal = tctx.get("malicious_goal", "")
        task_id = str(tctx.get("task_id") or ctx.trace_id or "dtap")
        guard = await self._ensure_task(instruction, malicious_goal, task_id)
        assistant_text = tctx.get("assistant_text", "") or ""

        blocked, reason = await asyncio.to_thread(
            guard.should_block_tool,
            instruction,
            ctx.tool_name,
            dict(ctx.arguments or {}),
            malicious_goal=malicious_goal or "",
            assistant_text=assistant_text,
        )
        ctx.metadata["drift"] = "block" if blocked else "allow"
        ctx.metadata["drift_reason"] = reason[:500]
        if not blocked:
            return ctx
        block_msg = f"{REFUSAL} (DRIFT blocked {ctx.server}/{ctx.tool_name})"
        ctx.metadata["drift_blocked"] = True
        raise DriftBlocked(block_msg)

    async def on_post_tool_call(
        self, ctx: ToolCallContext, result: ToolCallResult
    ) -> Optional[ToolCallResult]:
        return result
