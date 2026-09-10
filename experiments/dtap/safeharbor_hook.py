"""SafeHarbor pre-tool MCP hook for DTap."""

from __future__ import annotations

import asyncio
import json
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

from dt_arena.src.hooks.guard_context import guard_malicious_goal  # noqa: E402

from experiments.defenses.safeharbor_guard import SafeHarborGuard  # noqa: E402

REFUSAL = "I'm not able to help with that request."

_guard: SafeHarborGuard | None = None


def _get_guard() -> SafeHarborGuard:
    global _guard
    if _guard is None:
        _guard = SafeHarborGuard()
    return _guard


async def ensure_safeharbor_context(instruction: str, existing_system: str = "") -> str:
    """Inject retrieve_query rules into the agent system prompt (AgentDojo parity)."""
    guard = _get_guard()
    return await asyncio.to_thread(guard.system_augmentation, instruction, existing_system)


class SafeHarborBlocked(Exception):
    """Raised from pre-hook to stop a tool call before MCP dispatch."""


class SafeHarborHook:
    """Pre-tool hard-block + system-message augmentation via task_runner."""

    def __init__(self) -> None:
        self._init_lock = asyncio.Lock()

    async def _ensure(self) -> SafeHarborGuard:
        async with self._init_lock:
            return _get_guard()

    def _tool_context(self, instruction: str, malicious_goal: str, ctx: ToolCallContext) -> str:
        lines = [f"Task: {instruction or '(unknown)'}"]
        if malicious_goal:
            lines.append(f"Adversarial test goal: {malicious_goal}")
        lines.append(
            f"Proposed MCP call: server={ctx.server} tool={ctx.tool_name} "
            f"args={json.dumps(ctx.arguments, ensure_ascii=False)[:2000]}"
        )
        return "\n".join(lines)

    async def on_pre_tool_call(self, ctx: ToolCallContext) -> Optional[ToolCallContext]:
        if os.environ.get("DTAP_DEFENSE", "") != "safeharbor":
            return ctx
        guard = await self._ensure()
        tctx = get_task_context()
        instruction = tctx.get("instruction", "")
        malicious_goal = guard_malicious_goal(tctx)
        user_input = self._tool_context(instruction, malicious_goal, ctx)
        hard_block = await asyncio.to_thread(guard.is_hard_block, user_input)
        ctx.metadata["safeharbor"] = "block" if hard_block else "allow"
        if not hard_block:
            return ctx
        block_msg = f"{REFUSAL} (SafeHarbor blocked {ctx.server}/{ctx.tool_name})"
        ctx.metadata["safeharbor_blocked"] = True
        raise SafeHarborBlocked(block_msg)

    async def on_post_tool_call(
        self, ctx: ToolCallContext, result: ToolCallResult
    ) -> Optional[ToolCallResult]:
        return result
