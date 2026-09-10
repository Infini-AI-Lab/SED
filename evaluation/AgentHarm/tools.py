"""Inspect-AI solvers for AgentHarm tool setup."""

from __future__ import annotations

import os
import random
from typing import Literal

from inspect_ai.solver import Generate, Solver, TaskState, solver


TaskName = Literal["harmful", "benign", "chat"]


def get_benchmark_tool_names(task_name: Literal["harmful", "benign"]) -> list[str]:
    tool_dir = os.path.join(os.path.dirname(__file__), "benchmark", f"{task_name}_tools")
    return sorted(
        path[:-3]
        for path in os.listdir(tool_dir)
        if path.endswith(".py") and path != "__init__.py"
    )


@solver
def setup_agentharm_tools(
    task_name: TaskName,
    n_irrelevant_tools: int = 0,
) -> Solver:
    """Attach the target tools listed in each AgentHarm sample's metadata."""

    from evaluation.AgentHarm.benchmark import benign_tools, harmful_tools

    tools_module = harmful_tools if task_name == "harmful" else benign_tools
    all_tool_names = (
        get_benchmark_tool_names(task_name) if task_name in ("harmful", "benign") else []
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if task_name == "chat":
            return state

        metadata = state.metadata or {}
        target_functions = list(metadata.get("target_functions") or [])
        tools = [getattr(tools_module, name)() for name in target_functions]

        if n_irrelevant_tools > 0:
            candidates = [name for name in all_tool_names if name not in target_functions]
            sampled = random.sample(candidates, min(n_irrelevant_tools, len(candidates)))
            tools.extend(getattr(tools_module, name)() for name in sampled)

        random.shuffle(tools)
        state.tools = tools
        return state

    return solve
