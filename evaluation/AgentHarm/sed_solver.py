"""SED solver adapter for Inspect-AI AgentHarm tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
import uuid

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message
from inspect_ai.tool import ToolFunction

from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent


ToolChoice = Literal["auto", "none", "forced_first"]
PolicyMode = Literal["retrieve", "all"]


@dataclass
class SEDAgentHarmConfig:
    tool_choice: ToolChoice = "auto"
    policy_mode: PolicyMode = "retrieve"
    system_prompt: str = DEFENSE_BASE_PROMPT
    max_tool_messages: int = 20
    conversation_id_prefix: str = "agentharm"


@solver
def sed_agentharm_solver(
    defense_agent: DefenseAgent,
    config: SEDAgentHarmConfig | None = None,
) -> Solver:
    """Inject retrieved SED policies, then delegate the tool loop to Inspect."""

    config = config or SEDAgentHarmConfig()

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        conversation_id = _conversation_id(config, state)
        query = _state_user_text(state)
        attack_context, policy_ids = _retrieve_policy_context(
            defense_agent, query, config.policy_mode
        )

        _set_metadata(state, "_sed_conversation_id", conversation_id)
        _set_metadata(state, "_sed_policy_mode", config.policy_mode)
        _set_metadata(state, "_sed_policy_ids", list(policy_ids))

        system_content = _build_system_prompt(config.system_prompt, attack_context)
        state = await system_message(system_content)(state, generate_fn)

        if config.tool_choice == "forced_first":
            target_functions = (state.metadata or {}).get("target_functions") or []
            if target_functions:
                state.tool_choice = ToolFunction(name=target_functions[0])
            state = await generate(tool_calls="single")(state, generate_fn)
            state.tool_choice = "auto"
            return await generate(tool_calls="loop")(state, generate_fn)

        state.tool_choice = config.tool_choice
        return await generate(tool_calls="loop")(state, generate_fn)

    return solve


@solver
def agentharm_baseline_solver(config: SEDAgentHarmConfig | None = None) -> Solver:
    """No-defense baseline: a plain tool-using agent — no policy injection, no
    memory. Mirrors the SED solver's tool_choice handling so the two are
    comparable, but the system prompt carries no safety instructions."""

    config = config or SEDAgentHarmConfig()

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        state = await system_message(config.system_prompt)(state, generate_fn)

        if config.tool_choice == "forced_first":
            target_functions = (state.metadata or {}).get("target_functions") or []
            if target_functions:
                state.tool_choice = ToolFunction(name=target_functions[0])
            state = await generate(tool_calls="single")(state, generate_fn)
            state.tool_choice = "auto"
            return await generate(tool_calls="loop")(state, generate_fn)

        state.tool_choice = config.tool_choice
        return await generate(tool_calls="loop")(state, generate_fn)

    return solve


def _retrieve_policy_context(
    defense_agent: DefenseAgent, query: str, policy_mode: PolicyMode
) -> tuple[str, list[str]]:
    if policy_mode == "all":
        return _all_policy_context(defense_agent)
    return defense_agent.memory_manager.retrieve_attack_context(query)


def _all_policy_context(defense_agent: DefenseAgent) -> tuple[str, list[str]]:
    entries = defense_agent.memory_manager.l2.get_all()
    policy_ids = [entry["id"] for entry in entries]
    policy_metas = [entry["metadata"] for entry in entries]
    if not policy_metas:
        return "", []
    from sed.synthesizer import format_policies_for_injection

    return format_policies_for_injection(policy_metas), policy_ids


def _build_system_prompt(base_prompt: str, attack_context: str) -> str:
    if attack_context:
        return base_prompt.rstrip() + "\n\n" + attack_context
    return base_prompt


def _conversation_id(config: SEDAgentHarmConfig, state: TaskState) -> str:
    sample_id = getattr(state, "sample_id", None) or (state.metadata or {}).get("id") or "sample"
    return f"{config.conversation_id_prefix}:{sample_id}:{uuid.uuid4()}"


def _state_user_text(state: TaskState) -> str:
    for attr in ("input_text", "user_prompt", "input"):
        value = getattr(state, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    for message in getattr(state, "messages", []) or []:
        if getattr(message, "role", None) == "user":
            text = getattr(message, "text", None)
            if text:
                return str(text)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content

    metadata_prompt = (state.metadata or {}).get("prompt")
    return str(metadata_prompt or "")


def _set_metadata(state: TaskState, key: str, value: Any) -> None:
    if state.metadata is None:
        state.metadata = {}
    state.metadata[key] = value
