from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentdojo.agent_pipeline import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from sed.synthesizer import format_policies_for_injection


class SEDPolicyInjector(BasePipelineElement):
    def __init__(self, defense_agent: Any) -> None:
        self.defense_agent = defense_agent
        self.conversation_id = "default"
        self.last_policy_ids: list[str] = []
        self.record_attribution = True

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        policy_entries = self.defense_agent.l2_store.get_all()
        policy_ids = [entry["id"] for entry in policy_entries]
        policy_metas = [entry["metadata"] for entry in policy_entries]
        self.last_policy_ids = policy_ids

        policy_block = format_policies_for_injection(policy_metas)
        if not policy_block:
            return query, runtime, env, messages, extra_args

        if not messages or messages[0].get("role") != "system":
            raise ValueError("SEDPolicyInjector must run after SystemMessage and before InitQuery.")

        updated_messages = list(messages)
        system_content = _content_to_text(updated_messages[0].get("content"))
        updated_messages[0] = {
            **updated_messages[0],
            "content": [text_content_block_from_string(f"{system_content.rstrip()}\n\n{policy_block}")],
        }
        return query, runtime, env, updated_messages, extra_args


class MessageCapture(BasePipelineElement):
    """Capture final AgentDojo messages after a run."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        self.messages = list(messages)
        return query, runtime, env, messages, extra_args


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("content") or block.get("text") or ""))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)
