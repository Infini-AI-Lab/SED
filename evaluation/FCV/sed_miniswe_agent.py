"""SED-wrapped mini-SWE-agent for FCV coding-agent attacks."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from contextlib import nullcontext
from typing import Any, Literal

from sed.synthesizer import format_policies_for_injection
from evaluation.FCV.cwe_goals import judge_goal_for_cwe

logger = logging.getLogger(__name__)

JudgeMode = Literal["end", "step"]


def build_sed_agent_class(default_agent_cls: type) -> type:
    """Create a DefaultAgent subclass with SED policy injection + judge updates."""

    class SEDDefaultAgent(default_agent_cls):  # type: ignore[misc,valid-type]
        def __init__(
            self,
            *args,
            defense_agent: Any,
            judge: Any,
            instance_id: str = "",
            record_attribution: bool = True,
            memory_lock: threading.Lock | None = None,
            judge_mode: JudgeMode = "end",
            defer_l2_persist: bool = True,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.defense_agent = defense_agent
            self.judge = judge
            self._memory_lock = memory_lock
            self.instance_id = instance_id
            self.record_attribution = record_attribution
            self.judge_mode = judge_mode
            self.defer_l2_persist = defer_l2_persist
            self.conversation_id = f"fcv:{instance_id}:{uuid.uuid4().hex[:8]}"
            self._base_system_template = self.config.system_template
            self.last_policy_ids: list[str] = []
            self.judge_events: list[dict] = []
            self.defense_agent._current_conversation_id = self.conversation_id

        def run(self, task: str, **kwargs) -> tuple[str, str]:
            try:
                return super().run(task, **kwargs)
            finally:
                if self.judge_mode == "end":
                    lock = self._memory_lock or nullcontext()
                    with lock:
                        self._judge_and_maybe_store()
                else:
                    self.defense_agent.memory_manager.flush_attribution(
                        self.conversation_id,
                        outcome=False,
                    )

        def step(self) -> dict:
            lock = self._memory_lock or nullcontext()
            with lock:
                self._inject_policies_into_system_message()
            response = super().step()
            if self.judge_mode == "step":
                with lock:
                    self._judge_and_maybe_store()
            return response

        def _inject_policies_into_system_message(self) -> None:
            if not self.messages or self.messages[0].get("role") != "system":
                return

            policy_entries = self.defense_agent.l2_store.get_all()
            policy_ids = [entry["id"] for entry in policy_entries]
            policy_metas = [entry["metadata"] for entry in policy_entries]
            self.last_policy_ids = policy_ids

            if self.record_attribution and policy_entries:
                for entry in policy_entries:
                    metadata = entry["metadata"]
                    metadata["times_retrieved"] = metadata.get("times_retrieved", 0) + 1
                    if not self.defer_l2_persist:
                        self.defense_agent.l2_store.update(entry["id"], "")
                self.defense_agent.memory_manager.record_retrieval(
                    self.conversation_id,
                    policy_ids,
                )

            policy_block = format_policies_for_injection(policy_metas)
            base_system = self.render_template(self._base_system_template)
            system_content = (
                base_system if not policy_block else f"{base_system.rstrip()}\n\n{policy_block}"
            )
            self.messages[0] = {**self.messages[0], "content": system_content}

        def _judge_and_maybe_store(self) -> None:
            history = copy.deepcopy(self.messages)
            goal = judge_goal_for_cwe(self.config.cwe_type)
            judge_result = self.judge.judge(history, goal=goal)

            self.judge_events.append(
                {
                    "instance_id": self.instance_id,
                    "conversation_id": self.conversation_id,
                    "is_harmful": judge_result.is_harmful,
                    "should_store": judge_result.should_store,
                    "policy_ids": list(self.last_policy_ids),
                }
            )

            if judge_result.should_store:
                self.defense_agent.update_attack_memory(
                    self.conversation_id,
                    history,
                    judge_result,
                    full_conversation=history,
                )
            else:
                self.defense_agent.memory_manager.flush_attribution(
                    self.conversation_id,
                    outcome=False,
                )

    return SEDDefaultAgent
