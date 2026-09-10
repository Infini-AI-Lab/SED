"""
Target Agent
============
Safety-focused target agent backed by Fireworks AI and evolving attack memory.
"""

import os
import logging
from typing import List, Dict, Optional

from openai import OpenAI

from sed import config
from sed.store import MemoryStore
from sed.memory import MemoryManager

logger = logging.getLogger(__name__)

# ─── Defense system prompt ────────────────────────────────────────────────────

DEFENSE_BASE_PROMPT = (
    "You are a helpful AI assistant.\n"
)


class DefenseAgent:
    """
    Safety-focused agent with evolving attack memory.

    DefenseAgent maintains only attack memory: L1 scored episodes plus an L2
    policy library. L2 policies are retrieved during chat and organized into a
    hierarchy only when new policies are written.

    Timing contract (enforced by SelfPlayRunner, not this class):
      chat()                — intra-test-time, READ-ONLY from memory
      update_attack_memory() — inter-test-time, WRITE, called after judge scores

    """

    def __init__(
        self,
        fireworks_api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        memory_model: Optional[str] = None,
        # Attack memory file paths (stores are created automatically)
        attack_memory_jsonl_l1: str = "data/attack_memory/layer1.jsonl",
        attack_memory_npy_l1: str = "data/attack_memory/layer1.npy",
        attack_memory_jsonl_l2: str = "data/attack_memory/layer2.jsonl",
        attack_memory_npy_l2: str = "data/attack_memory/layer2.npy",
        embed_model: Optional[str] = None,
        policy_synthesizer=None,  # PolicySynthesizer — injected dependency
        l2_top_k: int = config.K_INJECT,
        retrieval_cap: int = config.FAMILY_CAP,
        parent_pull: bool = config.PARENT_PULL,
        max_history_turns: int = config.MAX_HISTORY_TURNS,
        system_prompt: str = DEFENSE_BASE_PROMPT,
    ):
        api_key = fireworks_api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(
                "Fireworks API key required. Pass fireworks_api_key= or set FIREWORKS_API_KEY."
            )

        resolved_chat_model = chat_model or os.environ.get("FIREWORKS_MODEL", "")
        if not resolved_chat_model:
            raise ValueError(
                "No model specified. Pass chat_model= or set FIREWORKS_MODEL."
            )

        resolved_memory_model = (
            memory_model
            or os.environ.get("FIREWORKS_MEMORY_MODEL", "")
            or resolved_chat_model
        )

        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )
        self.chat_model        = resolved_chat_model
        self.memory_model      = resolved_memory_model
        self.max_history_turns = max_history_turns
        self.base_system_prompt = system_prompt

        store_kwargs = dict(fireworks_api_key=api_key, **({"model_name": embed_model} if embed_model else {}))

        self.l1_store = MemoryStore(
            jsonl_path=attack_memory_jsonl_l1,
            embed_path=attack_memory_npy_l1,
            **store_kwargs,
        )

        def _l2_content_fn(entry: dict) -> str:
            m = entry.get("metadata", {})
            detection = m.get("detection", [])
            detection_text = " ".join(detection) if isinstance(detection, list) else ""
            return " ".join(filter(None, [
                (m.get("policy_name", "") or "").replace("-", " "),
                m.get("description", ""),
                m.get("scope", ""),
                detection_text,
                m.get("harm", ""),
            ]))

        self.l2_store = MemoryStore(
            jsonl_path=attack_memory_jsonl_l2,
            embed_path=attack_memory_npy_l2,
            content_fn=_l2_content_fn,
            **store_kwargs,
        )

        self.memory_manager = MemoryManager(
            l1_store=self.l1_store,
            l2_store=self.l2_store,
            policy_synthesizer=policy_synthesizer,
            l2_top_k=l2_top_k,
            retrieval_cap=retrieval_cap,
            parent_pull=parent_pull,
        )

        self.history: List[Dict[str, str]] = []
        self.last_policy_ids: List[str] = []
        # Set by some runners for result bookkeeping; memory writes receive the
        # explicit conversation_id passed to update_attack_memory().
        self._current_conversation_id: str = "default"

        logger.info(
            f"DefenseAgent ready | chat_model={self.chat_model} | "
            f"L1={len(self.l1_store)} entries | L2={len(self.l2_store)} entries"
        )

    # ------------------------------------------------------------------
    # Core chat method  (intra-test-time: READ-ONLY)
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        prebuilt_context: Optional[tuple] = None,
    ) -> str:
        """
        Process one user turn. Retrieves L2 policy context and injects it into
        the system prompt before calling the LLM.

        Args:
            user_message:     the incoming user turn
            prebuilt_context: optional (attack_context_str, policy_ids) tuple. When
                              supplied, skips semantic retrieval and uses this context
                              directly (oracle / perfect-retrieval mode).
        """
        if prebuilt_context is not None:
            attack_context, policy_ids = prebuilt_context
        else:
            query = self._build_retrieval_query(user_message)
            attack_context, policy_ids = self.memory_manager.retrieve_attack_context(query)
        self.last_policy_ids = list(policy_ids)
        system_content = self._build_defense_prompt(attack_context)

        trimmed  = self._trim_history()
        messages = (
            [{"role": "system", "content": system_content}]
            + trimmed
            + [{"role": "user", "content": user_message}]
        )

        try:
            resp  = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                reasoning_effort=None
            )
            reply = resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------
    # Attack memory write  (inter-test-time: WRITE)
    # ------------------------------------------------------------------

    def update_attack_memory(
        self,
        conversation_id: str,
        conversation_history: List[Dict[str, str]],
        judge_result,   # JudgeResult — passes harm metadata + outcome in one object
        full_conversation: Optional[List[dict]] = None,
    ) -> None:
        """
        Persist the last scored turn to attack memory.

        All metadata (harm_description, failure mode/evidence fields, is_harmful)
        comes directly from JudgeResult. The first user message is used as the attack
        turn — in agentic sessions, later "user" messages are environment observations.

        Args:
            conversation_id:      unique id for this exchange (set by runner)
            conversation_history: interleaved [{"role": "user"|"assistant", "content": str}]
            judge_result:         JudgeResult from agent/judge.py
            full_conversation:    optional full message list (system+all turns) for multi-turn
                                  agentic sessions; stored in L1 so PolicySynthesizer can see
                                  the complete execution trace when generating policies.
        """
        # Extract attack turn: first user message is the original attack prompt.
        # In agentic multi-turn sessions, later "user" messages are environment
        # observations — using reversed() would capture those instead of the attack.
        last_user      = next(
            (m["content"] for m in conversation_history if m["role"] == "user"), ""
        )
        last_assistant = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"), ""
        )
        # Record which policies were retrieved for this turn so the L1 episode keeps them.
        self.memory_manager.record_retrieval(conversation_id, getattr(self, "last_policy_ids", []))
        self.memory_manager.process_attack(
            conversation_id=conversation_id,
            attack_turn=last_user,
            agent_response=last_assistant,
            harm_description=judge_result.harm_description,
            failure_mode=judge_result.failure_mode,
            failure_mode_description=judge_result.failure_mode_description,
            evidence_source=judge_result.evidence_source,
            evidence_excerpt=judge_result.evidence_excerpt,
            stable_signature=judge_result.stable_signature,
            requested_harmful_action=judge_result.requested_harmful_action,
            complied_action=judge_result.complied_action,
            outcome=judge_result.is_harmful,
            full_conversation=full_conversation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_retrieval_query(self, user_message: str) -> str:
        """
        Concatenate the last turn (2 history entries) with the current message.
        Using recent context (not just the current turn) is important for catching
        multi-turn escalation attacks where each individual turn looks benign.
        """
        recent = self._trim_history()[-2:]
        parts  = [m["content"] for m in recent] + [user_message]
        return " ".join(parts)

    def _build_defense_prompt(self, attack_context: str) -> str:
        """Attack context is injected immediately after the base prompt."""
        if attack_context:
            return self.base_system_prompt.rstrip() + "\n\n" + attack_context
        return self.base_system_prompt

    def _trim_history(self) -> List[Dict[str, str]]:
        """Keep only the last max_history_turns turns (each turn = 2 entries)."""
        max_entries = self.max_history_turns * 2
        return self.history[-max_entries:] if len(self.history) > max_entries else self.history

    def reset_history(self) -> None:
        """Clear in-process conversation history. Does NOT clear attack memory."""
        self.history = []
        logger.info("DefenseAgent conversation history cleared.")
