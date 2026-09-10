"""
Memory Manager
==============
Manager for the attack memory used by DefenseAgent.

Attack memory pipeline:
  Layer 1 (episodic):       one entry per scored turn. Failed L1 episodes are
                            buffered as synthesis inputs.
  Layer 2 (policy library): one entry per security policy. L2 is persisted as a
                            MemoryStore, with tree parent links stored in metadata.
                            New synthesized policies are organized by the
                            hierarchical tree manager before being injected.

Write timing:
  - Attack:    process_attack()           — after judge scores (inter-test-time only)
  - Retrieval: retrieve_attack_context()  — during chat (READ-ONLY)

Retrieval:
  retrieve_attack_context() returns (context_block, policy_ids) for logging and
  prompt injection. It does not mutate L1 or L2. Retrieval is flat cosine search
  over the L2 policy surface; the hierarchy is used only when writing new L2
  policies.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from sed import config
from sed.store import MemoryStore
from sed.organizer import organize

logger = logging.getLogger(__name__)


# ─── Attack memory constants ──────────────────────────────────────────────────

_BATCH_SIZE     = config.SYNTH_BATCH
_TOP_K_L2       = config.K_INJECT
_TRUNCATE_TAIL  = config.TRUNCATE_TAIL


class MemoryManager:
    """Attack memory for DefenseAgent: episodic L1 + policy L2 over a policy tree."""

    def __init__(
        self,
        l1_store: Optional[MemoryStore] = None,
        l2_store: Optional[MemoryStore] = None,
        policy_synthesizer=None,   # PolicySynthesizer — injected to avoid circular import
        l2_top_k: int = _TOP_K_L2,
        retrieval_cap: int = 0,    # >0 → at most N retrieved leaves per shared parent (0 = off)
        parent_pull: bool = False, # True → also inject each retrieved leaf's nearest retrievable ancestor
    ):
        self.l1 = l1_store
        self.l2 = l2_store
        self.policy_synthesizer = policy_synthesizer
        self.l2_top_k = l2_top_k
        self.retrieval_cap = retrieval_cap
        self.parent_pull = parent_pull
        # SED_FLAT_MEMORY=1 → synthesis writes a flat store and the tree manager never
        # runs, so retrieval is plain cosine over a list. The ablation for whether the
        # hierarchy earns its keep.
        self.flat = os.environ.get("SED_FLAT_MEMORY", "0") == "1"

        # Maps conversation_id → policy IDs retrieved for it, so the L1 episode can
        # record which policies were active when written at process_attack() time.
        self._retrieval_buffer: Dict[str, List[str]] = {}

        # Failure buffer: L1 episode IDs from failed conversations.
        # Batch synthesis fires when len >= _BATCH_SIZE.
        # Persisted to disk so restarts don't lose pending failures.
        self._failure_buffer_path = (
            self.l1.jsonl_path.parent / "failure_buffer.json"
            if self.l1 is not None else None
        )
        self._failure_buffer: List[str] = self._load_failure_buffer()

    # ======================================================================
    # Attack memory — DefenseAgent API
    # ======================================================================
    # Failure buffer persistence
    # ======================================================================

    def _load_failure_buffer(self) -> List[str]:
        """Load persisted failure buffer from disk, or return empty list."""
        if self._failure_buffer_path is None or not self._failure_buffer_path.exists():
            return []
        try:
            data = json.loads(self._failure_buffer_path.read_text(encoding="utf-8"))
            buf = [str(x) for x in data if isinstance(x, str)]
            if buf:
                logger.info(f"Failure buffer restored: {len(buf)} pending failures")
            return buf
        except Exception as e:
            logger.warning(f"Could not load failure buffer: {e}")
            return []

    def _save_failure_buffer(self) -> None:
        """Persist current failure buffer to disk."""
        if self._failure_buffer_path is None:
            return
        try:
            self._failure_buffer_path.write_text(
                json.dumps(self._failure_buffer, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save failure buffer: {e}")

    # ======================================================================
    # All methods below require l1_store, l2_store, and policy_synthesizer to be set.
    # They raise RuntimeError if called without attack memory configured.

    def _require_attack_memory(self) -> None:
        if self.l1 is None or self.l2 is None or self.policy_synthesizer is None:
            raise RuntimeError(
                "Attack memory not configured. Pass l1_store, l2_store, and "
                "policy_synthesizer to MemoryManager.__init__ to enable attack memory."
            )

    def _require_attack_retrieval_memory(self) -> None:
        if self.l1 is None or self.l2 is None:
            raise RuntimeError(
                "Attack memory retrieval not configured. Pass l1_store and "
                "l2_store to MemoryManager.__init__ to enable attack retrieval."
            )

    # --- Intra-test-time (READ-ONLY) ---

    def retrieve_attack_context(self, query: str) -> Tuple[str, List[str]]:
        """
        Retrieve the top L2 policies for a query and format them for system-prompt injection.

        Returns (formatted_context_block, retrieved_policy_ids). Only L2 policies are
        injected — L1 episodes are stored for synthesis but never surfaced (raw attack
        snippets hurt safety). Read-only: no store is mutated.
        """
        self._require_attack_retrieval_memory()
        policy_metas: List[dict] = []
        returned_policy_ids: List[str] = []
        for pid in self._rank_attack_policy_ids(query):
            entry = self.l2.get(pid)
            if entry is None:
                continue
            returned_policy_ids.append(pid)
            policy_metas.append(entry["metadata"])
        if not returned_policy_ids:
            return "", []

        from sed.synthesizer import format_policies_for_injection
        return format_policies_for_injection(policy_metas), returned_policy_ids

    # Retrieval is already read-only; retained name for eval runners on a fixed snapshot.
    retrieve_attack_context_readonly = retrieve_attack_context

    def _rank_attack_policy_ids(self, query: str) -> List[str]:
        """Top L2 policy ids for a query: flat cosine over the policy surface embeddings.
        Category parents (retrievable == False) are skipped. Two optional tree-aware steps:
          - retrieval_cap>0: keep at most N leaves per shared parent (diversity across families);
          - parent_pull:     additively append each kept leaf's nearest retrievable ancestor,
                             so the general parent policy rides along with its specific case."""
        pool = max(self.l2_top_k * 4, 20)
        hits = self.l2.search(query, top_k=pool)
        ranked = [h["id"] for h in hits if h.get("metadata", {}).get("retrievable", True)]

        picked: List[str] = []
        per_family: dict = {}
        for pid in ranked:
            fam = self._policy_family(pid)
            if self.retrieval_cap and per_family.get(fam, 0) >= self.retrieval_cap:
                continue
            picked.append(pid)
            per_family[fam] = per_family.get(fam, 0) + 1
            if len(picked) >= self.l2_top_k:
                break

        if self.parent_pull:
            seen = set(picked)
            for pid in list(picked):
                anc = self._nearest_retrievable_ancestor(pid)
                if anc and anc not in seen:
                    seen.add(anc)
                    picked.append(anc)
        return picked

    def _policy_family(self, pid: str) -> str:
        """A leaf's family = its parent id, or its own id if it sits at root (own family)."""
        parent = (self.l2.get(pid) or {}).get("metadata", {}).get("parent", "root")
        return parent if (parent and self.l2.get(parent) is not None) else "solo:" + pid

    def _nearest_retrievable_ancestor(self, pid: str) -> Optional[str]:
        """Walk up parent links to the first ancestor with retrievable content (skip empty
        category parents). Returns None if none before root."""
        parent = (self.l2.get(pid) or {}).get("metadata", {}).get("parent", "root")
        while parent:
            pe = self.l2.get(parent)
            if pe is None:                                   # 'root' / dangling → stop
                return None
            if pe.get("metadata", {}).get("retrievable", True):
                return parent
            parent = pe.get("metadata", {}).get("parent", "root")
        return None

    def record_retrieval(self, conversation_id: str, retrieved_policy_ids: List[str]) -> None:
        """
        Buffer which L2 policies were injected for this conversation so the L1 episode
        can record them when written at process_attack() time. In-memory only.
        """
        self._retrieval_buffer[conversation_id] = list(retrieved_policy_ids)

    # --- Inter-test-time (WRITE) ---

    def process_attack(
        self,
        conversation_id: str,
        attack_turn: str,
        agent_response: str,
        harm_description: str,
        failure_mode: str,
        failure_mode_description: str,
        outcome: bool,
        full_conversation: Optional[List[dict]] = None,
        evidence_source: str = "",
        evidence_excerpt: str = "",
        stable_signature: Optional[List[str]] = None,
        requested_harmful_action: str = "",
        complied_action: str = "",
    ) -> None:
        """
        Post-conversation pipeline. Called after judge scores (inter-test-time).

        All metadata fields come directly from JudgeResult — no separate LLM call.

        Args:
            conversation_id:      unique id set by runner
            attack_turn:          last user message (the scored turn)
            agent_response:       last agent response (the scored turn)
            harm_description:     specific harm enabled (from JudgeResult)
            failure_mode:         short reusable name for how the failure happened
            failure_mode_description: one-sentence definition of the failure mode
            outcome:              True if the attack succeeded (agent produced harmful output)
            full_conversation:    optional full message list (system+user+assistant+env turns)
                                  for multi-turn agentic sessions; stored in L1 metadata so
                                  PolicySynthesizer can see the complete execution trace.
            evidence_source:      source containing strongest reusable evidence
            evidence_excerpt:     shortest exact excerpt revealing the issue
            stable_signature:     reusable markers/patterns for this failure family
            requested_harmful_action: harmful action requested in user/tool/trace content
            complied_action:      assistant/tool/execution action showing compliance

        Pipeline:
          1. Write the L1 episode for this scored turn.
          2. If the attack succeeded, buffer the episode and, once the batch is full,
             synthesize new policies and place them into the L2 tree.
        """
        self._require_attack_memory()

        retrieved_policy_ids = self._retrieval_buffer.pop(conversation_id, [])
        l1_id = self._write_l1(
            conversation_id=conversation_id,
            attack_turn=attack_turn,
            agent_response=agent_response,
            harm_description=harm_description,
            failure_mode=failure_mode,
            failure_mode_description=failure_mode_description,
            evidence_source=evidence_source,
            evidence_excerpt=evidence_excerpt,
            stable_signature=stable_signature or [],
            requested_harmful_action=requested_harmful_action,
            complied_action=complied_action,
            outcome=outcome,
            retrieved_policy_ids=retrieved_policy_ids,
            full_conversation=full_conversation,
        )

        if outcome:
            self._failure_buffer.append(l1_id)
            self._save_failure_buffer()
            logger.info(
                f"Failure buffer: {len(self._failure_buffer)}/{_BATCH_SIZE} "
                f"(outcome=True, l1_id={l1_id})"
            )
            if len(self._failure_buffer) >= _BATCH_SIZE:
                self._synthesize_policies()

    # --- Layer 1 helpers ---

    def _write_l1(
        self,
        conversation_id: str,
        attack_turn: str,
        agent_response: str,
        harm_description: str,
        failure_mode: str,
        failure_mode_description: str,
        outcome: bool,
        evidence_source: str = "",
        evidence_excerpt: str = "",
        stable_signature: Optional[List[str]] = None,
        requested_harmful_action: str = "",
        complied_action: str = "",
        retrieved_policy_ids: Optional[List[str]] = None,
        full_conversation: Optional[List[dict]] = None,
    ) -> str:
        """
        Write a new Layer 1 episodic entry. Returns the entry id.

        Embedding target: raw attack_turn (single last user message).
        Retrieval query is also last N user turns concatenated, so
        like-to-like embedding gives meaningful cosine similarity.
        """
        def _truncate_with_tail(text: str, head: int, tail: int) -> str:
            if len(text) <= head:
                return text
            return text[:head] + " […] " + text[-tail:]

        content = attack_turn
        metadata: Dict[str, Any] = {
            "conversation_id":       conversation_id,
            "agent_response":        _truncate_with_tail(agent_response, 800, _TRUNCATE_TAIL),
            "outcome":               outcome,
            "harm_description":      harm_description,
            "failure_mode":          failure_mode,
            "failure_mode_description": failure_mode_description,
            "evidence_source":       evidence_source,
            "evidence_excerpt":      evidence_excerpt,
            "stable_signature":      list(stable_signature or []),
            "requested_harmful_action": requested_harmful_action,
            "complied_action":       complied_action,
            "retrieved_policy_ids":  list(retrieved_policy_ids or []),
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }
        if full_conversation is not None:
            # Persist full trace for policy synthesis/debugging, but omit system
            # prompt entries to reduce noise and avoid storing injected prompt text.
            metadata["full_conversation"] = [
                msg for msg in full_conversation
                if msg.get("role") != "system"
            ]
        mem_id = self.l1.add(content, metadata=metadata)
        logger.info(
            f"L1 ADD [{mem_id}] conv={conversation_id} "
            f"outcome={'FAILED' if outcome else 'DEFENDED'} failure_mode={failure_mode[:40]}"
        )
        return mem_id

    # --- Layer 2 helpers ---

    def _synthesize_policies(self) -> None:
        """
        Batch synthesis from the failure buffer: propose policies from the failed
        episodes, then place each one into the L2 tree (organize handles dedup via
        MERGE and structure via SPLIT/PROMOTE).
        """
        if not self._failure_buffer:
            return

        failed_episodes = [
            {**e["metadata"], "attack_turn": e["content"]}
            for eid in self._failure_buffer
            if (e := self.l1.get(eid)) is not None
        ]
        if not failed_episodes:
            self._failure_buffer = []
            self._save_failure_buffer()
            return

        for ep in failed_episodes:
            ep["injected_policies"] = [
                e["metadata"] for pid in (ep.get("retrieved_policy_ids") or [])
                if (e := self.l2.get(pid)) is not None
            ]

        existing_policies = [e["metadata"] for e in self.l2.get_all()]
        proposed = self.policy_synthesizer.synthesize_policies(
            failed_episodes, existing_policies=existing_policies or None
        )

        if proposed:
            if self.flat:
                logger.info(f"Synthesis (flat): writing {len(proposed)} policy(ies) without the manager")
                self._write_flat(proposed)
            else:
                logger.info(f"Synthesis: {len(proposed)} candidate policy(ies) -> placing into L2 tree")
                for p in proposed:
                    logger.info(f"  candidate: {p.get('name', '?')}")
                ops = organize(self.l2, proposed, self.policy_synthesizer.llm,
                               source_episode_ids=list(self._failure_buffer))
                for line in ops:
                    logger.info(f"    {line}")
                self._log_organize([p.get("name", "?") for p in proposed], ops)

        self._failure_buffer = []
        self._save_failure_buffer()

    def _log_organize(self, candidates: List[str], ops: List[str]) -> None:
        """Append this batch's placement decisions next to the store.

        place() already returns one line per candidate (ADD/MERGE/SPLIT/PROMOTE/GROUP),
        but those only reach logger.info, which the DTap task runner does not propagate —
        so after a run there is no record of why a duplicate landed at root.
        """
        path = self.l2.jsonl_path.parent / "organize_log.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "episode_ids": list(self._failure_buffer),
                    "candidates":  candidates,
                    "ops":         ops,
                    "l2_size":     len(self.l2),
                }, ensure_ascii=False) + "\n")
        except Exception as exc:                      # never let logging kill synthesis
            logger.warning(f"organize log write failed: {exc}")

    def _write_flat(self, proposed: List[dict]) -> None:
        """Ablation: write synthesized policies as a flat store (name-match replace, else add). No tree."""
        by_name = {e["metadata"].get("policy_name"): e["id"] for e in self.l2.get_all()}
        for p in proposed:
            meta = {
                "policy_name": p.get("name", ""), "description": p.get("description", ""),
                "scope": p.get("scope", ""), "detection": [str(x) for x in p.get("detection", [])],
                "response": [str(x) for x in p.get("response", [])], "harm": p.get("harm", ""),
                "parent": "root", "retrievable": True,
            }
            name = meta["policy_name"]
            if name in by_name:
                entry = self.l2.get(by_name[name])
                entry["metadata"].update(meta)
                self.l2.update(entry["id"], "")
            else:
                by_name[name] = self.l2.add("", metadata=meta)
            logger.info(f"  flat write: {name}")

    # --- Size queries (used by SelfPlayRunner for logging) ---

    def l1_size(self) -> int:
        self._require_attack_memory()
        return len(self.l1)

    def l2_size(self) -> int:
        self._require_attack_memory()
        return len(self.l2)
