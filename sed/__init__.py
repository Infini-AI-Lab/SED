"""Self-Evolving Defense (SED).

Module layout follows the components in the paper (§4.3):

    judge.py       Judge J          — scores a completed trajectory
    synthesizer.py Synthesizer S    — distils a harmful episode into policies
    organizer.py   Organizer O      — places a policy in the tree (Place / Generalize)
    memory.py      MemoryManager    — episodic L1 + policy L2, and retrieval rho
    store.py       MemoryStore      — JSONL + numpy cosine store
    tree.py        Tree             — the policy tree itself
    agent.py       DefenseAgent     — the frozen agent, with policies injected
"""

from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent
from sed.judge import Judge, JudgeResult
from sed.memory import MemoryManager
from sed.store import MemoryStore
from sed.synthesizer import PolicySynthesizer, format_policies_for_injection

__all__ = [
    "DefenseAgent",
    "DEFENSE_BASE_PROMPT",
    "Judge",
    "JudgeResult",
    "MemoryManager",
    "MemoryStore",
    "PolicySynthesizer",
    "format_policies_for_injection",
]
