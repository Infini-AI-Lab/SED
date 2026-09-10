"""Guard prompt helpers (blind vs oracle malicious_goal)."""
from __future__ import annotations

import os
from typing import Any, Mapping


def blind_guards_enabled() -> bool:
    return os.environ.get("DTAP_BLIND_GUARDS", "0") == "1"


def guard_malicious_goal(tctx: Mapping[str, Any] | None = None, raw: str | None = None) -> str:
    """Goal text for tool-level guard prompts. Empty when DTAP_BLIND_GUARDS=1."""
    if blind_guards_enabled():
        return ""
    if raw is not None:
        return raw or ""
    if tctx is None:
        return ""
    return str(tctx.get("malicious_goal") or "")


def guard_instruction(tctx: Mapping[str, Any], *, allow_goal_fallback: bool = False) -> str:
    """User instruction for guards. Optionally fall back to goal (GuardAgent legacy)."""
    instr = str(tctx.get("instruction") or "")
    if instr:
        return instr
    if allow_goal_fallback and not blind_guards_enabled():
        return str(tctx.get("malicious_goal") or "")
    return instr
