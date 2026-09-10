#!/usr/bin/env python3
"""Install SED hooks + defense wiring into a DecodingTrust-Agent checkout."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SED_EXP = Path(__file__).resolve().parent
DTAP = Path(sys.argv[1]) if len(sys.argv) > 1 else SED_EXP.parents[1] / "third_party" / "DecodingTrust-Agent"

HOOKS_DIR = DTAP / "dt_arena" / "src" / "hooks"
TASK_RUNNER = DTAP / "dt_arena" / "eval" / "task_runner.py"
if not TASK_RUNNER.is_file():
    TASK_RUNNER = DTAP / "eval" / "task_runner.py"
MCP_WRAPPER = DTAP / "agent" / "openaisdk" / "src" / "mcp_wrapper.py"

# All hooks always registered; each hook self-gates on DTAP_DEFENSE / SED_ENABLE.
ALL_HOOKS = [
    "dt_arena.src.hooks.sed_guard:SEDGuardHook",
    "dt_arena.src.hooks.llama_guard_hook:LlamaGuardHook",
    "dt_arena.src.hooks.safeharbor_hook:SafeHarborHook",
    "dt_arena.src.hooks.guardagent_hook:GuardAgentHook",
    "dt_arena.src.hooks.drift_hook:DriftHook",
]

TASK_RUNNER_OLD = """                try:
                    result = await agent.run(turn_instruction, metadata=metadata)"""

TASK_RUNNER_NEW = """                try:
                    from dt_arena.src.hooks.task_context import set_task_context
                    set_task_context(
                        task_id=task_cfg.task_id,
                        instruction=turn_instruction,
                        malicious_goal=(attack_cfg.malicious_goal if attack_cfg else None),
                        domain=task_cfg.domain,
                        task_split=("benign" if attack_cfg is None else "malicious"),
                    )
                    try:
                        from dt_arena.src.hooks.assistant_context import apply_assistant_context_patch
                        apply_assistant_context_patch()
                    except Exception:
                        pass
                    try:
                        from dt_arena.src.hooks.defense_turninject import inject_turn_defenses
                        await inject_turn_defenses(
                            agent,
                            turn_instruction=turn_instruction,
                            malicious_goal=(attack_cfg.malicious_goal if attack_cfg else None),
                            task_id=task_cfg.task_id,
                        )
                    except Exception as _def_err:
                        print(f"[DTAP defense] turn inject skipped: {_def_err}")
                    result = await agent.run(turn_instruction, metadata=metadata)
                    if os.environ.get("SED_ENABLE", "0") == "1":
                        try:
                            from dt_arena.src.hooks.sed_guard import SEDGuardHook, transcript_from_run_result
                            _sed_j = SEDGuardHook()
                            await _sed_j._ensure()
                            _rr = getattr(agent, "_last_result", None)
                            _final = getattr(result, "final_output", "") or ""
                            _conv = transcript_from_run_result(_rr, turn_instruction, _final)
                            await asyncio.to_thread(
                                _sed_j.judge_and_store, f"dtap-{task_cfg.task_id}", _conv
                            )
                        except Exception as _sed_j_err:
                            print(f"[SED] judge-once skipped: {_sed_j_err}")"""

MCP_OLD = """        ctx = ToolCallContext(
            framework="openaisdk",
            server=self.name or "unknown",
            tool_name=tool_name,
            arguments=dict(arguments) if arguments else {},
        )"""

MCP_NEW = """        from dt_arena.src.hooks.task_context import get_task_context
        ctx = ToolCallContext(
            framework="openaisdk",
            server=self.name or "unknown",
            tool_name=tool_name,
            arguments=dict(arguments) if arguments else {},
            metadata=get_task_context(),
        )"""


def write_hooks_json(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"hooks": ALL_HOOKS}, indent=2) + "\n", encoding="utf-8")
    print(f"hooks.json -> {dest} ({len(ALL_HOOKS)} hooks)")


def _patch_file(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new.strip() in text:
        print(f"  {label}: already patched")
        return
    if old not in text:
        raise SystemExit(f"{label}: anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  {label}: patched")


def _upgrade_task_runner(tr: str) -> str:
    """Upgrade legacy patches to unified inject_turn_defenses + judge-once."""
    broken = (
        "                try:\n"
        "                    try:\n"
        "                    from dt_arena.src.hooks.task_context import set_task_context"
    )
    if broken in tr:
        tr = tr.replace(
            broken,
            "                try:\n"
            "                    from dt_arena.src.hooks.task_context import set_task_context",
            1,
        )

    if "inject_turn_defenses" in tr and "judge_and_store" in tr:
        if "apply_assistant_context_patch" not in tr:
            anchor = (
                "                    try:\n"
                "                        from dt_arena.src.hooks.defense_turninject import inject_turn_defenses"
            )
            if anchor in tr:
                tr = tr.replace(
                    anchor,
                    "                    try:\n"
                    "                        from dt_arena.src.hooks.assistant_context import apply_assistant_context_patch\n"
                    "                        apply_assistant_context_patch()\n"
                    "                    except Exception:\n"
                    "                        pass\n"
                    + anchor,
                    1,
                )
        return tr

    # Jul 7 patch without Minh judge-once: append judge block after agent.run
    if "inject_turn_defenses" in tr and "judge_and_store" not in tr:
        anchor = "                    result = await agent.run(turn_instruction, metadata=metadata)"
        judge_tail = """
                    if os.environ.get("SED_ENABLE", "0") == "1":
                        try:
                            from dt_arena.src.hooks.sed_guard import SEDGuardHook, transcript_from_run_result
                            _sed_j = SEDGuardHook()
                            await _sed_j._ensure()
                            _rr = getattr(agent, "_last_result", None)
                            _final = getattr(result, "final_output", "") or ""
                            _conv = transcript_from_run_result(_rr, turn_instruction, _final)
                            await asyncio.to_thread(
                                _sed_j.judge_and_store, f"dtap-{task_cfg.task_id}", _conv
                            )
                        except Exception as _sed_j_err:
                            print(f"[SED] judge-once skipped: {_sed_j_err}")"""
        if anchor in tr:
            return tr.replace(anchor, anchor + judge_tail, 1)

    if TASK_RUNNER_OLD in tr:
        return tr.replace(TASK_RUNNER_OLD, TASK_RUNNER_NEW, 1)

    # Minh-style partial patch without inject_turn_defenses: replace whole block
    if "set_task_context" in tr and "inject_turn_defenses" not in tr:
        start = tr.find("from dt_arena.src.hooks.task_context import set_task_context")
        if start != -1:
            end_anchor = 'print(f"[SED] judge-once skipped: {_sed_j_err}")'
            end = tr.find(end_anchor)
            if end != -1:
                end = tr.find("\n", end) + 1
                before = tr[: tr.rfind("                try:", 0, start)]
                after_idx = tr.find("                except", end)
                if after_idx == -1:
                    after_idx = end
                return before + TASK_RUNNER_NEW.strip() + "\n" + tr[after_idx:]

    return tr


def install(defense: str = "sed") -> None:
    del defense  # all hooks always installed; defense selected via env at runtime
    if not DTAP.is_dir():
        raise SystemExit(f"DTAP dir not found: {DTAP}")

    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    hook_files = (
        "sed_guard.py",
        "llama_guard_hook.py",
        "safeharbor_hook.py",
        "guardagent_hook.py",
        "drift_hook.py",
        "task_context.py",
        "assistant_context.py",
        "defense_turninject.py",
        "guard_context.py",
    )
    for name in hook_files:
        shutil.copy2(SED_EXP / name, HOOKS_DIR / name)
    print(f"copied hooks -> {HOOKS_DIR}")

    hooks_dest = Path(os.environ.get("DTAP_HOOKS_JSON", HOOKS_DIR / "hooks.json"))
    write_hooks_json(hooks_dest)

    tr = TASK_RUNNER.read_text(encoding="utf-8")
    upgraded = _upgrade_task_runner(tr)
    if upgraded != tr:
        TASK_RUNNER.write_text(upgraded, encoding="utf-8")
        print("  task_runner: patched")
    elif TASK_RUNNER_NEW.strip() in tr:
        print("  task_runner: already patched")
    else:
        raise SystemExit(
            "task_runner patch failed — restore eval/task_runner.py and re-run setup"
        )

    _patch_file(MCP_WRAPPER, MCP_OLD, MCP_NEW, "mcp_wrapper")

    import subprocess

    subprocess.run(
        [sys.executable, str(SED_EXP / "patch_hooks_loader.py"), str(DTAP)],
        check=False,
    )


def write_hooks_json_only(dest: Path | None = None) -> None:
    dest = dest or Path(os.environ.get("DTAP_HOOKS_JSON", HOOKS_DIR / "hooks.json"))
    write_hooks_json(dest)


_VALID_DEFENSES = ("baseline", "sed", "llama_guard", "safeharbor", "guardagent", "drift")


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "install"
    if mode == "hooks-only":
        write_hooks_json_only()
    elif mode in ("baseline", *_VALID_DEFENSES):
        install(defense=mode)
    else:
        install()
