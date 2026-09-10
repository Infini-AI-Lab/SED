"""Capture the latest assistant message for GuardAgent/DRIFT per-tool hooks."""

from __future__ import annotations

from typing import Any

_patched = False


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def assistant_text_from_turn_input(turn_input: list[Any] | None) -> str:
    """Best-effort extraction from OpenAI Agents SDK turn_input items."""
    for item in reversed(turn_input or []):
        if isinstance(item, dict):
            role = item.get("role")
            if role == "assistant":
                text = _content_to_text(item.get("content"))
                if text.strip():
                    return text
            if item.get("type") in ("message", "output_message") and item.get("role") == "assistant":
                text = _content_to_text(item.get("content"))
                if text.strip():
                    return text
            continue

        role = getattr(item, "role", None) or getattr(item, "type", None)
        if role in ("assistant", "message", "output_message"):
            text = _content_to_text(getattr(item, "content", None))
            if text.strip():
                return text
    return ""


def apply_assistant_context_patch() -> None:
    """Monkey-patch agents.mcp.util so tool calls see the latest assistant text."""
    global _patched
    if _patched:
        return
    try:
        import agents.mcp.util as mcp_util
    except ImportError:
        return

    if getattr(mcp_util, "_dtap_assistant_patch", False):
        _patched = True
        return

    orig = mcp_util.MCPToolResolver.invoke_mcp_tool

    async def _patched_invoke(
        server,
        tool,
        context,
        input_json,
        *,
        meta=None,
        tool_display_name=None,
    ):
        try:
            from dt_arena.src.hooks.task_context import set_task_context

            txt = assistant_text_from_turn_input(getattr(context, "turn_input", None))
            if txt.strip():
                set_task_context(assistant_text=txt)
        except Exception:
            pass
        return await orig(
            server,
            tool,
            context,
            input_json,
            meta=meta,
            tool_display_name=tool_display_name,
        )

    mcp_util.MCPToolResolver.invoke_mcp_tool = _patched_invoke
    mcp_util._dtap_assistant_patch = True
    _patched = True
