"""Per-task context for DTap hooks (async-safe across parallel tasks)."""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict

_task_ctx: ContextVar[Dict[str, Any]] = ContextVar("dtap_task_ctx", default={})


def set_task_context(**fields: Any) -> None:
    cur = dict(_task_ctx.get())
    cur.update(fields)
    _task_ctx.set(cur)


def get_task_context() -> Dict[str, Any]:
    return dict(_task_ctx.get())


def clear_task_context() -> None:
    _task_ctx.set({})
