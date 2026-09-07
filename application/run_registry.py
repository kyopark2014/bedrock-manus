"""In-process registry of active / recently finished chat runs.

Used so browser refresh can discover that an agent is still working (or just
finished) without reading LangGraph checkpoint files directly.
"""

from __future__ import annotations

import threading
import time
from typing import Any

# Keep finished entries briefly so a refresh right after completion can see them
# before messages DB / late-persist catches up.
_DONE_TTL_SECONDS = 1800

_lock = threading.Lock()
# task_id -> run record
_runs: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def mark_running(task_id: str, user_id: str) -> None:
    if not task_id:
        return
    with _lock:
        _runs[task_id] = {
            "task_id": task_id,
            "user_id": user_id,
            "status": "running",
            "content": "",
            "images": [],
            "error": None,
            "started_at": _now(),
            "finished_at": None,
        }


def mark_done(
    task_id: str,
    *,
    content: str = "",
    images: list[str] | None = None,
    error: str | None = None,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    if not task_id:
        return
    with _lock:
        prev = _runs.get(task_id) or {
            "task_id": task_id,
            "user_id": "",
            "started_at": _now(),
        }
        update: dict[str, Any] = {
            "status": "error" if error else "done",
            "content": content or "",
            "images": list(images or []),
            "error": error,
            "finished_at": _now(),
        }
        if tool_events is not None:
            update["tool_events"] = list(tool_events)
        elif "tool_events" not in prev:
            update["tool_events"] = []
        prev.update(update)
        _runs[task_id] = prev


def clear(task_id: str) -> None:
    with _lock:
        _runs.pop(task_id, None)


def get(task_id: str) -> dict[str, Any] | None:
    """Return a shallow copy of the run record, or None if missing/expired."""
    with _lock:
        rec = _runs.get(task_id)
        if not rec:
            return None
        if rec.get("status") in ("done", "error"):
            finished = rec.get("finished_at") or 0
            if _now() - float(finished) > _DONE_TTL_SECONDS:
                _runs.pop(task_id, None)
                return None
        return dict(rec)


def is_running(task_id: str) -> bool:
    rec = get(task_id)
    return bool(rec and rec.get("status") == "running")
