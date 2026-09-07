"""Query chat run status without exposing checkpoint file I/O to the UI.

Resolution order:
1. Messages DB (assistant already persisted)
2. In-process run registry (same process, refresh mid-run)
3. LangGraph checkpoint (durable/working DB via SqliteSaver) — hydrate messages

Checkpoint hydrate reconstructs tool_events from AIMessage/ToolMessage so a
browser refresh does not drop the Tools / Tool result timeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from application import chat
from application import run_registry
from application import task_store
from application.task_store_persistence import flush_persist

logger = logging.getLogger("run_state")

_TOOL_RESULT_MAX_CHARS = 12000


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    return chat._content_to_text(content)


def _truncate_tool_result(text: str, limit: int = _TOOL_RESULT_MAX_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…(truncated)"


def _ai_has_tool_calls(msg: AIMessage) -> bool:
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        return True
    additional = getattr(msg, "additional_kwargs", None) or {}
    return bool(additional.get("tool_calls"))


def _tool_calls_list(msg: AIMessage) -> list[dict[str, Any]]:
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        return [tc for tc in tool_calls if isinstance(tc, dict)]
    additional = getattr(msg, "additional_kwargs", None) or {}
    raw = additional.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        if "function" in tc:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            out.append(
                {
                    "id": tc.get("id") or "",
                    "name": fn.get("name") or "unknown",
                    "args": args if isinstance(args, dict) else {"value": args},
                }
            )
        else:
            out.append(tc)
    return out


def _tool_events_from_checkpoint_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Rebuild UI tool timeline for the latest human turn from checkpoint messages."""
    if not messages:
        return []

    start = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            start = i + 1

    events: list[dict[str, Any]] = []
    for msg in messages[start:]:
        if isinstance(msg, AIMessage):
            text = _message_text(msg).strip()
            tool_calls = _tool_calls_list(msg)
            if text and tool_calls:
                events.append({"type": "text", "data": text})
            # Final AI text (no tool_calls) lives in message.content — skip duplicate.
            for tc in tool_calls:
                tid = str(tc.get("id") or "")
                name = str(tc.get("name") or "unknown")
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                event: dict[str, Any] = {
                    "type": "tool",
                    "tool": name,
                    "input": args or {},
                    "toolUseId": tid,
                }
                if name == "get_skill_instructions" and isinstance(args, dict):
                    skill = (
                        args.get("skill_name")
                        or args.get("skill")
                        or args.get("name")
                    )
                    if skill:
                        event["skillName"] = str(skill)
                events.append(event)
        elif isinstance(msg, ToolMessage):
            tid = str(getattr(msg, "tool_call_id", None) or "")
            name = str(getattr(msg, "name", None) or "")
            if not name:
                for ev in reversed(events):
                    if ev.get("type") == "tool" and ev.get("toolUseId") == tid:
                        name = str(ev.get("tool") or "")
                        break
            result_event: dict[str, Any] = {
                "type": "tool_result",
                "tool": name or "unknown",
                "toolUseId": tid,
                "data": _truncate_tool_result(_message_text(msg)),
            }
            for ev in reversed(events):
                if ev.get("type") == "tool" and ev.get("toolUseId") == tid:
                    if ev.get("skillName"):
                        result_event["skillName"] = ev["skillName"]
                    if ev.get("mcpServer"):
                        result_event["mcpServer"] = ev["mcpServer"]
                    break
            events.append(result_event)
    return events


def _classify_checkpoint_messages(messages: list[Any]) -> dict[str, Any]:
    """Derive run status from LangGraph channel messages."""
    if not messages:
        return {
            "status": "pending",
            "content": "",
            "tool_events": [],
            "incomplete": True,
        }

    last = messages[-1]
    tool_events = _tool_events_from_checkpoint_messages(messages)
    # Walk backwards for the latest final AI reply (no pending tool calls).
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not _ai_has_tool_calls(msg):
            text = _message_text(msg).strip()
            if text:
                return {
                    "status": "done",
                    "content": text,
                    "tool_events": tool_events,
                    "incomplete": False,
                }

    if isinstance(last, ToolMessage) or (
        isinstance(last, AIMessage) and _ai_has_tool_calls(last)
    ):
        return {
            "status": "pending",
            "content": "",
            "tool_events": tool_events,
            "incomplete": True,
        }

    if isinstance(last, HumanMessage):
        return {
            "status": "pending",
            "content": "",
            "tool_events": [],
            "incomplete": True,
        }

    return {
        "status": "pending",
        "content": "",
        "tool_events": tool_events,
        "incomplete": True,
    }


def _read_checkpoint_messages(session_id: str) -> list[Any] | None:
    """Load channel_values.messages for thread_id=session_id via SqliteSaver."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as exc:
        logger.warning("SqliteSaver unavailable: %s", exc)
        return None

    candidates: list[str] = []
    working = chat.get_checkpoint_db_path(session_id)
    durable = chat.get_persistent_checkpoint_db_path(session_id)
    # Prefer the newer non-empty DB.
    for path in (working, durable):
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            candidates.append(path)
    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    config = {"configurable": {"thread_id": session_id}}

    for path in candidates:
        try:
            with SqliteSaver.from_conn_string(path) as saver:
                tup = saver.get_tuple(config)
            if not tup or not tup.checkpoint:
                continue
            channel_values = tup.checkpoint.get("channel_values") or {}
            messages = channel_values.get("messages") or []
            if messages:
                return list(messages)
        except Exception as exc:
            logger.warning(
                "Checkpoint read failed for session=%s path=%s: %s",
                session_id,
                path,
                exc,
            )
            continue
    return None


def _resolve_tool_events_for_hydrate(
    *,
    session_id: str,
    registry_events: list[dict[str, Any]] | None = None,
    checkpoint_messages: list[Any] | None = None,
) -> list[dict[str, Any]]:
    if registry_events:
        return list(registry_events)
    msgs = checkpoint_messages
    if msgs is None:
        msgs = _read_checkpoint_messages(session_id)
    if msgs:
        return _tool_events_from_checkpoint_messages(msgs)
    return []


def _hydrate_assistant_if_needed(
    *,
    task_id: str,
    user_id: str,
    content: str,
    images: list[str] | None = None,
    tool_events: list[dict[str, Any]] | None = None,
) -> bool:
    """Persist assistant message when messages DB still ends on user.

    If an assistant row already exists without tool_events (race with an older
    text-only hydrate), backfill tool_events when available.
    """
    content = (content or "").strip()
    events = list(tool_events or [])
    messages = task_store.list_messages(task_id, user_id)
    if messages and messages[-1].get("role") == "assistant":
        last = messages[-1]
        existing_events = last.get("tool_events") or []
        if events and not existing_events:
            updated = task_store.update_message_tool_events(
                last["id"],
                task_id,
                user_id,
                events,
            )
            if updated:
                flush_persist(user_id)
                logger.info(
                    "Backfilled tool_events on hydrated assistant (%s events) task=%s",
                    len(events),
                    task_id,
                )
                return True
        return False
    if not content and not events:
        return False
    task_store.add_message(
        task_id,
        "assistant",
        content,
        user_id=user_id,
        images=images or [],
        tool_events=events,
    )
    flush_persist(user_id)
    logger.info(
        "Hydrated assistant message from run query (%s chars, %s events) task=%s",
        len(content),
        len(events),
        task_id,
    )
    return True


def query_task_run(task_id: str, user_id: str) -> dict[str, Any]:
    """Return run status for a task (query API surface for UI / future runtime)."""
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        return {
            "task_id": task_id,
            "status": "missing",
            "content": "",
            "images": [],
            "error": "Task not found",
            "source": None,
            "hydrated": False,
        }

    session_id = task.get("runtime_session_id") or task_id
    messages = task_store.list_messages(task_id, user_id)
    last_role = messages[-1]["role"] if messages else None

    # 1) Messages already complete → idle (still allow tool_events backfill)
    if last_role == "assistant":
        last = messages[-1]
        existing_events = last.get("tool_events") or []
        hydrated = False
        if not existing_events:
            tool_events = _resolve_tool_events_for_hydrate(session_id=session_id)
            if tool_events:
                hydrated = _hydrate_assistant_if_needed(
                    task_id=task_id,
                    user_id=user_id,
                    content=last.get("content") or "",
                    images=last.get("images") or [],
                    tool_events=tool_events,
                )
                if hydrated:
                    last = task_store.list_messages(task_id, user_id)[-1]
        return {
            "task_id": task_id,
            "status": "idle",
            "content": last.get("content") or "",
            "images": last.get("images") or [],
            "error": None,
            "source": "messages",
            "hydrated": hydrated,
        }

    # 2) In-process registry (browser refresh while same process still runs)
    reg = run_registry.get(task_id)
    if reg and reg.get("status") == "running":
        return {
            "task_id": task_id,
            "status": "running",
            "content": "",
            "images": [],
            "error": None,
            "source": "registry",
            "hydrated": False,
        }

    if reg and reg.get("status") in ("done", "error"):
        content = (reg.get("content") or "").strip()
        error = reg.get("error")
        images = list(reg.get("images") or [])
        if error and not content:
            content = f"Error: {error}"
        tool_events = _resolve_tool_events_for_hydrate(
            session_id=session_id,
            registry_events=reg.get("tool_events") or [],
        )
        hydrated = False
        if (content or tool_events) and last_role == "user":
            hydrated = _hydrate_assistant_if_needed(
                task_id=task_id,
                user_id=user_id,
                content=content,
                images=images,
                tool_events=tool_events,
            )
        return {
            "task_id": task_id,
            "status": "error" if error and not (reg.get("content") or "").strip() else "done",
            "content": content,
            "images": images,
            "error": error,
            "source": "registry",
            "hydrated": hydrated,
        }

    # Empty conversation
    if not messages:
        return {
            "task_id": task_id,
            "status": "idle",
            "content": "",
            "images": [],
            "error": None,
            "source": "messages",
            "hydrated": False,
        }

    # 3) Checkpoint query (last message is user; recover finished answer)
    cp_messages = _read_checkpoint_messages(session_id)
    if cp_messages is not None:
        classified = _classify_checkpoint_messages(cp_messages)
        if classified["status"] == "done" and classified.get("content"):
            tool_events = classified.get("tool_events") or []
            hydrated = _hydrate_assistant_if_needed(
                task_id=task_id,
                user_id=user_id,
                content=classified["content"],
                tool_events=tool_events,
            )
            return {
                "task_id": task_id,
                "status": "done",
                "content": classified["content"],
                "images": [],
                "error": None,
                "source": "checkpoint",
                "hydrated": hydrated,
            }
        return {
            "task_id": task_id,
            "status": "pending",
            "content": "",
            "images": [],
            "error": None,
            "source": "checkpoint",
            "hydrated": False,
        }

    # No registry, no checkpoint — waiting / abandoned
    return {
        "task_id": task_id,
        "status": "pending",
        "content": "",
        "images": [],
        "error": None,
        "source": None,
        "hydrated": False,
    }
