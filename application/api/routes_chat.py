import json
import logging
import queue
import re
import threading
import time
from typing import Any, Generator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from application.api.routes_auth import _kick_graph_job, require_user_id
from application import task_store
from application.task_store_persistence import flush_persist
from application import chat
from application import run_registry
from application import run_cancel
from application.notification_queue import QueueNotificationSink
from application.runtime_mode import run_agent

logger = logging.getLogger("routes_chat")

router = APIRouter(prefix="/api/tasks", tags=["chat"])

SSE_HEARTBEAT_INTERVAL_SECONDS = 15
# Idle timeout: no agent queue output for this long → treat as stuck.
# (Previously this was a wall-clock limit from stream start, so long Manus
# runs hit "Agent timeout" even while tools were still progressing.)
AGENT_STREAM_TIMEOUT_SECONDS = 1200
# Absolute safety cap for a single chat turn (4 hours).
AGENT_STREAM_MAX_SECONDS = 14400
# After SSE times out / client disconnects, wait this long for the agent
# worker to finish so the final answer can still be persisted for refresh.
LATE_PERSIST_WAIT_SECONDS = 1800
DEFAULT_IMAGE_PROMPT = "첨부한 이미지를 분석해주세요."

_TOOL_INPUT_RE = re.compile(r"^Tool: (.+?), Input:\s*(.*)$", re.DOTALL)
_TOOL_RESULT_RE = re.compile(r"^Tool Result: (.+)$", re.DOTALL)


def _parse_tool_input(raw_input: str) -> Any:
    stripped = raw_input.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw_input


class ChatRequest(BaseModel):
    prompt: str = ""
    files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_prompt_or_files(self):
        if not self.prompt.strip() and not self.files:
            raise ValueError("prompt or files is required")
        return self


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_keepalive() -> str:
    return ": keepalive\n\n"


def _is_segment_reset(previous: str, new: str) -> bool:
    if not previous.strip():
        return False
    if not new:
        return True
    return not new.startswith(previous)


def _handle_token(
    tool_events: list[dict[str, Any]],
    streamed_text: str,
    new_text: str,
) -> tuple[str, dict[str, Any] | None]:
    if new_text == streamed_text:
        return streamed_text, None
    committed = None
    if _is_segment_reset(streamed_text, new_text):
        committed = _flush_text_segment(tool_events, streamed_text)
    return new_text, committed


def _flush_text_segment(timeline: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if (
        timeline
        and timeline[-1].get("type") == "text"
        and timeline[-1].get("data", "").strip() == stripped
    ):
        return None
    timeline.append({"type": "text", "data": stripped})
    return {"type": "text", "data": stripped}


def _is_streaming_prefix_of_final(partial: str, final: str) -> bool:
    """True when partial looks like a streaming artifact of the same answer."""
    if not partial or not final:
        return False
    if final.startswith(partial) or partial.startswith(final):
        return True
    head_len = min(len(partial), len(final), 80)
    return partial[:head_len] == final[:head_len]


def _set_final_text_in_timeline(
    timeline: list[dict[str, Any]], final_content: str
) -> None:
    stripped = final_content.strip()
    if not stripped:
        return
    if timeline and timeline[-1].get("type") == "text":
        last = timeline[-1].get("data", "").strip()
        if last == stripped:
            return
        if _is_streaming_prefix_of_final(last, stripped):
            timeline[-1] = {"type": "text", "data": stripped}
            return
    timeline.append({"type": "text", "data": stripped})


def _upsert_tool_event(tool_events: list[dict[str, Any]], mapped: dict[str, Any]) -> None:
    if mapped["type"] == "info":
        data = str(mapped.get("data", ""))
        if _TOOL_INPUT_RE.match(data) or _TOOL_RESULT_RE.match(data):
            return
        tool_events.append(mapped)
        return

    # Legacy status payloads → same as info notifications
    if mapped["type"] == "status":
        tool_events.append({"type": "info", "data": mapped.get("data", "")})
        return

    if mapped["type"] == "plan":
        tool_events.append(mapped)
        return

    if mapped["type"] in ("tool", "tool_result"):
        tool_use_id = mapped.get("toolUseId")
        for i, existing in enumerate(tool_events):
            if existing.get("type") == mapped["type"] and existing.get("toolUseId") == tool_use_id:
                tool_events[i] = mapped
                return
        if mapped["type"] == "tool":
            tool_name = mapped.get("tool")
            if tool_name:
                for i in range(len(tool_events) - 1, -1, -1):
                    existing = tool_events[i]
                    if existing.get("type") == "tool" and existing.get("tool") == tool_name:
                        existing_id = existing.get("toolUseId")
                        # Same call (or placeholder id=tool_name): update in place.
                        # Distinct call ids: fall through and append for chronological order.
                        if (
                            not tool_use_id
                            or not existing_id
                            or existing_id == tool_use_id
                            or existing_id == tool_name
                        ):
                            if tool_use_id and tool_use_id != tool_name:
                                tool_events[i] = mapped
                            else:
                                tool_events[i] = {**existing, **mapped}
                            return
                        break
    tool_events.append(mapped)


def _track_tool_event(
    tool_events: list[dict[str, Any]],
    tool_meta: dict[str, dict[str, Any]],
    mapped: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist tool events and backfill tool call when only result arrives."""
    events_to_emit = [mapped]
    tool_use_id = mapped.get("toolUseId")

    if mapped["type"] == "tool":
        tool_meta[tool_use_id] = {
            "tool": mapped.get("tool"),
            "input": mapped.get("input", {}),
        }
        _upsert_tool_event(tool_events, mapped)
        return events_to_emit

    if mapped["type"] == "tool_result" and tool_use_id:
        meta = tool_meta.get(tool_use_id, {})
        tool_name = meta.get("tool") or mapped.get("tool")
        has_tool_event = any(
            event.get("type") == "tool" and event.get("toolUseId") == tool_use_id
            for event in tool_events
        )
        if not has_tool_event:
            tool_event = {
                "type": "tool",
                "tool": tool_name or "unknown",
                "input": meta.get("input", {}),
                "toolUseId": tool_use_id,
            }
            _upsert_tool_event(tool_events, tool_event)
            events_to_emit.insert(0, tool_event)
            if tool_name:
                tool_meta[tool_use_id] = {
                    "tool": tool_name,
                    "input": meta.get("input", {}),
                }
        if tool_name:
            mapped = {**mapped, "tool": tool_name}
            events_to_emit[-1] = mapped
        _upsert_tool_event(tool_events, mapped)
        return events_to_emit

    if mapped["type"] in ("tool", "tool_result", "info", "plan", "status"):
        _upsert_tool_event(tool_events, mapped)
    return events_to_emit


def _normalize_tool_use_id(tool_use_id: str) -> str:
    if tool_use_id.endswith(":input"):
        return tool_use_id[: -len(":input")]
    if tool_use_id.endswith(":result"):
        return tool_use_id[: -len(":result")]
    return tool_use_id


def _map_sink_event(event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = event.get("type")
    data = event.get("data", "")

    if event_type == "markdown":
        return {"type": "token", "data": data}

    if event_type == "text_segment":
        return {"type": "text", "data": data}

    if event_type == "plan":
        return {"type": "plan", "data": data}

    if event_type == "status":
        # Legacy: treat dedicated status events as Info notifications.
        data = str(data)
        if data.startswith("[status]"):
            data = data[len("[status]") :].lstrip("\n").strip()
        return {"type": "info", "data": data}

    if event_type == "info":
        raw = str(data)
        if raw.startswith("[status]"):
            return {
                "type": "info",
                "data": raw[len("[status]") :].lstrip("\n").strip(),
            }
        tool_match = _TOOL_INPUT_RE.match(raw)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_input = _parse_tool_input(tool_match.group(2))
            return {
                "type": "tool",
                "tool": tool_name,
                "input": tool_input,
                "toolUseId": _normalize_tool_use_id(event.get("toolUseId", tool_name)),
            }
        result_match = _TOOL_RESULT_RE.match(raw)
        if result_match:
            tool_name = event.get("toolName") or ""
            mapped_result = {
                "type": "tool_result",
                "toolUseId": _normalize_tool_use_id(event.get("toolUseId", "")),
                "data": result_match.group(1),
            }
            if tool_name:
                mapped_result["tool"] = tool_name
            return mapped_result
        return {"type": "info", "data": raw}

    return event



def _consume_queue_item(
    item: dict[str, Any],
    *,
    tool_events: list[dict[str, Any]],
    tool_meta: dict[str, dict[str, Any]],
    streamed_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply one notification-queue item to the timeline. Returns (text, emit)."""
    mapped = _map_sink_event(item)
    if not mapped:
        return streamed_text, []

    if mapped["type"] == "token":
        before = streamed_text
        streamed_text, committed = _handle_token(
            tool_events, streamed_text, mapped["data"]
        )
        events: list[dict[str, Any]] = []
        if streamed_text == before and committed is None:
            return streamed_text, events
        if committed:
            events.append(committed)
        events.append(mapped)
        return streamed_text, events

    if mapped["type"] == "text":
        committed = _flush_text_segment(tool_events, mapped["data"])
        events = [committed] if committed else []
        return "", events

    if mapped["type"] in ("tool", "tool_result", "info", "plan", "status"):
        events = []
        if mapped["type"] in ("tool", "tool_result", "plan", "status"):
            committed = _flush_text_segment(tool_events, streamed_text)
            if mapped["type"] == "tool":
                streamed_text = ""
            if committed:
                events.append(committed)
        events.extend(_track_tool_event(tool_events, tool_meta, mapped))
        return streamed_text, events

    return streamed_text, [mapped]


def _build_final_payload(
    *,
    result_holder: dict[str, Any],
    tool_events: list[dict[str, Any]],
    streamed_text: str,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    authoritative_final = (result_holder.get("content") or "").strip()
    final_content = authoritative_final or streamed_text
    if authoritative_final:
        _set_final_text_in_timeline(tool_events, final_content)
    else:
        _flush_text_segment(tool_events, streamed_text)
        _set_final_text_in_timeline(tool_events, final_content)
    images = result_holder.get("images") or []
    return final_content, images, tool_events


def _build_partial_error_payload(
    *,
    tool_events: list[dict[str, Any]],
    streamed_text: str,
    error_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Keep streamed progress on timeout/error; append an error notice."""
    events = list(tool_events)
    partial = (streamed_text or "").strip()
    if partial:
        _flush_text_segment(events, partial)
    notice = error_text if error_text.startswith("Error:") else f"Error: {error_text}"
    text_parts = [
        str(e.get("data") or "").strip()
        for e in events
        if e.get("type") == "text" and str(e.get("data") or "").strip()
    ]
    body = text_parts[-1] if text_parts else partial
    content = f"{body}\n\n{notice}".strip() if body else notice
    events.append({"type": "info", "data": notice})
    return content, events


def _spawn_late_persist(
    *,
    task_id: str,
    user_id: str,
    message_queue: queue.Queue,
    result_holder: dict[str, Any],
    tool_events: list[dict[str, Any]],
    tool_meta: dict[str, dict[str, Any]],
    streamed_text: str,
) -> None:
    """Keep draining the agent queue after SSE ends; persist the final answer."""

    def _late_persist() -> None:
        text = streamed_text
        deadline = time.monotonic() + LATE_PERSIST_WAIT_SECONDS
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Late persist timed out waiting for agent worker"
                    )
                    return
                try:
                    item = message_queue.get(timeout=min(5.0, remaining))
                except queue.Empty:
                    continue
                if item is None:
                    break
                text, _ = _consume_queue_item(
                    item,
                    tool_events=tool_events,
                    tool_meta=tool_meta,
                    streamed_text=text,
                )

            if result_holder.get("cancelled") or run_cancel.is_cancelled(task_id):
                logger.info("Late persist skipped: user cancelled")
                return

            if "error" in result_holder:
                logger.info(
                    "Late persist skipped: agent worker error=%s",
                    result_holder.get("error"),
                )
                return

            final_content, images, events = _build_final_payload(
                result_holder=result_holder,
                tool_events=tool_events,
                streamed_text=text,
            )
            if not (final_content or events):
                logger.info("Late persist skipped: empty final payload")
                return

            # Another path (SSE finally / hydrate) may have already written.
            existing = task_store.list_messages(task_id, user_id)
            if existing and existing[-1].get("role") == "assistant":
                # Text-only hydrate may have won the race — backfill tools.
                last = existing[-1]
                if events and not (last.get("tool_events") or []):
                    updated = task_store.update_message_tool_events(
                        last["id"],
                        task_id,
                        user_id,
                        events,
                    )
                    if updated:
                        run_registry.mark_done(
                            task_id,
                            content=final_content or (last.get("content") or ""),
                            images=images or (last.get("images") or []),
                            tool_events=events,
                        )
                        flush_persist(user_id)
                        _kick_graph_job(user_id)
                        logger.info(
                            "Late persist backfilled tool_events (%s)",
                            len(events),
                        )
                        return
                logger.info("Late persist skipped: assistant already persisted")
                return

            logger.info(
                "Late persist saving assistant message (%s chars, %s events)",
                len(final_content),
                len(events),
            )
            task_store.add_message(
                task_id,
                "assistant",
                final_content,
                user_id=user_id,
                images=images,
                tool_events=events,
            )
            run_registry.mark_done(
                task_id,
                content=final_content,
                images=images,
                tool_events=events,
            )
            flush_persist(user_id)
            _kick_graph_job(user_id)
        except Exception:
            logger.exception("Late persist failed")

    threading.Thread(
        target=_late_persist,
        name="chat-late-persist",
        daemon=True,
    ).start()


def _messages_need_assistant(task_id: str, user_id: str) -> bool:
    messages = task_store.list_messages(task_id, user_id)
    return not messages or messages[-1].get("role") != "assistant"


def _persist_assistant_now(
    *,
    task_id: str,
    user_id: str,
    result_holder: dict[str, Any],
    tool_events: list[dict[str, Any]],
    streamed_text: str,
) -> bool:
    """Write assistant message if DB still ends on user. Returns True if written."""
    if not _messages_need_assistant(task_id, user_id):
        return False
    if "error" in result_holder and not (result_holder.get("content") or "").strip():
        error_text = f"Error: {result_holder['error']}"
        content, events = _build_partial_error_payload(
            tool_events=tool_events,
            streamed_text=streamed_text,
            error_text=error_text,
        )
        task_store.add_message(
            task_id,
            "assistant",
            content,
            user_id=user_id,
            tool_events=events,
        )
        flush_persist(user_id)
        return True
    final_content, images, events = _build_final_payload(
        result_holder=result_holder,
        tool_events=tool_events,
        streamed_text=streamed_text,
    )
    if not (final_content or events):
        return False
    task_store.add_message(
        task_id,
        "assistant",
        final_content,
        user_id=user_id,
        images=images,
        tool_events=events,
    )
    flush_persist(user_id)
    return True


def _run_agent_thread(
    *,
    task_id: str,
    prompt: str,
    user_id: str,
    mcp_servers: list[str],
    model_name: str,
    skill_list: list[str],
    guardrail_enabled: bool,
    llm_gateway_enabled: bool,
    memory_enabled: bool,
    runtime_session_id: str,
    files: list[str],
    message_queue: queue.Queue,
    result_holder: dict[str, Any],
) -> None:
    sink = QueueNotificationSink(message_queue)
    run_registry.mark_running(task_id, user_id)

    try:
        logger.info("Using local LangGraph agent (chat → langgraph_agent)")
        response, image_url = run_agent(
            prompt,
            user_id,
            mcp_servers,
            model_name,
            runtime_session_id,
            notification_queue=sink,
            skill_list=skill_list,
            guardrail_enabled=guardrail_enabled,
            llm_gateway_enabled=llm_gateway_enabled,
            memory_enabled=memory_enabled,
            files=files,
        )
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        result_holder["content"] = response
        result_holder["images"] = image_url or []
        if run_cancel.is_cancelled(task_id) or run_cancel.is_cancelled(runtime_session_id):
            result_holder["cancelled"] = True
        run_registry.mark_done(
            task_id,
            content=response,
            images=image_url or [],
        )
    except Exception as exc:
        logger.exception("Agent run failed")
        result_holder["error"] = str(exc)
        if run_cancel.is_cancelled(task_id) or run_cancel.is_cancelled(runtime_session_id):
            result_holder["cancelled"] = True
        run_registry.mark_done(task_id, error=str(exc))
    finally:
        message_queue.put(None)


@router.post("/{task_id}/chat")
def chat_stream(task_id: str, body: ChatRequest, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    files = [url.strip() for url in (body.files or []) if url and url.strip()]
    prompt = body.prompt.strip()
    if not prompt and files:
        prompt = DEFAULT_IMAGE_PROMPT

    chat.update(
        userId=user_id,
        modelName=task["model_name"],
        debugMode="Enable",
        guardrailEnabled=task["guardrail_enabled"],
        llmGatewayEnabled=task.get("llm_gateway_enabled", False),
        memoryEnabled=task["memory_enabled"],
    )

    task_store.add_message(task_id, "user", prompt, user_id=user_id, images=files)

    runtime_session_id = task.get("runtime_session_id") or task_id
    message_queue: queue.Queue = queue.Queue()
    result_holder: dict[str, Any] = {"content": "", "images": []}

    worker = threading.Thread(
        target=_run_agent_thread,
        kwargs={
            "task_id": task_id,
            "prompt": prompt,
            "user_id": user_id,
            "mcp_servers": task["mcp_servers"],
            "model_name": task["model_name"],
            "skill_list": task["skills"],
            "guardrail_enabled": task["guardrail_enabled"],
            "llm_gateway_enabled": task.get("llm_gateway_enabled", False),
            "memory_enabled": task["memory_enabled"],
            "runtime_session_id": runtime_session_id,
            "files": files,
            "message_queue": message_queue,
            "result_holder": result_holder,
        },
        daemon=True,
    )
    worker.start()


    def event_generator() -> Generator[str, None, None]:
        tool_events: list[dict[str, Any]] = []
        tool_meta: dict[str, dict[str, Any]] = {}
        streamed_text = ""
        sse_closed_early = False
        should_kick_graph = False

        try:
            started_at = time.monotonic()
            # Idle deadline resets whenever the agent emits something.
            deadline = started_at + AGENT_STREAM_TIMEOUT_SECONDS
            max_deadline = started_at + AGENT_STREAM_MAX_SECONDS
            while True:
                now = time.monotonic()
                hard_remaining = max_deadline - now
                idle_remaining = deadline - now
                remaining = min(hard_remaining, idle_remaining)
                if remaining <= 0:
                    timed_out_idle = idle_remaining <= 0
                    elapsed = int(now - started_at)
                    logger.warning(
                        "Agent SSE stream timed out after %ss (%s); scheduling late persist",
                        elapsed,
                        "idle" if timed_out_idle else "max",
                    )
                    error_text = "Agent timeout"
                    content, events = _build_partial_error_payload(
                        tool_events=tool_events,
                        streamed_text=streamed_text,
                        error_text=error_text,
                    )
                    task_store.add_message(
                        task_id,
                        "assistant",
                        content,
                        user_id=user_id,
                        tool_events=events,
                    )
                    yield _sse_event({"type": "error", "data": error_text})
                    yield _sse_event(
                        {
                            "type": "done",
                            "content": content,
                            "images": [],
                            "tool_events": events,
                        }
                    )
                    sse_closed_early = True
                    _spawn_late_persist(
                        task_id=task_id,
                        user_id=user_id,
                        message_queue=message_queue,
                        result_holder=result_holder,
                        tool_events=tool_events,
                        tool_meta=tool_meta,
                        streamed_text=streamed_text,
                    )
                    return

                try:
                    item = message_queue.get(
                        timeout=min(SSE_HEARTBEAT_INTERVAL_SECONDS, remaining)
                    )
                except queue.Empty:
                    yield _sse_keepalive()
                    continue

                if item is None:
                    break

                # Progress received — extend idle window.
                deadline = time.monotonic() + AGENT_STREAM_TIMEOUT_SECONDS

                streamed_text, events_to_emit = _consume_queue_item(
                    item,
                    tool_events=tool_events,
                    tool_meta=tool_meta,
                    streamed_text=streamed_text,
                )
                for event in events_to_emit:
                    yield _sse_event(event)

            if "error" in result_holder:
                error_text = f"Error: {result_holder['error']}"
                content, events = _build_partial_error_payload(
                    tool_events=tool_events,
                    streamed_text=streamed_text,
                    error_text=error_text,
                )
                task_store.add_message(
                    task_id,
                    "assistant",
                    content,
                    user_id=user_id,
                    tool_events=events,
                )
                yield _sse_event({"type": "error", "data": result_holder["error"]})
                yield _sse_event(
                    {
                        "type": "done",
                        "content": content,
                        "images": [],
                        "tool_events": events,
                    }
                )
                return

            final_content, images, events = _build_final_payload(
                result_holder=result_holder,
                tool_events=tool_events,
                streamed_text=streamed_text,
            )

            if result_holder.get("cancelled") or run_cancel.is_cancelled(task_id):
                logger.info(
                    "Agent finished after cancel; skip server persist "
                    "(client stop message)"
                )
                yield _sse_event(
                    {
                        "type": "done",
                        "content": "",
                        "images": images,
                        "tool_events": events,
                        "cancelled": True,
                    }
                )
                return

            task_store.add_message(
                task_id,
                "assistant",
                final_content,
                user_id=user_id,
                images=images,
                tool_events=events,
            )
            should_kick_graph = True

            yield _sse_event(
                {
                    "type": "done",
                    "content": final_content,
                    "images": images,
                    "tool_events": events,
                }
            )
        except GeneratorExit:
            # Stop calls POST /cancel before abort; refresh only aborts SSE.
            user_cancelled = bool(
                result_holder.get("cancelled")
                or run_cancel.is_cancelled(task_id)
                or run_cancel.is_cancelled(runtime_session_id)
            )
            if user_cancelled:
                result_holder["cancelled"] = True
                logger.info(
                    "SSE disconnected after cancel; skip server persist "
                    "(client stop message)"
                )
            elif not sse_closed_early and _messages_need_assistant(task_id, user_id):
                already_final = bool(
                    (result_holder.get("content") or "").strip()
                ) or ("error" in result_holder)
                if already_final:
                    logger.warning(
                        "SSE client disconnected after agent finished; "
                        "persisting assistant immediately"
                    )
                    wrote = _persist_assistant_now(
                        task_id=task_id,
                        user_id=user_id,
                        result_holder=result_holder,
                        tool_events=tool_events,
                        streamed_text=streamed_text,
                    )
                    if wrote:
                        should_kick_graph = True
                else:
                    logger.warning(
                        "SSE client disconnected before agent finished; "
                        "scheduling late persist"
                    )
                    _spawn_late_persist(
                        task_id=task_id,
                        user_id=user_id,
                        message_queue=message_queue,
                        result_holder=result_holder,
                        tool_events=tool_events,
                        tool_meta=tool_meta,
                        streamed_text=streamed_text,
                    )
            raise
        finally:
            if not sse_closed_early:
                flush_persist(user_id)
                if should_kick_graph:
                    _kick_graph_job(user_id)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
