from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from application.api.routes_auth import require_user_id, _llm_gateway_ready
from application import task_store
from application import utils
from application import run_cancel
from application.run_state import query_task_run
from application.task_store_persistence import flush_persist
import logging

logger = logging.getLogger("routes_tasks")

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

_LLM_GATEWAY_NOT_CONFIGURED = (
    "LLM Gateway가 설정되어 있지 않아 활성화할 수 없습니다."
)


class TaskCreate(BaseModel):
    model_name: str | None = None
    skills: list[str] | None = None
    mcp_servers: list[str] | None = None
    guardrail_enabled: bool = False
    llm_gateway_enabled: bool = False
    memory_enabled: bool = True
    title: str = "New task"


class TaskPatch(BaseModel):
    title: str | None = None
    model_name: str | None = None
    skills: list[str] | None = None
    mcp_servers: list[str] | None = None
    guardrail_enabled: bool | None = None
    llm_gateway_enabled: bool | None = None
    memory_enabled: bool | None = None
    pinned: bool | None = None



class MessageCreate(BaseModel):
    role: str = "assistant"
    content: str = ""
    images: list[str] = Field(default_factory=list)
    tool_events: list[dict] = Field(default_factory=list)


def _require_gateway_if_enabling(enabled: bool | None) -> None:
    if enabled and not _llm_gateway_ready():
        raise HTTPException(status_code=400, detail=_LLM_GATEWAY_NOT_CONFIGURED)


@router.get("")
def list_tasks(request: Request, limit: int = 100):
    user_id = require_user_id(request)
    return {"tasks": task_store.list_tasks(user_id, limit=limit)}


@router.post("")
def create_task(body: TaskCreate, request: Request):
    user_id = require_user_id(request)
    _require_gateway_if_enabling(body.llm_gateway_enabled)
    skills, mcp_servers = _resolve_tool_defaults(
        user_id, body.skills, body.mcp_servers
    )
    # Remember the user's last selection for subsequent new tasks.
    utils.save_user_tool_defaults(
        user_id, skills=skills, mcp_servers=mcp_servers
    )
    task = task_store.create_task(
        user_id,
        model_name=body.model_name,
        skills=skills,
        mcp_servers=mcp_servers,
        guardrail_enabled=body.guardrail_enabled,
        llm_gateway_enabled=body.llm_gateway_enabled,
        memory_enabled=body.memory_enabled,
        title=body.title,
    )
    return task


@router.get("/{task_id}")
def get_task(task_id: str, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}")
def patch_task(task_id: str, body: TaskPatch, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    patch = body.model_dump(exclude_unset=True)
    _require_gateway_if_enabling(patch.get("llm_gateway_enabled"))
    updated = task_store.update_task(
        task_id,
        user_id,
        **patch,
    )
    if updated and ("skills" in patch or "mcp_servers" in patch):
        utils.save_user_tool_defaults(
            user_id,
            skills=patch.get("skills"),
            mcp_servers=patch.get("mcp_servers"),
        )
    return updated


@router.delete("/{task_id}")
def remove_task(task_id: str, request: Request):
    user_id = require_user_id(request)
    if not task_store.delete_task(task_id, user_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}



def _resolve_tool_defaults(
    user_id: str,
    skills: list[str] | None,
    mcp_servers: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Fill missing skill/MCP from settings.json (else favorite_tools)."""
    default_skills, default_mcp = utils.get_user_tool_defaults(user_id)
    resolved_skills = list(skills) if skills is not None else list(default_skills)
    resolved_mcp = (
        list(mcp_servers) if mcp_servers is not None else list(default_mcp)
    )
    return resolved_skills, resolved_mcp

@router.get("/{task_id}/messages")
def get_messages(task_id: str, request: Request):
    user_id = require_user_id(request)
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"messages": task_store.list_messages(task_id, user_id)}


@router.get("/{task_id}/run")
def get_task_run(task_id: str, request: Request):
    """Query agent run status for refresh recovery (no direct checkpoint I/O for clients).

    Status values: idle | running | pending | done | error | missing
    source: messages | registry | checkpoint | null
    When status is done and messages still end on user, hydrates the assistant row.
    """
    user_id = require_user_id(request)
    result = query_task_run(task_id, user_id)
    if result.get("status") == "missing":
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.post("/{task_id}/messages")
def create_message(task_id: str, body: MessageCreate, request: Request):
    """Persist a message (e.g. client-side stop notice) into the task transcript."""
    user_id = require_user_id(request)
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    role = (body.role or "assistant").strip().lower()
    if role not in ("assistant", "user"):
        raise HTTPException(status_code=400, detail="role must be assistant or user")
    content = (body.content or "").strip()
    if not content and not body.tool_events:
        raise HTTPException(status_code=400, detail="content or tool_events required")
    message = task_store.add_message(
        task_id,
        role,
        content,
        user_id=user_id,
        images=body.images,
        tool_events=body.tool_events,
    )
    flush_persist(user_id)
    return message


@router.post("/{task_id}/cancel")
def cancel_task_run(task_id: str, request: Request):
    """Signal the in-process agent worker to stop tool/model loops for this task.

    History (LangGraph checkpoint thread_id == task_id) is preserved so the next
    user message continues the same conversation.
    """
    user_id = require_user_id(request)
    task = task_store.get_task_refreshing(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    session_id = task.get("runtime_session_id") or task_id
    run_cancel.request_cancel(task_id)
    if session_id != task_id:
        run_cancel.request_cancel(session_id)
    logger.info(
        "Cancel requested: task_id=%s session_id=%s user=%s",
        task_id,
        session_id,
        user_id,
    )
    return {"ok": True, "task_id": task_id, "cancelled": True}
