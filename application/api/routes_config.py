import logging
import os

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel

try:
    from application import utils
    from application.api.routes_auth import get_optional_user_id, local_auth_bypass_enabled
    from application.api.routes_admin import is_admin_user
    from application.llm_gateway_models import ui_models_for_gateway_ids
except ImportError:
    import utils
    from routes_auth import get_optional_user_id, local_auth_bypass_enabled  # type: ignore
    from routes_admin import is_admin_user  # type: ignore
    from llm_gateway_models import ui_models_for_gateway_ids  # type: ignore

logger = logging.getLogger("routes_config")

router = APIRouter(prefix="/api/config", tags=["config"])

_APPLICATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    "Claude 5.0 Sonnet",
    "Claude 5.0 Opus",
    "Claude 4.6 Sonnet",
    "Claude Fable 5",
    "Claude Fable 5.1",
    "Claude 4.8 Opus",
    "Claude 4.7 Opus",
    "Claude 4.6 Opus",
    "Claude 4.5 Opus",
    "Claude 4.5 Sonnet",
    "Claude 4.5 Haiku",
    "OpenAI GPT 5.4",
    "OpenAI GPT 5.5",
    "OpenAI GPT 6 Astra",
    "OpenAI GPT 5.6 Sol",
    "OpenAI GPT 5.6 Terra",
    "OpenAI GPT 5.6 Luna",
    "OpenAI OSS 120B",
    "OpenAI OSS 20B",
]

DEFAULT_MODEL = "Claude 4.6 Sonnet"
DEFAULT_GATEWAY_MODEL = "Claude 4.6 Sonnet"


def load_capability_list_from_path(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        logger.warning("Capability list not found: %s", path)
        return []


def load_capability_list(filename: str) -> list[str]:
    path = os.path.join(_APPLICATION_DIR, filename)
    return load_capability_list_from_path(path)


class DefaultsPatch(BaseModel):
    default_skills: list[str] | None = None
    default_mcp_servers: list[str] | None = None


class LlmGatewaySettings(BaseModel):
    url: str = ""
    key: str = ""


def _llm_gateway_from_config() -> tuple[str, str]:
    cfg = utils.load_config()
    url = (cfg.get("llm_gateway_url") or "").strip().rstrip("/")
    key = (cfg.get("llm_gateway_key") or "").strip()
    return url, key


def _save_llm_gateway(url: str, key: str) -> None:
    utils.persist_config_updates(
        {
            "llm_gateway_url": url.rstrip("/"),
            "llm_gateway_key": key,
        }
    )


def _probe_llm_gateway(url: str, key: str, *, timeout: float = 15.0) -> dict:
    models_url = f"{url.rstrip('/')}/v1/models"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(
                models_url,
                headers={"Authorization": f"Bearer {key}"},
            )
        if response.status_code >= 400:
            logger.warning(
                "LLM Gateway verify failed: status=%s body=%s",
                response.status_code,
                response.text[:300],
            )
            return {
                "ok": False,
                "message": f"모델 확인 실패 (HTTP {response.status_code})",
                "models": [],
                "ui_models": [],
            }
        payload = response.json()
        models = [
            item.get("id")
            for item in (payload.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        ui_models = ui_models_for_gateway_ids(models, preferred_order=MODELS)
        return {
            "ok": True,
            "message": (
                f"모델 {len(ui_models)}개 확인됨"
                if ui_models
                else f"등록 모델 {len(models)}개 (UI 매핑 없음)"
            ),
            "models": models,
            "ui_models": ui_models,
        }
    except Exception as exc:
        logger.exception("LLM Gateway verify error")
        return {
            "ok": False,
            "message": f"모델 확인 요청 실패: {exc}",
            "models": [],
            "ui_models": [],
        }


def _gateway_ui_models() -> list[str]:
    """Resolve UI model list from live gateway, or mapped catalog fallback."""
    url, key = _llm_gateway_from_config()
    if not url or not key:
        return []

    result = _probe_llm_gateway(url, key, timeout=5.0)
    if result.get("ok") and result.get("ui_models"):
        return result["ui_models"]

    # Gateway configured but probe failed: still expose known mapped UI models.
    return ui_models_for_gateway_ids(None, preferred_order=MODELS)


@router.get("")
def get_config(request: Request):
    session_user = get_optional_user_id(request)
    if session_user:
        skills_path = utils.ensure_user_skills_list(session_user)
        skill_options = load_capability_list_from_path(skills_path)
        logger.info("Loaded skills from %s (%d)", skills_path, len(skill_options))
    else:
        skill_options = load_capability_list("skills.list")
    mcp_options = load_capability_list("mcp.list")
    # Prefer per-user settings.json; fall back to favorite_tools.json.
    default_skills, default_mcp = utils.get_user_tool_defaults(session_user)
    default_skills = [s for s in default_skills if s in skill_options]
    default_mcp = [m for m in default_mcp if m in mcp_options]
    if not default_skills and "skill-creator" in skill_options:
        default_skills = ["skill-creator"]
    if not default_mcp:
        logger.info("No initial MCP defaults matched current capability list")
    config = utils.load_config()
    gateway_url, gateway_key = _llm_gateway_from_config()
    gateway_models = _gateway_ui_models()
    default_gateway_model = (
        DEFAULT_GATEWAY_MODEL
        if DEFAULT_GATEWAY_MODEL in gateway_models
        else (gateway_models[0] if gateway_models else DEFAULT_MODEL)
    )
    return {
        "projectName": config.get("projectName", "agent"),
        "google_client_id": (config.get("google_client_id") or "").strip(),
        "local_auth_bypass": local_auth_bypass_enabled(request),
        "is_admin": bool(session_user and is_admin_user(session_user)),
        "skills": skill_options,
        "mcp_servers": mcp_options,
        "models": MODELS,
        "gateway_models": gateway_models,
        "default_model": DEFAULT_MODEL,
        "default_gateway_model": default_gateway_model,
        "default_skills": default_skills,
        "default_mcp_servers": default_mcp,
        "llm_gateway_configured": bool(gateway_url and gateway_key),
    }


@router.get("/llm-gateway")
def get_llm_gateway():
    url, key = _llm_gateway_from_config()
    return {
        "url": url,
        "key": key,
        "configured": bool(url and key),
        "key_configured": bool(key),
    }


@router.post("/llm-gateway/verify")
def verify_llm_gateway(body: LlmGatewaySettings | None = None):
    """Probe LiteLLM /v1/models with form (or config) values; save on success."""
    stored_url, stored_key = _llm_gateway_from_config()
    if body is not None:
        url = (body.url or "").strip().rstrip("/") or stored_url
        key = (body.key or "").strip() or stored_key
    else:
        url, key = stored_url, stored_key

    if not url or not key:
        return {
            "ok": False,
            "message": "url과 key가 모두 필요합니다.",
            "models": [],
            "ui_models": [],
        }

    result = _probe_llm_gateway(url, key)
    if result.get("ok"):
        _save_llm_gateway(url, key)
    return result


@router.patch("/defaults")
def patch_defaults(body: DefaultsPatch):
    utils.save_favorite_tools(
        skills=body.default_skills,
        mcp_servers=body.default_mcp_servers,
    )
    return {"ok": True}
