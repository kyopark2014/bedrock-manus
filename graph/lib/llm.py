"""LLM chat client: LiteLLM gateway first, Bedrock Converse fallback.

Gateway path uses OpenAI-compatible /v1 (any LiteLLM model id). When
``application/config.json`` has no llm_gateway_*, falls back to AWS Bedrock
``converse`` like ``runtime_agent/langgraph`` (boto3 credential chain).
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from lib.config import (
    bedrock_settings,
    graphify_llm_model,
    llm_gateway_settings,
)

# Friendly / UI names → LiteLLM gateway model ids (same as application map).
# Unknown ids are passed through unchanged so any LiteLLM model works.
_MODEL_ALIASES: dict[str, str] = {
    "Claude 5.0 Sonnet": "claude-sonnet-5",
    "Claude 5.0 Opus": "claude-opus-5",
    "Claude 4.6 Sonnet": "claude-sonnet-4-6",
    "Claude 4.5 Sonnet": "claude-sonnet-4-5",
    "Claude Fable 5": "claude-fable-5",
    "Claude Fable 5.1": "claude-fable-5-1",
    "Claude 4.8 Opus": "claude-opus-4-8",
    "Claude 4.7 Opus": "claude-opus-4-7",
    "Claude 4.6 Opus": "claude-opus-4-6",
    "Claude 4.5 Opus": "claude-opus-4-5",
    "Claude 4.5 Haiku": "claude-haiku-4-5",
    "haiku 4.5": "claude-haiku-4-5",
    "haiku-4.5": "claude-haiku-4-5",
    "OpenAI GPT 5.5": "gpt-5.5",
    "OpenAI GPT 5.4": "gpt-5.4",
    "OpenAI GPT 6 Astra": "gpt-6-astra",
    "OpenAI GPT 5.6 Sol": "gpt-5.6-sol",
    "OpenAI GPT 5.6 Terra": "gpt-5.6-terra",
    "OpenAI GPT 5.6 Luna": "gpt-5.6-luna",
}

# Gateway / short ids → Bedrock inference profile ids (langgraph info.py).
_BEDROCK_MODEL_IDS: dict[str, str] = {
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}


def resolve_model_id(model: str) -> str:
    """Map UI / shorthand names to gateway ids; pass LiteLLM ids through."""
    raw = (model or "").strip()
    if not raw:
        return raw
    if raw in _MODEL_ALIASES:
        return _MODEL_ALIASES[raw]
    lower = raw.lower()
    for key, value in _MODEL_ALIASES.items():
        if key.lower() == lower:
            return value
    return raw


def resolve_bedrock_model_id(model: str) -> str:
    """Map gateway alias / UI name to a Bedrock model id."""
    raw = (model or "").strip()
    if not raw:
        return _BEDROCK_MODEL_IDS["claude-haiku-4-5"]
    # Already a Bedrock / inference-profile id.
    if (
        raw.startswith("us.")
        or raw.startswith("eu.")
        or raw.startswith("apac.")
        or ".anthropic." in raw
        or raw.startswith("anthropic.")
        or raw.startswith("amazon.")
        or raw.startswith("openai.")
    ):
        return raw
    gateway_id = resolve_model_id(raw)
    if gateway_id in _BEDROCK_MODEL_IDS:
        return _BEDROCK_MODEL_IDS[gateway_id]
    lower = gateway_id.lower()
    for key, value in _BEDROCK_MODEL_IDS.items():
        if key.lower() == lower:
            return value
    raise SystemExit(
        f"No Bedrock model mapping for '{raw}'.\n"
        "Set GRAPHIFY_BEDROCK_MODEL to a full Bedrock id "
        "(e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0), "
        "or configure llm_gateway_url / llm_gateway_key in application/config.json."
    )


def _model_family(model: str) -> str:
    """Heuristic provider family for request-option adaptation."""
    m = model.lower()
    if m.startswith("claude") or "anthropic" in m or m.startswith("haiku"):
        return "claude"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if m.startswith("gemini") or m.startswith("palm"):
        return "google"
    if m.startswith("command") or "cohere" in m:
        return "cohere"
    if "llama" in m or m.startswith("meta"):
        return "meta"
    if "mistral" in m or m.startswith("mixtral"):
        return "mistral"
    if "nova" in m or m.startswith("amazon"):
        return "amazon"
    return "other"


def default_model() -> str:
    """Resolved GRAPHIFY_LLM_MODEL (gateway id); works without gateway."""
    return resolve_model_id(graphify_llm_model())


def make_client() -> tuple[OpenAI, str]:
    """Return OpenAI-compatible client pointed at LiteLLM + resolved default model.

    Raises SystemExit if gateway is not configured (use ``chat_json`` for
    automatic Bedrock fallback).
    """
    gw = llm_gateway_settings()
    if gw is None:
        raise SystemExit(
            "LiteLLM gateway is not configured.\n"
            "Set llm_gateway_url / llm_gateway_key in application/config.json,\n"
            "or use chat_json() which falls back to Bedrock Converse."
        )
    client = OpenAI(api_key=gw["key"], base_url=gw["base_url"])
    return client, resolve_model_id(gw["model"])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    use_json_object: bool,
) -> dict[str, Any]:
    """Build chat.completions kwargs safe for the target model family."""
    family = _model_family(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    # Newer Claude (Opus 4.8+) and some reasoning models reject temperature.
    if temperature is not None and family not in ("claude",):
        kwargs["temperature"] = temperature
    elif temperature is not None and family == "claude":
        # Older Claude via LiteLLM often accepts temperature; try only if set
        # and not a known-strict id. Prefer omitting for haiku/sonnet/opus 4.x+.
        if not re.search(r"claude-(opus|sonnet|haiku)-4", model.lower()):
            kwargs["temperature"] = temperature

    # json_object is OpenAI-oriented; Claude/others often need plain text JSON.
    if use_json_object and family == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


def _split_system_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Split OpenAI-style messages into Bedrock system + converse messages."""
    system: list[dict[str, str]] = []
    converse_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        if role == "system":
            system.append({"text": content})
            continue
        if role not in ("user", "assistant"):
            role = "user"
        converse_messages.append(
            {"role": role, "content": [{"text": content}]}
        )
    if not converse_messages:
        converse_messages = [{"role": "user", "content": [{"text": ""}]}]
    return system, converse_messages


def _chat_json_bedrock(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Call Bedrock Converse and parse a JSON object response."""
    import boto3
    from botocore.config import Config

    settings = bedrock_settings()
    model_id = resolve_bedrock_model_id(model or settings["model"])
    region = settings["region"]

    system, converse_messages = _split_system_messages(messages)
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 10}, read_timeout=300),
    )

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": converse_messages,
    }
    if system:
        kwargs["system"] = system

    resp = client.converse(**kwargs)
    blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("text")]
    content = "\n".join(parts).strip()
    data = json.loads(_strip_fences(content))
    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON")

    usage = resp.get("usage") or {}
    data.setdefault("input_tokens", int(usage.get("inputTokens") or 0))
    data.setdefault("output_tokens", int(usage.get("outputTokens") or 0))
    data.setdefault("_llm_backend", "bedrock")
    data.setdefault("_llm_model", model_id)
    data.setdefault("_llm_region", region)
    return data


def _chat_json_gateway(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float | None,
    gw: dict[str, str],
) -> dict[str, Any]:
    client = OpenAI(api_key=gw["key"], base_url=gw["base_url"])
    model = resolve_model_id(model or gw["model"])

    # Try progressively more compatible request shapes.
    attempts: list[dict[str, Any]] = [
        _completion_kwargs(model, messages, temperature=temperature, use_json_object=True),
        _completion_kwargs(model, messages, temperature=temperature, use_json_object=False),
        _completion_kwargs(model, messages, temperature=None, use_json_object=False),
    ]

    # Deduplicate identical kwargs
    seen: set[str] = set()
    unique_attempts: list[dict[str, Any]] = []
    for kwargs in attempts:
        key = json.dumps(kwargs, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique_attempts.append(kwargs)

    last_err: Exception | None = None
    resp = None
    for kwargs in unique_attempts:
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    if resp is None:
        assert last_err is not None
        raise last_err

    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    data = json.loads(_strip_fences(content))
    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON")
    if usage is not None:
        data.setdefault("input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
        data.setdefault("output_tokens", getattr(usage, "completion_tokens", 0) or 0)
    data.setdefault("_llm_backend", "gateway")
    data.setdefault("_llm_model", model)
    data.setdefault("_llm_source", gw.get("source", "gateway"))
    return data


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call LiteLLM gateway, or Bedrock Converse if gateway is unset.

    ``model`` may be a LiteLLM gateway id (claude-*, gpt-*, …), a UI alias,
    or a Bedrock model id when using the Bedrock fallback.
    """
    gw = llm_gateway_settings()
    if gw is not None:
        return _chat_json_gateway(
            messages, model=model, temperature=temperature, gw=gw
        )
    return _chat_json_bedrock(messages, model=model)
