"""UI model names ↔ LiteLLM gateway model ids."""

from __future__ import annotations

# UI display name → LiteLLM model_name (litellm-guide/install/models.py + gateway /v1/models)
LLM_GATEWAY_MODEL_MAP: dict[str, str] = {
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
    "OpenAI GPT 5.5": "gpt-5.5",
    "OpenAI GPT 5.4": "gpt-5.4",
    "OpenAI GPT 6 Astra": "gpt-6-astra",
    "OpenAI GPT 5.6 Sol": "gpt-5.6-sol",
    "OpenAI GPT 5.6 Terra": "gpt-5.6-terra",
    "OpenAI GPT 5.6 Luna": "gpt-5.6-luna",
}


def ui_models_for_gateway_ids(
    gateway_ids: list[str] | None,
    *,
    preferred_order: list[str] | None = None,
) -> list[str]:
    """Map gateway /v1/models ids to UI names, preserving preferred_order when given."""
    if preferred_order is None:
        preferred_order = list(LLM_GATEWAY_MODEL_MAP.keys())

    if gateway_ids is None:
        return [name for name in preferred_order if name in LLM_GATEWAY_MODEL_MAP]

    available = {gid for gid in gateway_ids if gid}
    return [
        name
        for name in preferred_order
        if LLM_GATEWAY_MODEL_MAP.get(name) in available
    ]
