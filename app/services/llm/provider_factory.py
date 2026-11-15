from __future__ import annotations

import os
from dataclasses import dataclass

from app.services.llm.base import LLMProviderError
from app.services.llm.evon_provider import create_evon_llm
from app.services.llm.fallback import FallbackChatModel
from app.services.llm.openai_provider import create_openai_llm


@dataclass(frozen=True)
class ProviderSettings:
    primary: str
    fallback: str
    openai_model: str
    evon_model: str
    evon_base_url: str
    timeout: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        primary = os.getenv("LLM_PRIMARY_PROVIDER", "openai").strip().lower()
        fallback = os.getenv("LLM_FALLBACK_PROVIDER", "openai").strip().lower()
        if primary not in {"openai", "evon"} or fallback not in {"openai", "evon"}:
            raise LLMProviderError("Unsupported LLM provider configuration.", category="provider_configuration_error")
        try:
            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
            retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
        except ValueError as exc:
            raise LLMProviderError("Invalid LLM retry or timeout configuration.", category="provider_configuration_error") from exc
        if timeout <= 0 or retries < 0 or retries > 3:
            raise LLMProviderError("Invalid LLM retry or timeout configuration.", category="provider_configuration_error")
        return cls(primary, fallback, os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(), os.getenv("EVON_MODEL", "gnani-evon-v3.3").strip(), os.getenv("EVON_BASE_URL", "").strip(), timeout, retries)


def _make(name: str, settings: ProviderSettings, *, openai_api_key: str):
    if name == "openai":
        return create_openai_llm(api_key=openai_api_key or os.getenv("OPENAI_API_KEY", ""), model=settings.openai_model, timeout=settings.timeout, max_retries=settings.max_retries)
    return create_evon_llm(api_key=os.getenv("EVON_API_KEY", ""), base_url=settings.evon_base_url, model=settings.evon_model, timeout=settings.timeout, max_retries=settings.max_retries)


def get_provider_gateway(openai_api_key: str = "", metadata_sink=None):
    settings = ProviderSettings.from_env()
    primary = _make(settings.primary, settings, openai_api_key=openai_api_key)
    fallback = None if settings.fallback == settings.primary else _make(settings.fallback, settings, openai_api_key=openai_api_key)
    return FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_name=settings.primary,
        fallback_name=settings.fallback if fallback is not None else None,
        primary_model=settings.evon_model if settings.primary == "evon" else settings.openai_model,
        fallback_model=(settings.evon_model if settings.fallback == "evon" else settings.openai_model) if fallback is not None else None,
        metadata_sink=metadata_sink,
    )


def provider_status() -> dict:
    settings = ProviderSettings.from_env()
    return {
        "primary_provider": settings.primary,
        "fallback_provider": settings.fallback,
        "openai_model": settings.openai_model,
        "evon_model": settings.evon_model,
        "evon_base_url_configured": bool(settings.evon_base_url),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "evon_key_configured": bool(os.getenv("EVON_API_KEY", "").strip()),
        "evon_live_verification": "pending_official_endpoint_and_credentials",
    }
