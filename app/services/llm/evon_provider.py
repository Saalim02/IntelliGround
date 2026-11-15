"""Evon adapter for an explicitly supplied OpenAI-compatible hosted endpoint.

Gnani's public Evon v3.3 material does not currently publish a stable endpoint
or authentication contract. Therefore no endpoint is inferred here: deployments
must provide EVON_BASE_URL and EVON_API_KEY after receiving official access.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.services.llm.base import LLMProviderError


def create_evon_llm(*, api_key: str, base_url: str, model: str, timeout: float, max_retries: int):
    key = str(api_key or "").strip()
    url = str(base_url or "").strip()
    if not key:
        raise LLMProviderError("Evon credentials are not configured.", category="provider_configuration_error")
    if not url:
        raise LLMProviderError("Evon base URL is not configured.", category="provider_configuration_error")
    if not model.strip():
        raise LLMProviderError("Evon model is not configured.", category="provider_configuration_error")
    return ChatOpenAI(model=model, temperature=0, api_key=key, base_url=url, timeout=timeout, max_retries=max_retries)
