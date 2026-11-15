from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.services.llm.base import LLMProviderError


def create_openai_llm(*, api_key: str, model: str, timeout: float, max_retries: int):
    key = str(api_key or "").strip()
    if not key:
        raise LLMProviderError("OpenAI credentials are not configured.", category="provider_configuration_error")
    if not model.strip():
        raise LLMProviderError("OpenAI model is not configured.", category="provider_configuration_error")
    return ChatOpenAI(model=model, temperature=0, api_key=key, timeout=timeout, max_retries=max_retries)
