from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.services.llm.base import LLMProviderError, ProviderMetadata, classify_provider_exception

logger = logging.getLogger(__name__)


class FallbackChatModel(BaseChatModel):
    """Invoke primary once, then fallback only for retryable infrastructure errors."""

    primary: Any
    fallback: Any | None = None
    primary_name: str
    fallback_name: str | None = None
    primary_model: str
    fallback_model: str | None = None
    metadata_sink: Any = None

    @property
    def _llm_type(self) -> str:
        return "intelliground-provider-gateway"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        started = time.perf_counter()
        try:
            response = self.primary.invoke(messages, stop=stop, **kwargs)
            if not getattr(response, "content", ""):
                raise LLMProviderError("Primary provider returned empty output.", category="invalid_provider_response", retryable=True)
            self._record(ProviderMetadata(self.primary_name, self.primary_model, duration_ms=(time.perf_counter() - started) * 1000))
            return ChatResult(generations=[ChatGeneration(message=response)])
        except Exception as exc:
            category, retryable = classify_provider_exception(exc)
            if not retryable or self.fallback is None:
                logger.warning("llm_provider_failed provider=%s category=%s", self.primary_name, category)
                if isinstance(exc, LLMProviderError):
                    raise
                raise LLMProviderError("Primary LLM provider failed.", category=category, retryable=retryable) from exc
            logger.warning("llm_provider_fallback primary=%s fallback=%s reason=%s", self.primary_name, self.fallback_name, category)
            try:
                response = self.fallback.invoke(messages, stop=stop, **kwargs)
                if not getattr(response, "content", ""):
                    raise LLMProviderError("Fallback provider returned empty output.", category="invalid_provider_response")
            except Exception as fallback_exc:
                fallback_category, _ = classify_provider_exception(fallback_exc)
                raise LLMProviderError("LLM providers are unavailable.", category=fallback_category) from fallback_exc
            self._record(ProviderMetadata(self.fallback_name or "unknown", self.fallback_model or "unknown", fallback_triggered=True, fallback_reason=category, duration_ms=(time.perf_counter() - started) * 1000))
            return ChatResult(generations=[ChatGeneration(message=response)])

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Preserve provider-native structured output while retaining fallback."""
        primary = self.primary.with_structured_output(schema, include_raw=include_raw, **kwargs)
        fallback = self.fallback.with_structured_output(schema, include_raw=include_raw, **kwargs) if self.fallback is not None else None

        def invoke_structured(value):
            try:
                return primary.invoke(value)
            except Exception as exc:
                category, retryable = classify_provider_exception(exc)
                if not retryable or fallback is None:
                    raise
                logger.warning("llm_provider_fallback primary=%s fallback=%s reason=%s", self.primary_name, self.fallback_name, category)
                return fallback.invoke(value)

        runnable = RunnableLambda(invoke_structured)
        return runnable

    def _record(self, metadata: ProviderMetadata) -> None:
        if self.metadata_sink is not None:
            self.metadata_sink(metadata)
