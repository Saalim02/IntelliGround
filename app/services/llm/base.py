"""Provider-neutral types and safe error classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    model: str
    request_id: str | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    token_usage_available: bool = False
    duration_ms: float | None = None


class LLMProviderError(RuntimeError):
    """Safe provider error; details are intentionally not exposed to callers."""

    def __init__(self, message: str, *, category: str, retryable: bool = False):
        super().__init__(message)
        self.category = category
        self.retryable = retryable


def classify_provider_exception(exc: Exception) -> tuple[str, bool]:
    """Return a non-sensitive category and whether fallback is appropriate."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text:
        return "timeout", True
    if "connection" in name or "connect" in text:
        return "connection_failure", True
    if any(code in text for code in ("429", "rate limit", "rate_limit")):
        return "rate_limited", True
    if any(code in text for code in ("500", "502", "503", "504", "server error")):
        return "provider_server_error", True
    if "outputparser" in name or "structured output" in text or "jsondecode" in name or "json decode" in text or "parsing" in text:
        return "invalid_provider_response", True
    if "authentication" in name or "unauthorized" in text or "401" in text or "403" in text:
        return "provider_authentication_error", False
    if "configuration" in name or "missing" in text:
        return "provider_configuration_error", False
    return "provider_error", False
