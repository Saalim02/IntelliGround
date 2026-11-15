"""Backward-compatible entry point for all application LLM generation."""

from app.services.llm.provider_factory import get_provider_gateway
from app.services.observability.langsmith_service import record_provider_metadata


def get_llm(openai_api_key: str = ""):
    """Return the configured provider gateway.

    The parameter name is retained for compatibility with the existing
    authorization flow. Evon credentials are always read server-side.
    """
    return get_provider_gateway(openai_api_key, metadata_sink=record_provider_metadata)
