import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.services.llm.base import LLMProviderError, classify_provider_exception
from app.services.llm.evon_provider import create_evon_llm
from app.services.llm.fallback import FallbackChatModel
from app.services.llm.provider_factory import ProviderSettings, get_provider_gateway


def test_default_provider_configuration_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PRIMARY_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    settings = ProviderSettings.from_env()
    assert settings.primary == "openai"
    assert settings.fallback == "openai"
    assert settings.openai_model == "gpt-4o-mini"


def test_evon_requires_explicit_credentials_and_endpoint():
    with pytest.raises(LLMProviderError, match="Evon credentials"):
        create_evon_llm(api_key="", base_url="https://example.invalid", model="gnani-evon-v3.3", timeout=60, max_retries=1)
    with pytest.raises(LLMProviderError, match="Evon base URL"):
        create_evon_llm(api_key="secret", base_url="", model="gnani-evon-v3.3", timeout=60, max_retries=1)


def test_factory_selects_evon_without_leaking_key(monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "evon")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openai")
    monkeypatch.setenv("EVON_API_KEY", "evon-secret")
    monkeypatch.setenv("EVON_BASE_URL", "https://evon.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    gateway = get_provider_gateway("request-openai-key")
    assert gateway.primary.model_name == "gnani-evon-v3.3"
    assert gateway.primary.openai_api_key.get_secret_value() == "evon-secret"
    assert gateway.fallback.openai_api_key.get_secret_value() == "request-openai-key"


def _model(result=None, error=None):
    class Fake:
        def invoke(self, messages, **kwargs):
            if error:
                raise error
            return result
    return Fake()


def test_fallback_after_timeout_and_records_reason():
    captured = []
    gateway = FallbackChatModel(
        primary=_model(error=TimeoutError("network timeout")),
        fallback=_model(AIMessage(content="fallback")),
        primary_name="evon", fallback_name="openai", primary_model="evon", fallback_model="gpt-4o-mini",
        metadata_sink=captured.append,
    )
    result = gateway.invoke("question")
    assert result.content == "fallback"
    assert captured[0].fallback_triggered is True
    assert captured[0].fallback_reason == "timeout"


def test_authentication_failure_does_not_fallback():
    captured = []
    gateway = FallbackChatModel(
        primary=_model(error=RuntimeError("401 unauthorized")), fallback=_model(AIMessage(content="wrong")),
        primary_name="evon", fallback_name="openai", primary_model="evon", fallback_model="gpt-4o-mini",
        metadata_sink=captured.append,
    )
    with pytest.raises(LLMProviderError):
        gateway.invoke("question")
    assert not captured


def test_provider_error_classification_is_safe():
    category, retryable = classify_provider_exception(RuntimeError("token=secret 503"))
    assert category == "provider_server_error"
    assert retryable is True


def test_structured_output_is_supported_by_gateway():
    class Result(BaseModel):
        answer: str

    class StructuredFake:
        def invoke(self, value):
            return Result(answer="ok")

    class Fake:
        def with_structured_output(self, schema, **kwargs):
            return StructuredFake()

        def invoke(self, messages, **kwargs):
            return AIMessage(content="ok")

    gateway = FallbackChatModel(primary=Fake(), primary_name="openai", primary_model="gpt-4o-mini")
    assert gateway.with_structured_output(Result).invoke("question").answer == "ok"
