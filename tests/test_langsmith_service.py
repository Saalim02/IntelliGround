from types import SimpleNamespace

import pytest

from app.services.observability import langsmith_service as service


@pytest.fixture(autouse=True)
def clear_client_cache(monkeypatch):
    service._client.cache_clear()
    yield
    if hasattr(service._client, "cache_clear"):
        service._client.cache_clear()


def configure(monkeypatch, enabled="true"):
    monkeypatch.setenv("LANGSMITH_TRACING", enabled)
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("LANGSMITH_PROJECT", "intelliground")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")


def test_disabled_tracing_is_noop(monkeypatch):
    configure(monkeypatch, "false")
    handle = service.start_trace("test_operation", environment="local")
    assert handle.run_id is None
    assert service.tracing_status()["enabled"] is False


def test_missing_configuration_is_safe(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert service.create_connectivity_trace() is False
    assert service.tracing_status()["configured"] is False


def test_enabled_trace_uses_project_and_safe_metadata(monkeypatch):
    configure(monkeypatch)
    calls = []

    class FakeClient:
        def create_run(self, **kwargs): calls.append(("create", kwargs))
        def update_run(self, *args, **kwargs): calls.append(("update", args, kwargs))
        def close(self): pass

    monkeypatch.setattr(service, "_client", lambda: FakeClient())
    handle = service.start_trace("test_operation", environment="local", route="test")
    handle.finish()
    assert calls[0][1]["project_name"] == "intelliground"
    assert calls[0][1]["extra"]["metadata"]["route"] == "test"


def test_sdk_initialization_failure_is_safe(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(service, "_client", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert service.start_trace("test_operation").run_id is None


def test_creation_and_finalization_failures_are_safe(monkeypatch):
    configure(monkeypatch)

    class BrokenClient:
        def create_run(self, **kwargs): raise RuntimeError("create failed")
        def update_run(self, *args, **kwargs): raise RuntimeError("update failed")

    monkeypatch.setattr(service, "_client", lambda: BrokenClient())
    handle = service.start_trace("test_operation")
    handle.finish(status="error", error_category="test")
    assert handle.run_id is None


def test_sensitive_and_unknown_metadata_are_rejected():
    with pytest.raises(ValueError, match="Sensitive"):
        service._metadata(api_key="secret")
    with pytest.raises(ValueError, match="Unsupported"):
        service._metadata(raw_question="do not record")


def test_safe_trace_preserves_application_exceptions(monkeypatch):
    monkeypatch.setattr(service, "start_trace", lambda *args, **kwargs: SimpleNamespace(
        run_id=None, finish=lambda **_kwargs: None,
    ))
    with pytest.raises(RuntimeError, match="application failure"):
        with service.safe_trace("test_operation"):
            raise RuntimeError("application failure")
