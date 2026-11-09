from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.services.security import rag_authorization


def _user(email: str, auth_provider: str = "local"):
    return SimpleNamespace(
        email=email,
        auth_provider=auth_provider,
        is_active=True,
    )


def test_local_development_identity_uses_backend_key(monkeypatch):
    monkeypatch.setattr(rag_authorization, "LOCAL_DEV_AUTH_BYPASS", True)
    monkeypatch.setenv("OPENAI_API_KEY", "server-only-test-key")

    result = rag_authorization.resolve_authorized_api_key(
        user=_user("localdev@intelliground.local", "local-dev"),
    )

    assert result == "server-only-test-key"


def test_normal_user_still_requires_project_authorization(monkeypatch):
    monkeypatch.setattr(rag_authorization, "LOCAL_DEV_AUTH_BYPASS", True)
    monkeypatch.setenv("OPENAI_API_KEY", "server-only-test-key")
    monkeypatch.setenv("RAG_ACCESS_PASSWORD", "project-password")

    with pytest.raises(HTTPException) as error:
        rag_authorization.resolve_authorized_api_key(
            user=_user("person@example.com"),
        )

    assert error.value.status_code == 403


def test_local_bypass_disabled_preserves_existing_rules(monkeypatch):
    monkeypatch.setattr(rag_authorization, "LOCAL_DEV_AUTH_BYPASS", False)
    monkeypatch.setenv("OPENAI_API_KEY", "server-only-test-key")
    monkeypatch.setenv("RAG_ACCESS_PASSWORD", "project-password")

    with pytest.raises(HTTPException) as error:
        rag_authorization.resolve_authorized_api_key(
            user=_user("localdev@intelliground.local", "local-dev"),
        )

    assert error.value.status_code == 403
