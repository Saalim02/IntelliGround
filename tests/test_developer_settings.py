from datetime import datetime, timedelta, timezone
import uuid

import jwt

from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.auth_models import User
from app.services.security.auth_service import create_access_token, create_user, get_jwt_secret
from app.services.security.developer_access import (
    DEVELOPER_ACCESS_COOKIE,
    DEVELOPER_ACCESS_RATE_LIMITER,
)


client = TestClient(app)


def auth_user():
    db = SessionLocal()
    try:
        user = create_user(db, f"developer-{uuid.uuid4().hex}@example.com", "DeveloperPassword123!")
        return user, {"Authorization": f"Bearer {create_access_token(user.id, user.token_version)}"}
    finally:
        db.close()


def unlock(headers, monkeypatch, password="developer-secret"):
    monkeypatch.setattr("app.api.routes.auth.DEVELOPER_ACCESS_PASSWORD", password)
    response = client.post("/api/auth/developer-access", headers=headers, json={"password": password})
    assert response.status_code == 200


def test_developer_dashboard_requires_developer_session():
    _, headers = auth_user()
    response = client.get("/api/developer/dashboard", headers=headers)
    assert response.status_code == 403


def test_developer_access_session_authorizes_dashboard(monkeypatch):
    DEVELOPER_ACCESS_RATE_LIMITER.reset()
    _, headers = auth_user()
    unlock(headers, monkeypatch)
    response = client.get("/api/developer/dashboard", headers=headers)
    assert response.status_code == 200
    assert "observability" in response.json()


def test_missing_developer_password_fails_closed(monkeypatch):
    DEVELOPER_ACCESS_RATE_LIMITER.reset()
    _, headers = auth_user()
    monkeypatch.setattr("app.api.routes.auth.DEVELOPER_ACCESS_PASSWORD", "")
    response = client.post("/api/auth/developer-access", headers=headers, json={"password": "anything"})
    assert response.status_code == 503
    assert "anything" not in response.text


def test_expired_developer_session_is_rejected(monkeypatch):
    DEVELOPER_ACCESS_RATE_LIMITER.reset()
    user, headers = auth_user()
    expired = jwt.encode(
        {"sub": str(user.id), "type": "developer_access", "token_version": user.token_version,
         "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        get_jwt_secret(), algorithm="HS256",
    )
    client.cookies.set(DEVELOPER_ACCESS_COOKIE, expired)
    response = client.get("/api/developer/dashboard", headers=headers)
    assert response.status_code == 403
    client.cookies.delete(DEVELOPER_ACCESS_COOKIE)
