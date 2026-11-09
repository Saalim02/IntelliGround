import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.api.routes import voice as voice_route
from app.main import app
from app.core import database as database_module
from app.core.database import Base, SessionLocal
from app.models.auth_models import User
from app.services.security.auth_service import create_access_token, create_user


client = TestClient(app)


def authenticated_user() -> tuple[User, dict[str, str]]:
    db = SessionLocal()
    user = create_user(
        db,
        email=f"onboarding-{uuid.uuid4().hex}@example.com",
        password="Onboarding123!",
    )
    token = create_access_token(user.id, user.token_version)
    db.close()
    return user, {"Authorization": f"Bearer {token}"}


def test_onboarding_completion_requires_authentication():
    response = client.post("/api/auth/onboarding/complete")
    assert response.status_code == 401


def test_onboarding_completion_is_scoped_to_authenticated_user():
    first, first_headers = authenticated_user()
    second, second_headers = authenticated_user()

    assert client.post("/api/auth/onboarding/complete", headers=first_headers).status_code == 200

    db = SessionLocal()
    try:
        first_state = db.query(User).filter(User.id == first.id).one().onboarding_completed
        second_state = db.query(User).filter(User.id == second.id).one().onboarding_completed
    finally:
        db.close()

    assert first_state is True
    assert second_state is False
    assert client.get("/api/auth/me", headers=second_headers).json()["onboarding_completed"] is False


def test_onboarding_speech_requires_authentication():
    response = client.post("/api/voice/onboarding-speech", json={"text": "Hello", "language": "en"})
    assert response.status_code == 401


def test_onboarding_speech_rejects_unverified_urdu(monkeypatch):
    _, headers = authenticated_user()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Timbre must not be called for unsupported speech languages")

    monkeypatch.setattr(voice_route, "synthesize_speech", fail_if_called)
    response = client.post(
        "/api/voice/onboarding-speech",
        headers=headers,
        json={"text": "خوش آمدید", "language": "ur"},
    )

    assert response.status_code == 422
    assert called is False


def test_onboarding_speech_enforces_request_body_limit():
    _, headers = authenticated_user()
    response = client.post(
        "/api/voice/onboarding-speech",
        headers=headers,
        content=(b'{"text":"' + b"a" * 9000 + b'","language":"en"}'),
    )
    assert response.status_code == 413


def test_existing_sqlite_installation_gets_completion_column(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email VARCHAR(320) NOT NULL UNIQUE,
                password_hash TEXT,
                auth_provider VARCHAR(20) NOT NULL,
                google_sub VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT 1,
                token_version INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """))
        connection.execute(text(
            "INSERT INTO users (id, email, auth_provider, created_at, updated_at) "
            "VALUES (1, 'legacy@example.com', 'local', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ))

    monkeypatch.setattr(database_module, "engine", engine)
    database_module.init_db()

    columns = {column["name"] for column in inspect(engine).get_columns("users")}
    assert {"first_name", "last_name", "onboarding_completed"}.issubset(columns)
    session = sessionmaker(bind=engine)()
    try:
        assert session.execute(text("SELECT onboarding_completed FROM users")).scalar() in (False, 0)
    finally:
        session.close()

    # A new installation still creates the ORM schema, including the field.
    assert "onboarding_completed" in {column["name"] for column in inspect(engine).get_columns("users")}
