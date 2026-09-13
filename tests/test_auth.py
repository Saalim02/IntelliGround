import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.auth_models import PasswordResetToken, User
from app.services.security.auth_rate_limit import (
    GOOGLE_LOGIN_RATE_LIMITER,
    PASSWORD_RESET_CONFIRM_RATE_LIMITER,
    REGISTER_RATE_LIMITER,
)
from app.services.security.auth_service import (
    create_access_token,
    create_password_reset_token,
    create_user,
)
from app.services.security.google_oauth import verify_oauth_state


client = TestClient(app)


TEST_EMAIL = "password_reset_test@example.com"
TEST_PASSWORD = "OldPassword123!"


def get_or_create_local_user(
    email: str = TEST_EMAIL,
    password: str = TEST_PASSWORD,
):
    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        if user is None:
            user = create_user(
                db=db,
                email=email,
                password=password,
            )

        return user.id

    finally:
        db.close()


def test_forgot_password_returns_same_response_for_existing_and_missing_email():
    existing_user_id = get_or_create_local_user()

    existing_response = client.post(
        "/api/auth/forgot-password",
        json={
            "email": TEST_EMAIL,
        },
    )

    missing_response = client.post(
        "/api/auth/forgot-password",
        json={
            "email": "does-not-exist@example.com",
        },
    )

    assert existing_user_id is not None

    assert existing_response.status_code == 200
    assert missing_response.status_code == 200

    assert existing_response.json() == missing_response.json()

    assert "reset" in existing_response.json()["message"].lower()


def test_password_reset_token_is_stored_as_hash():
    user_id = get_or_create_local_user(
        email="password_hash_storage_test@example.com",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        raw_token = create_password_reset_token(
            db=db,
            user=user,
        )

        stored = (
            db.query(PasswordResetToken)
            .filter(
                PasswordResetToken.user_id == user.id
            )
            .order_by(PasswordResetToken.id.desc())
            .first()
        )

        assert stored is not None
        assert stored.token_hash != raw_token
        assert len(stored.token_hash) == 64

    finally:
        db.close()


def test_valid_password_reset_token_changes_password():
    user_id = get_or_create_local_user(
        email="password_change_test@example.com",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        raw_token = create_password_reset_token(
            db=db,
            user=user,
        )

    finally:
        db.close()

    response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_token,
            "new_password": "NewPassword456!",
        },
    )

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Password has been reset successfully."
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        assert user is not None
        assert user.password_hash is not None

        from app.services.security.auth_service import (
            verify_password,
        )

        assert verify_password(
            "NewPassword456!",
            user.password_hash,
        )

    finally:
        db.close()


def test_password_reset_token_can_only_be_used_once():
    user_id = get_or_create_local_user(
        email="single_use_reset_test@example.com",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        raw_token = create_password_reset_token(
            db=db,
            user=user,
        )

    finally:
        db.close()

    first_response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_token,
            "new_password": "FirstNewPassword123!",
        },
    )

    second_response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_token,
            "new_password": "SecondNewPassword123!",
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 400
    assert second_response.json()["detail"] == (
        "Invalid or expired password reset token."
    )


def test_invalid_password_reset_token_is_rejected():
    response = client.post(
        "/api/auth/reset-password",
        json={
            "token": "this-is-an-invalid-reset-token-123456",
            "new_password": "NewPassword789!",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid or expired password reset token."
    )


def test_expired_password_reset_token_is_rejected():
    user_id = get_or_create_local_user(
        email="expired_reset_test@example.com",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        raw_token = create_password_reset_token(
            db=db,
            user=user,
            expires_minutes=1,
        )

        reset_record = (
            db.query(PasswordResetToken)
            .filter(
                PasswordResetToken.user_id == user.id
            )
            .order_by(PasswordResetToken.id.desc())
            .first()
        )

        reset_record.expires_at = (
            datetime.now(timezone.utc)
            - timedelta(minutes=1)
        )

        db.commit()

    finally:
        db.close()

    response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_token,
            "new_password": "ExpiredPassword123!",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid or expired password reset token."
    )


def test_weak_password_is_rejected():
    user_id = get_or_create_local_user(
        email="weak_password_reset_test@example.com",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        raw_token = create_password_reset_token(
            db=db,
            user=user,
        )

    finally:
        db.close()

    response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_token,
            "new_password": "short",
        },
    )

    assert response.status_code == 422


def test_google_only_account_does_not_create_password_reset_token():
    email = "google_only_reset_test@example.com"

    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        if user is None:
            user = User(
                email=email,
                password_hash=None,
                auth_provider="google",
                google_sub="google-reset-test-sub",
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

    finally:
        db.close()

    response = client.post(
        "/api/auth/forgot-password",
        json={
            "email": email,
        },
    )

    assert response.status_code == 200

    db = SessionLocal()

    try:
        token_count = (
            db.query(PasswordResetToken)
            .filter(
                PasswordResetToken.user_id == user.id
            )
            .count()
        )

        assert token_count == 0

    finally:
        db.close()


def test_password_reset_revokes_existing_access_token():
    email = "jwt_revocation_reset_test@example.com"
    user_id = get_or_create_local_user(
        email=email,
        password="OldPassword123!",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        assert user is not None

        old_token_version = user.token_version

        access_token = create_access_token(
            user.id,
            token_version=user.token_version,
        )

        raw_reset_token = create_password_reset_token(
            db=db,
            user=user,
        )

    finally:
        db.close()

    PASSWORD_RESET_CONFIRM_RATE_LIMITER.reset()

    reset_response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_reset_token,
            "new_password": "RevokedPassword456!",
        },
    )

    assert reset_response.status_code == 200

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        assert user is not None
        assert user.token_version == old_token_version + 1

    finally:
        db.close()

    protected_response = client.get(
        "/api/auth/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )

    assert protected_response.status_code == 401
    assert protected_response.json()["detail"] == (
        "Authentication token has been revoked."
    )


def test_developer_access_grants_with_correct_password(monkeypatch):
    user_id = get_or_create_local_user(
        email="developer_access_granted@example.com",
        password="DeveloperPassword123!",
    )

    access_token = create_access_token(
        user_id,
        token_version=0,
    )

    monkeypatch.setattr(
        "app.api.routes.auth.DEVELOPER_ACCESS_PASSWORD",
        "20042002",
    )

    response = client.post(
        "/api/auth/developer-access",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "password": "20042002",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "authorized": True,
        "message": "Developer access granted.",
    }


def test_developer_access_rejects_wrong_password(monkeypatch):
    user_id = get_or_create_local_user(
        email="developer_access_denied@example.com",
        password="DeveloperPassword123!",
    )

    access_token = create_access_token(
        user_id,
        token_version=0,
    )

    monkeypatch.setattr(
        "app.api.routes.auth.DEVELOPER_ACCESS_PASSWORD",
        "20042002",
    )

    response = client.post(
        "/api/auth/developer-access",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "password": "wrong-password",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Developer access denied."
    )


def test_developer_access_requires_authentication(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.auth.DEVELOPER_ACCESS_PASSWORD",
        "20042002",
    )

    response = client.post(
        "/api/auth/developer-access",
        json={
            "password": "20042002",
        },
    )

    assert response.status_code == 401



def test_weak_password_does_not_consume_reset_token():
    email = "weak_password_reset_test@example.com"

    user_id = get_or_create_local_user(
        email=email,
        password="OldPassword123!",
    )

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.id == user_id).first()

        assert user is not None

        raw_reset_token = create_password_reset_token(
            db=db,
            user=user,
        )

    finally:
        db.close()

    weak_response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_reset_token,
            "new_password": "short",
        },
    )

    assert weak_response.status_code == 422
    assert "at least 8 characters" in (
        weak_response.json()["detail"][0]["msg"]
    )

    # Reset the test limiter so we can prove that the same
    # reset token is still usable after the rejected password.
    PASSWORD_RESET_CONFIRM_RATE_LIMITER.reset()

    # The token should still be usable because password
    # validation happens before token consumption.
    valid_response = client.post(
        "/api/auth/reset-password",
        json={
            "token": raw_reset_token,
            "new_password": "ValidPassword456!",
        },
    )

    assert valid_response.status_code == 200
    assert valid_response.json()["message"] == (
        "Password has been reset successfully."
    )




def test_register_uses_register_rate_limiter(monkeypatch):
    calls = []

    def fake_check(limiter, client_id):
        calls.append(limiter)

    monkeypatch.setattr(
        "app.api.routes.auth.check_auth_rate_limit",
        fake_check,
    )
    import uuid
    from datetime import datetime, timedelta, timezone  
    email = (
        f"register_rate_test_{uuid.uuid4().hex}"
        "@example.com"
    )

    response = client.post(
        "/api/auth/register",
        json={
            "first_name": "Rate",
            "last_name": "Test",
            "email": email,
            "password": "RegisterPassword123!",
        },
    )

    assert response.status_code == 201
    assert calls
    assert calls[0] is REGISTER_RATE_LIMITER


def test_registration_requires_first_and_last_name():
    response = client.post(
        "/api/auth/register",
        json={
            "email": f"missing-name-{uuid.uuid4().hex}@example.com",
            "password": "RegisterPassword123!",
        },
    )

    assert response.status_code == 422


def test_registration_persists_names_and_me_returns_profile():
    email = f"profile-{uuid.uuid4().hex}@example.com"
    response = client.post(
        "/api/auth/register",
        json={
            "first_name": "  Profile ",
            "last_name": " User ",
            "email": email,
            "password": "RegisterPassword123!",
        },
    )

    assert response.status_code == 201
    assert response.json()["first_name"] == "Profile"
    assert response.json()["last_name"] == "User"
    assert response.json()["display_name"] == "Profile User"

    login_response = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "password": "RegisterPassword123!",
        },
    )
    assert login_response.status_code == 200
    assert login_response.json()["first_name"] == "Profile"

    me_response = client.get("/api/auth/me")
    assert me_response.status_code == 200
    assert me_response.json()["display_name"] == "Profile User"


def test_google_profile_handles_missing_family_name():
    from app.services.security.auth_service import update_google_user_profile

    user = User(
        email="single-name@example.com",
        auth_provider="google",
        google_sub="single-name-sub",
        is_active=True,
        token_version=0,
    )

    update_google_user_profile(
        db=None,
        user=user,
        first_name="Single",
        last_name=None,
    )

    assert user.first_name == "Single"
    assert user.last_name is None


def test_google_login_uses_google_rate_limiter(monkeypatch):
    calls = []

    def fake_check(limiter, client_id):
        calls.append(limiter)

    monkeypatch.setattr(
        "app.api.routes.auth.check_auth_rate_limit",
        fake_check,
    )

    monkeypatch.setattr(
        "app.api.routes.auth.build_google_authorization_url",
        lambda state: "https://accounts.google.com/test",
    )

    response = client.get(
        "/api/auth/google/login",
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert calls
    assert calls[0] is GOOGLE_LOGIN_RATE_LIMITER


def test_oauth_state_requires_a_present_matching_value():
    assert verify_oauth_state("state-value", "state-value") is True
    assert verify_oauth_state("", "state-value") is False
    assert verify_oauth_state("state-value", "") is False
    assert verify_oauth_state("different-state", "state-value") is False


def test_google_state_cookie_is_scoped_to_callback(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.auth.build_google_authorization_url",
        lambda state: "https://accounts.google.com/test",
    )

    response = client.get(
        "/api/auth/google/login",
        follow_redirects=False,
    )

    assert response.status_code == 307
    set_cookie = response.headers["set-cookie"]
    assert "Path=/api/auth/google/callback" in set_cookie
    assert "Path=/api/auth/google;" not in set_cookie


def test_missing_or_mismatched_oauth_state_is_rejected():
    client.cookies.clear()

    missing = client.get(
        "/api/auth/google/callback?code=test-code",
        follow_redirects=False,
    )
    mismatched = client.get(
        "/api/auth/google/callback?state=received&code=test-code",
        follow_redirects=False,
    )

    assert missing.status_code == 400
    assert mismatched.status_code == 400
    assert missing.json()["detail"] == "Invalid OAuth state."
    assert mismatched.json()["detail"] == "Invalid OAuth state."


def test_successful_google_callback_consumes_state_cookie(monkeypatch):
    from app.api.routes import auth as auth_routes

    client.cookies.clear()
    test_user = User(
        id=999998,
        email="oauth_single_use@example.com",
        auth_provider="google",
        google_sub="oauth-single-use-sub",
        is_active=True,
        token_version=0,
    )

    monkeypatch.setattr(
        "app.api.routes.auth.build_google_authorization_url",
        lambda state: "https://accounts.google.com/test",
    )
    login_response = client.get(
        "/api/auth/google/login",
        follow_redirects=False,
    )
    state = login_response.cookies.get("google_oauth_state")
    assert state

    monkeypatch.setattr(
        auth_routes,
        "get_google_config",
        lambda: {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "redirect_uri": "http://localhost:8000/api/auth/google/callback",
        },
    )

    async def fake_exchange_code_for_tokens(code):
        return {"id_token": "test-id-token"}

    async def fake_verify_google_id_token(id_token, client_id):
        return {
            "email": test_user.email,
            "sub": test_user.google_sub,
            "email_verified": True,
        }

    monkeypatch.setattr(auth_routes, "exchange_code_for_tokens", fake_exchange_code_for_tokens)
    monkeypatch.setattr(auth_routes, "verify_google_id_token", fake_verify_google_id_token)
    monkeypatch.setattr(auth_routes, "get_or_create_google_user", lambda db, email, google_sub: test_user)

    callback_url = f"/api/auth/google/callback?state={state}&code=test-code"
    first = client.get(callback_url, follow_redirects=False)
    replay = client.get(callback_url, follow_redirects=False)

    assert first.status_code == 303
    assert "google_oauth_state=;" in first.headers["set-cookie"]
    assert replay.status_code == 400
    assert replay.json()["detail"] == "Invalid OAuth state."


def test_google_callback_redirects_to_configured_frontend(monkeypatch):
    from app.api.routes import auth as auth_routes
    from app.core.config import FRONTEND_URL

    test_user = User(
        id=999999,
        email="oauth_redirect_test@example.com",
        auth_provider="google",
        google_sub="oauth-redirect-test-sub",
        is_active=True,
        token_version=0,
    )

    monkeypatch.setattr(
        auth_routes,
        "get_google_config",
        lambda: {
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "redirect_uri": (
                "http://localhost:8000/api/auth/google/callback"
            ),
        },
    )

    async def fake_exchange_code_for_tokens(code):
        return {"id_token": "test-id-token"}

    async def fake_verify_google_id_token(
        id_token,
        client_id,
    ):
        return {
            "email": test_user.email,
            "sub": test_user.google_sub,
            "email_verified": True,
            "given_name": "OAuth",
            "family_name": "Redirect",
        }

    monkeypatch.setattr(
        auth_routes,
        "exchange_code_for_tokens",
        fake_exchange_code_for_tokens,
    )
    monkeypatch.setattr(
        auth_routes,
        "verify_google_id_token",
        fake_verify_google_id_token,
    )
    monkeypatch.setattr(
        auth_routes,
        "get_or_create_google_user",
        lambda db, email, google_sub: test_user,
    )

    monkeypatch.setattr(
        auth_routes,
        "verify_oauth_state",
        lambda received_state, stored_state: True,
    )

    response = client.get(
        "/api/auth/google/callback"
        "?state=test-state&code=test-code",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == FRONTEND_URL
    assert test_user.first_name == "OAuth"
    assert test_user.last_name == "Redirect"
