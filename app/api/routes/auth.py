import os
import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from app.core.config import (
    DEVELOPER_ACCESS_PASSWORD,
    FRONTEND_URL,
)

from app.core.database import get_db
from app.models.auth_models import User
from app.models.auth_schemas import (
    DeveloperAccessRequest,
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    UserResponse,
)
from app.services.security.audit_logger import log_security_event
from app.services.security.auth_dependencies import get_current_user
from app.services.security.auth_rate_limit import (
    GOOGLE_LOGIN_RATE_LIMITER,
    LOGIN_RATE_LIMITER,
    PASSWORD_RESET_CONFIRM_RATE_LIMITER,
    PASSWORD_RESET_REQUEST_RATE_LIMITER,
    REGISTER_RATE_LIMITER,
    check_auth_rate_limit,
)
from app.services.security.auth_service import (
    authenticate_user,
    consume_password_reset_token,
    create_access_token,
    create_password_reset_token,
    create_user,
    get_or_create_google_user,
    hash_password,
    validate_password,
    update_google_user_profile,
)
from app.services.security.google_oauth import (
    GOOGLE_AUTH_COOKIE_MAX_AGE_SECONDS,
    GOOGLE_AUTH_COOKIE_NAME,
    GOOGLE_STATE_COOKIE_NAME,
    GOOGLE_STATE_MAX_AGE_SECONDS,
    build_google_authorization_url,
    exchange_code_for_tokens,
    generate_oauth_state,
    get_google_config,
    verify_google_id_token,
    verify_oauth_state,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

AUTH_COOKIE_NAME = "access_token"
AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24
SESSION_HINT_COOKIE_NAME = "intelliground_session"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes"}


def to_user_response(user: User) -> UserResponse:
    first_name = (user.first_name or "").strip() or None
    last_name = (user.last_name or "").strip() or None
    display_name = " ".join(
        name for name in (first_name, last_name) if name
    ) or None
    return UserResponse(
        id=user.id,
        email=user.email,
        auth_provider=user.auth_provider,
        is_active=user.is_active,
        first_name=first_name,
        last_name=last_name,
        display_name=display_name,
    )


def set_browser_session(response: Response, access_token: str) -> None:
    """Set the browser session without exposing the JWT to frontend code."""
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        max_age=AUTH_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    # This contains no credential; it only lets the SPA decide whether it
    # should restore a session on refresh without probing /auth/me on landing.
    response.set_cookie(
        key=SESSION_HINT_COOKIE_NAME,
        value="1",
        max_age=AUTH_COOKIE_MAX_AGE_SECONDS,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: Request,
    payload: RegisterRequest,
    db: Session = Depends(get_db),
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    check_auth_rate_limit(
        REGISTER_RATE_LIMITER,
        client_host,
    )

    try:
        user = create_user(
            db=db,
            email=str(payload.email),
            password=payload.password,
            first_name=payload.first_name,
            last_name=payload.last_name,
        )
    except ValueError as exc:
        log_security_event(
            "registration_failed",
            endpoint="/api/auth/register",
            client_id=client_host,
            decision="BLOCK",
            reasons=[str(exc)],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    log_security_event(
        "registration_success",
        endpoint="/api/auth/register",
        client_id=client_host,
        decision="ALLOW",
    )

    return to_user_response(user)


@router.post(
    "/login",
    response_model=UserResponse,
)
def login(
    request: Request,
    payload: LoginRequest,
    db: Session = Depends(get_db),
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    check_auth_rate_limit(
        LOGIN_RATE_LIMITER,
        client_host,
    )

    user = authenticate_user(
        db=db,
        email=str(payload.email),
        password=payload.password,
    )

    if user is None:
        log_security_event(
            "login_failed",
            endpoint="/api/auth/login",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Invalid email or password."],
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        user.id,
        token_version=user.token_version,
    )

    log_security_event(
        "login_success",
        endpoint="/api/auth/login",
        client_id=client_host,
        decision="ALLOW",
    )

    response = Response(
        content=to_user_response(user).model_dump_json(),
        media_type="application/json",
    )
    set_browser_session(response, access_token)
    return response


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(key=SESSION_HINT_COOKIE_NAME, path="/")
    return {"message": "Signed out successfully."}

@router.post("/developer-access")
def developer_access(
    payload: DeveloperAccessRequest,
    user: User = Depends(get_current_user),
):
    if not DEVELOPER_ACCESS_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Developer access is not configured.",
        )

    if not secrets.compare_digest(
        payload.password,
        DEVELOPER_ACCESS_PASSWORD,
    ):
        log_security_event(
            "developer_access_failed",
            endpoint="/api/auth/developer-access",
            client_id=str(user.id),
            decision="BLOCK",
            reasons=["Invalid developer access password."],
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer access denied.",
        )

    log_security_event(
        "developer_access_granted",
        endpoint="/api/auth/developer-access",
        client_id=str(user.id),
        decision="ALLOW",
    )

    return {
        "authorized": True,
        "message": "Developer access granted.",
    }


@router.post(
    "/forgot-password",
)
def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    check_auth_rate_limit(
        PASSWORD_RESET_REQUEST_RATE_LIMITER,
        client_host,
    )

    normalized_email = str(payload.email).strip().lower()

    user = (
        db.query(User)
        .filter(User.email == normalized_email)
        .first()
    )

    # Always return the same response whether the account exists.
    # This prevents email/account enumeration.
    if user is not None and user.is_active and user.password_hash:
        create_password_reset_token(
            db=db,
            user=user,
        )

    log_security_event(
        "password_reset_requested",
        endpoint="/api/auth/forgot-password",
        client_id=client_host,
        decision="ALLOW",
    )

    return {
        "message": (
            "If an account exists for that email, "
            "a password reset link will be sent."
        )
    }


@router.post(
    "/reset-password",
)
def reset_password(
    request: Request,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    check_auth_rate_limit(
        PASSWORD_RESET_CONFIRM_RATE_LIMITER,
        client_host,
    )

    # Validate the new password before consuming the reset token.
    # This prevents a weak password from accidentally burning a valid token.
    try:
        validate_password(payload.new_password)
    except ValueError as exc:
        log_security_event(
            "password_reset_failed",
            endpoint="/api/auth/reset-password",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Invalid password supplied."],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    user = consume_password_reset_token(
        db=db,
        raw_token=payload.token,
    )

    if user is None:
        log_security_event(
            "password_reset_failed",
            endpoint="/api/auth/reset-password",
            client_id=client_host,
            decision="BLOCK",
            reasons=[
                "Invalid, expired, or already-used reset token."
            ],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired password reset token.",
        )

    try:
        user.password_hash = hash_password(
            payload.new_password
        )

        # Incrementing token_version immediately revokes every
        # access token issued before this password reset.
        user.token_version += 1

        # The reset-token consumption, password change, and JWT
        # invalidation are committed together.
        db.commit()

    except Exception:
        db.rollback()

        log_security_event(
            "password_reset_failed",
            endpoint="/api/auth/reset-password",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Password reset transaction failed."],
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset could not be completed.",
        )

    log_security_event(
        "password_reset_success",
        endpoint="/api/auth/reset-password",
        client_id=client_host,
        decision="ALLOW",
    )

    return {
        "message": "Password has been reset successfully."
    }


@router.get(
    "/google/login",
)
def google_login(
    request: Request,
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    check_auth_rate_limit(
        GOOGLE_LOGIN_RATE_LIMITER,    
        client_host,
    )

    try:
        state = generate_oauth_state()
        authorization_url = build_google_authorization_url(state)
    except RuntimeError as exc:
        log_security_event(
            "google_login_configuration_error",
            endpoint="/api/auth/google/login",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Google OAuth is not configured."],
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Sign-In is not configured.",
        ) from exc

    response = RedirectResponse(
        url=authorization_url,
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )

    response.set_cookie(
        key=GOOGLE_STATE_COOKIE_NAME,
        value=state,
        max_age=GOOGLE_STATE_MAX_AGE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/api/auth/google/callback",
    )

    log_security_event(
        "google_login_started",
        endpoint="/api/auth/google/login",
        client_id=client_host,
        decision="ALLOW",
    )

    return response

@router.get("/google/callback")
async def google_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    client_host = (
        request.client.host
        if request.client
        else "unknown"
    )

    state_cookie = request.cookies.get(
        GOOGLE_STATE_COOKIE_NAME
    )
    received_state = request.query_params.get("state")
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        log_security_event(
            "google_callback_failed",
            endpoint="/api/auth/google/callback",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Google authorization was denied or cancelled."],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Sign-In was cancelled or denied.",
        )

    if not verify_oauth_state(
        received_state,
        state_cookie,
    ):
        log_security_event(
            "google_callback_state_failed",
            endpoint="/api/auth/google/callback",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Invalid OAuth state."],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state.",
        )

    if not code:
        log_security_event(
            "google_callback_code_missing",
            endpoint="/api/auth/google/callback",
            client_id=client_host,
            decision="BLOCK",
            reasons=["Google authorization code is missing."],
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google authorization code is missing.",
        )

    try:
        config = get_google_config()

        token_response = await exchange_code_for_tokens(code)

        id_token = token_response.get("id_token")

        if not id_token:
            raise ValueError(
                "Google identity token is missing."
            )

        claims = await verify_google_id_token(
            id_token=id_token,
            client_id=config["client_id"],
        )

        user = get_or_create_google_user(
            db=db,
            email=claims["email"],
            google_sub=claims["sub"],
        )
        user = update_google_user_profile(
            db=db,
            user=user,
            first_name=claims.get("given_name"),
            last_name=claims.get("family_name"),
            full_name=claims.get("name"),
        )

        access_token = create_access_token(
        user.id,
        token_version=user.token_version,
    )

    except (ValueError, RuntimeError, KeyError) as exc:
        log_security_event(
            "google_callback_failed",
            endpoint="/api/auth/google/callback",
            client_id=client_host,
            decision="BLOCK",
            reasons=[str(exc)],
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Sign-In could not be completed.",
        ) from exc

    response = RedirectResponse(
        url=FRONTEND_URL,
        status_code=status.HTTP_303_SEE_OTHER,
    )

    response.set_cookie(
        key=GOOGLE_AUTH_COOKIE_NAME,
        value=access_token,
        max_age=GOOGLE_AUTH_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=SESSION_HINT_COOKIE_NAME,
        value="1",
        max_age=GOOGLE_AUTH_COOKIE_MAX_AGE_SECONDS,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )

    response.delete_cookie(
        key=GOOGLE_STATE_COOKIE_NAME,
        path="/api/auth/google/callback",
    )

    log_security_event(
        "google_login_success",
        endpoint="/api/auth/google/callback",
        client_id=client_host,
        decision="ALLOW",
    )

    return response

@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return to_user_response(current_user)
