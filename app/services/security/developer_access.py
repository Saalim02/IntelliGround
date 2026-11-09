"""Short-lived, server-validated developer access sessions."""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.auth_models import User
from app.services.security.auth_dependencies import get_current_user
from app.services.security.auth_service import get_jwt_secret
from app.services.security.rate_limiter import RateLimiter


DEVELOPER_ACCESS_COOKIE = "intelliground_developer_access"
DEVELOPER_ACCESS_MAX_AGE_SECONDS = 30 * 60
DEVELOPER_ACCESS_RATE_LIMITER = RateLimiter(max_requests=5, window_seconds=60)


def developer_cookie_secure() -> bool:
    return os.getenv("COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}


def create_developer_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user.id),
            "type": "developer_access",
            "iat": now,
            "exp": now + timedelta(seconds=DEVELOPER_ACCESS_MAX_AGE_SECONDS),
            "token_version": int(user.token_version),
        },
        get_jwt_secret(),
        algorithm="HS256",
    )


def _decode_developer_access_token(token: str) -> tuple[int, int]:
    payload = jwt.decode(token, get_jwt_secret(), algorithms=["HS256"])
    if payload.get("type") != "developer_access":
        raise ValueError("Invalid developer token type")
    return int(payload["sub"]), int(payload["token_version"])


def get_developer_user(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    token = request.cookies.get(DEVELOPER_ACCESS_COOKIE)
    if not token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Developer access required.")
    try:
        user_id, token_version = _decode_developer_access_token(token)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Developer access required.") from exc
    if user_id != current_user.id or token_version != current_user.token_version:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Developer access required.")
    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Developer access required.")
    return user
