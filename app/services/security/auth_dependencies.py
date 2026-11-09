from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import LOCAL_DEV_AUTH_BYPASS, LOCAL_DEV_EMAIL
from app.core.database import get_db
from app.models.auth_models import User
from app.services.security.auth_service import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


def get_local_development_user(db: Session) -> User:
    """Return the stable local user used when auth bypass is explicitly enabled."""
    user = db.query(User).filter(User.email == LOCAL_DEV_EMAIL).first()
    if user is None:
        user = User(
            email=LOCAL_DEV_EMAIL,
            first_name="Local",
            last_name="Developer",
            auth_provider="local-dev",
            is_active=True,
            onboarding_completed=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not user.is_active:
        user.is_active = True
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:
    if LOCAL_DEV_AUTH_BYPASS:
        return get_local_development_user(db)

    token = None

    # Prefer the explicit Authorization header when provided.
    if credentials is not None:
        if credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials

    # Browser sessions can use the HttpOnly authentication cookie.
    if token is None:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id, token_version = decode_access_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    if token_version != user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
