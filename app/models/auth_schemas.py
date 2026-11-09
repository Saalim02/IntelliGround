from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Name must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Name must not be blank.")
        return normalized


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class DeveloperAccessRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    auth_provider: str
    is_active: bool
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    onboarding_completed: bool = False


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=8)
    new_password: str = Field(min_length=8, max_length=128)


class GoogleCallbackRequest(BaseModel):
    code: str
