from typing import Literal

from pydantic import BaseModel, Field


class ConversationUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    response_language: str | None = None
    response_length: Literal["short", "medium", "large"] | None = None
