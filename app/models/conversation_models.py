from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="voice")
    query_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    response_language: Mapped[str] = mapped_column(String(32), nullable=False)
    speech_input_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    speech_output_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    evaluation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
