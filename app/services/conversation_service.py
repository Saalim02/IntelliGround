"""User-scoped conversation and voice-session persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.auth_models import User
from app.models.conversation_models import Conversation, ConversationTurn


NO_DOCUMENT_ID = "__no_document__"
NO_DOCUMENT_NAME = "No active PDF"


def create_conversation(
    db: Session,
    user: User,
    *,
    document: dict,
    mode: str,
    response_language: str,
    speech_input_language: str | None,
    speech_output_language: str | None,
) -> Conversation:
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        document_id=document["active_doc_id"],
        document_name=document["active_filename"],
        mode=mode,
        response_language=response_language,
        speech_input_language=speech_input_language,
        speech_output_language=speech_output_language,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def get_owned_conversation(db: Session, user: User, conversation_id: str) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


def validate_conversation_document(
    conversation: Conversation,
    active_document: dict | None,
) -> dict:
    """Require a conversation to remain attached to the active document."""
    if conversation.document_id == NO_DOCUMENT_ID:
        if active_document:
            raise HTTPException(
                status_code=409,
                detail="This conversation has no active document. Start a new conversation for the uploaded PDF.",
            )
        return {"active_doc_id": NO_DOCUMENT_ID, "active_filename": NO_DOCUMENT_NAME}
    if not active_document:
        raise HTTPException(
            status_code=409,
            detail="The conversation document is no longer active. Start a new conversation.",
        )
    conversation_document_name = getattr(conversation, "document_name", active_document.get("active_filename"))
    if (
        conversation.document_id != active_document.get("active_doc_id")
        or conversation_document_name != active_document.get("active_filename")
    ):
        raise HTTPException(
            status_code=409,
            detail="This conversation belongs to a different active document. Start a new conversation.",
        )
    return active_document


def turn_exists(
    db: Session,
    conversation: Conversation,
    *,
    provider_request_id: str | None,
    role: str,
) -> bool:
    """Check the existing request-id seam before persisting a retried turn."""
    if not provider_request_id:
        return False
    return (
        db.query(ConversationTurn)
        .filter(
            ConversationTurn.conversation_id == conversation.id,
            ConversationTurn.user_id == conversation.user_id,
            ConversationTurn.provider_request_id == provider_request_id,
            ConversationTurn.role == role,
        )
        .first()
        is not None
    )


def add_turn_if_new(
    db: Session,
    conversation: Conversation,
    *,
    transcript: str | None,
    answer: str | None,
    evidence: list,
    evaluation_id: str | None = None,
    provider_request_id: str | None = None,
    role: str | None = None,
) -> ConversationTurn | None:
    resolved_role = role or ("user" if transcript and not answer else "assistant")
    if turn_exists(
        db,
        conversation,
        provider_request_id=provider_request_id,
        role=resolved_role,
    ):
        return None
    return add_turn(
        db,
        conversation,
        transcript=transcript,
        answer=answer,
        evidence=evidence,
        evaluation_id=evaluation_id,
        provider_request_id=provider_request_id,
        role=resolved_role,
    )


def add_turn(
    db: Session,
    conversation: Conversation,
    *,
    transcript: str | None,
    answer: str | None,
    evidence: list,
    evaluation_id: str | None = None,
    provider_request_id: str | None = None,
    role: str | None = None,
) -> ConversationTurn:
    turn = ConversationTurn(
        conversation_id=conversation.id,
        user_id=conversation.user_id,
        role=role or ("user" if transcript and not answer else "assistant"),
        transcript=transcript,
        answer=answer,
        evidence_json=json.dumps(evidence, ensure_ascii=False),
        evaluation_id=evaluation_id,
        provider_request_id=provider_request_id,
    )
    conversation.updated_at = datetime.now(timezone.utc)
    db.add(turn)
    db.add(conversation)
    db.commit()
    db.refresh(turn)
    return turn


def recent_context(db: Session, conversation: Conversation, limit: int = 6) -> str:
    turns = (
        db.query(ConversationTurn)
        .filter(
            ConversationTurn.conversation_id == conversation.id,
            ConversationTurn.user_id == conversation.user_id,
        )
        .order_by(ConversationTurn.created_at.desc())
        .limit(limit)
        .all()
    )
    lines = []
    for turn in reversed(turns):
        if turn.transcript:
            lines.append(f"User: {turn.transcript}")
        if turn.answer:
            lines.append(f"Assistant: {turn.answer}")
            try:
                evidence = json.loads(turn.evidence_json or "[]")
            except (TypeError, ValueError):
                evidence = []
            pages = sorted({
                int(item["page_number"])
                for item in evidence
                if isinstance(item, dict) and str(item.get("page_number", "")).isdigit()
            })
            if pages:
                lines.append(f"Assistant evidence pages: {', '.join(map(str, pages))}")
    return "\n".join(lines)


def serialize_conversation(conversation: Conversation) -> dict:
    return {
        "id": conversation.id,
        "document_id": conversation.document_id,
        "document_name": conversation.document_name,
        "mode": conversation.mode,
        "response_language": conversation.response_language,
        "speech_input_language": conversation.speech_input_language,
        "speech_output_language": conversation.speech_output_language,
        "state": conversation.state,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }
