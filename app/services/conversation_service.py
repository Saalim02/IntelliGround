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
    response_length: str = "medium",
    title: str | None = None,
) -> Conversation:
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        document_id=document["active_doc_id"],
        document_name=document["active_filename"],
        document_path=document.get("source_path"),
        document_sha256=document.get("sha256"),
        document_input_language=document.get("input_language"),
        document_evidence_language=document.get("evidence_language"),
        document_total_pages=document.get("total_pages"),
        title=title,
        mode=mode,
        response_language=response_language,
        response_length=response_length,
        speech_input_language=speech_input_language,
        speech_output_language=speech_output_language,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def validate_document_reupload(
    conversation: Conversation,
    *,
    filename: str,
    sha256: str | None,
    confirmed_replacement: bool = False,
) -> tuple[bool, str]:
    """Validate a PDF before rebuilding a historical conversation's index."""
    if conversation.document_name == filename and conversation.document_sha256 and conversation.document_sha256 == sha256:
        return True, "verified"
    if confirmed_replacement:
        return True, "explicit replacement confirmed"
    return False, "The PDF identity could not be verified. Confirm that this upload should replace the unavailable document."


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
        "document_path": conversation.document_path,
        "document_sha256": conversation.document_sha256,
        "title": conversation.title or conversation.document_name,
        "mode": conversation.mode,
        "response_language": conversation.response_language,
        "response_length": conversation.response_length or "medium",
        "speech_input_language": conversation.speech_input_language,
        "speech_output_language": conversation.speech_output_language,
        "state": conversation.state,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


def _json_load(value: str | None, fallback):
    try:
        parsed = json.loads(value or "")
        return parsed if parsed is not None else fallback
    except (TypeError, ValueError):
        return fallback


def serialize_conversation_detail(db: Session, conversation: Conversation) -> dict:
    turns = (
        db.query(ConversationTurn)
        .filter(ConversationTurn.conversation_id == conversation.id, ConversationTurn.user_id == conversation.user_id)
        .order_by(ConversationTurn.created_at.asc(), ConversationTurn.id.asc())
        .all()
    )
    messages = []
    for turn in turns:
        text = turn.transcript if turn.role == "user" else turn.answer
        if text:
            messages.append({"role": turn.role, "text": text, "evidence": _json_load(turn.evidence_json, []), "answer_id": turn.id if turn.role == "assistant" else None, "time": turn.created_at.isoformat()})
    return {
        **serialize_conversation(conversation),
        "messages": messages,
        "intelligence": {
            "summary": _json_load(conversation.summary_json, None),
            "topics": _json_load(conversation.topics_json, None),
            "questions": _json_load(conversation.questions_json, None),
        },
        "document": {
            "active_doc_id": conversation.document_id,
            "active_filename": conversation.document_name,
            "source_path": conversation.document_path,
            "sha256": conversation.document_sha256,
            "input_language": conversation.document_input_language,
            "evidence_language": conversation.document_evidence_language,
            "total_pages": conversation.document_total_pages,
            "response_language": conversation.response_language,
        },
    }


def save_conversation_panel(db: Session, user: User, conversation_id: str, *, kind: str, result: dict, response_language: str, response_length: str, document_id: str | None = None) -> Conversation:
    conversation = get_owned_conversation(db, user, conversation_id)
    if document_id and conversation.document_id != document_id:
        raise HTTPException(status_code=409, detail="Conversation is attached to a different document.")
    conversation.response_language = response_language
    conversation.response_length = response_length
    field = {"summary": "summary_json", "topics": "topics_json", "questions": "questions_json"}.get(kind)
    if field is None:
        raise HTTPException(status_code=400, detail="Unknown intelligence panel.")
    setattr(conversation, field, json.dumps(result, ensure_ascii=False))
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def update_conversation_preferences(db: Session, user: User, conversation_id: str, *, title: str | None = None, response_language: str | None = None, response_length: str | None = None) -> Conversation:
    conversation = get_owned_conversation(db, user, conversation_id)
    if title is not None:
        conversation.title = title[:512]
    if response_language is not None:
        conversation.response_language = response_language
    if response_length is not None:
        conversation.response_length = response_length
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation
