import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import query as query_route
from app.core.database import Base
from app.models.auth_models import User
from app.models.conversation_models import ConversationTurn
from app.models.query_models import QueryRequest
from app.services.conversation_service import (
    add_turn_if_new,
    create_conversation,
    get_owned_conversation,
    validate_conversation_document,
)


@pytest.fixture
def db_user():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    user = User(email="phase5a@example.com", is_active=True)
    session.add(user)
    session.commit()
    session.refresh(user)
    try:
        yield session, user
    finally:
        session.close()
        engine.dispose()


def active_document(doc_id="doc-a", filename="a.pdf"):
    return {
        "active_doc_id": doc_id,
        "active_filename": filename,
        "response_language": "en",
    }


def test_text_route_creates_and_continues_owned_conversation(monkeypatch, db_user):
    db, user = db_user
    captured = []
    monkeypatch.setattr(query_route, "get_active_document", lambda _user_id: active_document())

    def fake_query(request, current_user, **_kwargs):
        captured.append((request.question, request.conversation_id, request.conversation_context, current_user.id))
        return {
            "status": "success", "active_document": "a.pdf", "question": request.question,
            "question_type": "DOCUMENT", "answer": f"Answer to {request.question}",
            "confidence": "high", "best_score": 1.0, "average_score": 1.0,
            "evidence": [{"page_number": 2, "chunk_id": 4, "snippet": "evidence"}],
            "retrieved_chunks": [],
            "evaluation": {"evaluation_id": f"eval-{len(captured)}"},
            "message": "ok",
        }

    monkeypatch.setattr(query_route, "run_document_query", fake_query)
    first = query_route.ask_question(
        QueryRequest(question="What is topic one?", turn_id="turn-1"), user, db,
    )
    conversation_id = first["conversation_id"]
    assert conversation_id

    second = query_route.ask_question(
        QueryRequest(question="Tell me more.", conversation_id=conversation_id, turn_id="turn-2"), user, db,
    )
    assert second["conversation_id"] == conversation_id
    assert captured[1][1] == conversation_id
    assert "What is topic one?" in captured[1][2]
    assert "Answer to What is topic one?" in captured[1][2]

    turns = db.query(ConversationTurn).filter(ConversationTurn.conversation_id == conversation_id).all()
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert json.loads(turns[1].evidence_json)[0]["page_number"] == 2
    assert turns[1].evaluation_id == "eval-1"


def test_text_turn_id_prevents_duplicate_persistence(db_user):
    db, user = db_user
    conversation = create_conversation(
        db, user, document=active_document(), mode="text", response_language="en",
        speech_input_language=None, speech_output_language=None,
    )
    first = add_turn_if_new(
        db, conversation, transcript="Question", answer=None, evidence=[],
        provider_request_id="client-turn-1", role="user",
    )
    duplicate = add_turn_if_new(
        db, conversation, transcript="Question", answer=None, evidence=[],
        provider_request_id="client-turn-1", role="user",
    )
    assert first is not None
    assert duplicate is None
    assert db.query(ConversationTurn).filter(ConversationTurn.conversation_id == conversation.id).count() == 1


def test_conversation_ownership_and_document_isolation(db_user):
    db, user = db_user
    owner_conversation = create_conversation(
        db, user, document=active_document(), mode="text", response_language="en",
        speech_input_language=None, speech_output_language=None,
    )
    other = User(email="phase5a-other@example.com", is_active=True)
    db.add(other)
    db.commit()

    with pytest.raises(HTTPException) as missing:
        get_owned_conversation(db, other, owner_conversation.id)
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as changed:
        validate_conversation_document(owner_conversation, active_document("doc-b", "b.pdf"))
    assert changed.value.status_code == 409


def test_existing_client_without_conversation_remains_compatible(monkeypatch, db_user):
    db, user = db_user
    monkeypatch.setattr(query_route, "get_active_document", lambda _user_id: None)
    monkeypatch.setattr(
        query_route,
        "run_document_query",
        lambda request, current_user, **_kwargs: {
            "status": "error", "active_document": None, "question": request.question,
            "question_type": "UNANSWERABLE", "answer": None, "confidence": "low",
            "best_score": None, "average_score": None, "evidence": [],
            "retrieved_chunks": [], "evaluation": None, "message": "No active PDF uploaded.",
        },
    )
    result = query_route.ask_question(QueryRequest(question="Hello"), user, db)
    assert result["conversation_id"] is None
