from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.auth_models import User
from app.models.conversation_models import Conversation, ConversationTurn
from app.services.conversation_service import (
    add_turn,
    create_conversation,
    get_owned_conversation,
    save_conversation_panel,
    serialize_conversation_detail,
    validate_document_reupload,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_panels_and_messages_are_saved_independently_per_conversation():
    db = _session()
    first = User(email="first@example.com", is_active=True)
    second = User(email="second@example.com", is_active=True)
    db.add_all([first, second])
    db.commit()
    document = {"active_doc_id": "doc-1", "active_filename": "source.pdf"}
    first_chat = create_conversation(db, first, document=document, mode="text", response_language="ta", speech_input_language=None, speech_output_language=None, response_length="small", title="Tamil document")
    second_chat = create_conversation(db, second, document=document, mode="text", response_language="en", speech_input_language=None, speech_output_language=None, response_length="large", title="Other chat")

    save_conversation_panel(db, first, first_chat.id, kind="summary", result={"short_summary": "தமிழ் சுருக்கம்"}, response_language="ta", response_length="small")
    save_conversation_panel(db, first, first_chat.id, kind="topics", result={"topics": [{"title": "Topic", "evidence": [{"page_number": 1, "chunk_id": 1, "text": "source"}]}]}, response_language="ta", response_length="small")
    add_turn(db, first_chat, transcript="ஒரு கேள்வி", answer="ஒரு பதில்", evidence=[{"page_number": 1, "snippet": "source"}], role="assistant")

    detail = serialize_conversation_detail(db, get_owned_conversation(db, first, first_chat.id))
    assert detail["title"] == "Tamil document"
    assert detail["response_length"] == "small"
    assert detail["intelligence"]["summary"]["short_summary"] == "தமிழ் சுருக்கம்"
    assert detail["intelligence"]["topics"]["topics"][0]["evidence"][0]["page_number"] == 1
    assert detail["intelligence"]["questions"] is None
    assert detail["messages"][0]["role"] == "assistant"

    other = serialize_conversation_detail(db, get_owned_conversation(db, second, second_chat.id))
    assert other["intelligence"] == {"summary": None, "topics": None, "questions": None}
    assert other["messages"] == []


def test_conversation_panel_cannot_be_written_by_another_user():
    db = _session()
    owner = User(email="owner@example.com", is_active=True)
    other = User(email="other@example.com", is_active=True)
    db.add_all([owner, other])
    db.commit()
    chat = create_conversation(db, owner, document={"active_doc_id": "doc-1", "active_filename": "source.pdf"}, mode="text", response_language="en", speech_input_language=None, speech_output_language=None)

    try:
        save_conversation_panel(db, other, chat.id, kind="summary", result={"short_summary": "not allowed"}, response_language="en", response_length="medium")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("A conversation panel was written by a different user")


def test_conversation_detail_keeps_document_association_for_reactivation():
    db = _session()
    owner = User(email="reactivate@example.com", is_active=True)
    db.add(owner)
    db.commit()
    conversation = create_conversation(
        db, owner,
        document={"active_doc_id": "doc-b", "active_filename": "b.pdf", "source_path": "/tmp/b.pdf", "evidence_language": "en", "total_pages": 2},
        mode="text", response_language="en", speech_input_language=None, speech_output_language=None,
    )
    detail = serialize_conversation_detail(db, conversation)
    assert detail["document"]["active_doc_id"] == "doc-b"
    assert detail["document"]["source_path"] == "/tmp/b.pdf"


def test_reupload_requires_digest_match_or_explicit_replacement():
    db = _session()
    owner = User(email="reupload@example.com", is_active=True)
    db.add(owner)
    db.commit()
    conversation = create_conversation(
        db, owner,
        document={"active_doc_id": "doc-a", "active_filename": "a.pdf", "sha256": "known"},
        mode="text", response_language="en", speech_input_language=None, speech_output_language=None,
    )
    assert validate_document_reupload(conversation, filename="a.pdf", sha256="known") == (True, "verified")
    assert validate_document_reupload(conversation, filename="other.pdf", sha256="wrong")[0] is False
    assert validate_document_reupload(conversation, filename="other.pdf", sha256="wrong", confirmed_replacement=True)[0] is True
