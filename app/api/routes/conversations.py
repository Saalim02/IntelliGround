from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.auth_models import User
from app.models.conversation_models import Conversation
from app.models.conversation_request_models import ConversationUpdateRequest
from app.services.conversation_service import get_owned_conversation, serialize_conversation, serialize_conversation_detail, update_conversation_preferences
from app.services.document_registry_service import activate_document
from app.services.security.auth_dependencies import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("")
def list_conversations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.updated_at.desc(), Conversation.created_at.desc()).all()
    return {"conversations": [serialize_conversation(row) for row in rows]}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = get_owned_conversation(db, current_user, conversation_id)
    document = {
        "active_doc_id": conversation.document_id,
        "active_filename": conversation.document_name,
        "source_path": conversation.document_path,
        "sha256": conversation.document_sha256,
        "input_language": conversation.document_input_language,
        "evidence_language": conversation.document_evidence_language,
        "total_pages": conversation.document_total_pages,
        "response_language": conversation.response_language,
    }
    available, reason = activate_document(document, current_user.id)
    result = serialize_conversation_detail(db, conversation)
    result["document_available"] = available
    result["document_unavailable_reason"] = reason if not available else None
    return result


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: str, request: ConversationUpdateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return serialize_conversation(update_conversation_preferences(db, current_user, conversation_id, title=request.title, response_language=request.response_language, response_length=request.response_length))
