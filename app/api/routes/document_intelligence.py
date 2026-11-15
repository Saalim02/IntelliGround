from fastapi import APIRouter, Depends, HTTPException
import logging

from app.models.auth_models import User
from app.models.document_intelligence_models import (
    DocumentLanguageRequest,
    DocumentLanguageResponse,
    DocumentQuestionsResponse,
    DocumentSummaryResponse,
    DocumentTopicsResponse,
)
from app.services.document_intelligence_service import (
    DocumentIntelligenceGenerationError,
    NoGroundedQuestionsError,
    generate_document_questions,
    generate_document_summary,
    generate_document_topics,
    save_response_language,
)
from app.services.document_registry_service import get_active_document
from app.services.security.auth_dependencies import get_current_user
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.language_registry import validate_language
from app.core.database import get_db
from app.services.conversation_service import create_conversation, get_owned_conversation, save_conversation_panel, validate_conversation_document
from sqlalchemy.orm import Session

router = APIRouter(prefix="/document", tags=["Document Intelligence"])
logger = logging.getLogger(__name__)


@router.post("/language", response_model=DocumentLanguageResponse)
def set_document_language(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user)):
    return save_response_language(current_user.id, request.response_language)


@router.get("/current")
def current_document(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's active document metadata for UI restore."""
    active = get_active_document(current_user.id)
    return {"status": "success", "document": active}


def _api_key(request: DocumentLanguageRequest, current_user: User) -> str:
    return resolve_authorized_api_key(user=current_user, access_password=request.access_password, user_openai_api_key=request.user_openai_api_key)


def _request_language(request: DocumentLanguageRequest, current_user: User, endpoint: str) -> str:
    try:
        return validate_language(request.response_language)
    except ValueError as exc:
        logger.warning(
            "document intelligence request validation: endpoint=%s document_id=%s response_language=%s validation_location=body.response_language validation_message=%s",
            endpoint, (get_active_document(current_user.id) or {}).get("active_doc_id"),
            request.response_language, str(exc),
        )
        raise HTTPException(status_code=422, detail={
            "loc": ["body", "response_language"],
            "msg": str(exc),
            "type": "value_error.language",
        }) from exc


def _generation_error(exc: ValueError, request: DocumentLanguageRequest, current_user: User, endpoint: str) -> HTTPException:
    active = get_active_document(current_user.id) or {}
    category = getattr(exc, "category", "document_generation_error")
    stage = getattr(exc, "stage", "validation")
    provider = getattr(exc, "provider", "application")
    model = getattr(exc, "model", "unknown")
    retry_attempted = bool(getattr(exc, "retry_attempted", False))
    original_exception_class = getattr(exc, "original_exception_class", type(exc).__name__)
    logger.warning(
        "document intelligence failure: endpoint=%s document_id=%s response_language=%s provider=%s model=%s failure_category=%s original_exception_class=%s stage=%s retry_attempted=%s safe_message=%s",
        endpoint, active.get("active_doc_id"), request.response_language, provider, model,
        category, original_exception_class, stage, retry_attempted, str(exc),
    )
    status_code = 503
    if category in {"provider_authentication_error", "provider_server_error", "connection_failure", "timeout", "invalid_provider_response", "provider_error"}:
        status_code = 502
    elif category == "rate_limited":
        status_code = 429
    elif category in {"empty_document", "document_not_found"}:
        status_code = 422
    return HTTPException(status_code=status_code, detail={
        "message": "Document intelligence generation failed.",
        "reason": str(exc),
        "endpoint": endpoint,
        "response_language": request.response_language,
        "provider": provider,
        "model": model,
        "failure_category": category,
        "original_exception_class": original_exception_class,
        "stage": stage,
        "retry_attempted": retry_attempted,
    })


def _call_generation_with_length(generator, *args, response_language: str, response_length: str):
    """Keep compatibility with older injected/test generators during rollout."""
    try:
        return generator(*args, response_language=response_language, response_length=response_length)
    except TypeError as exc:
        if "unexpected keyword argument 'response_length'" not in str(exc):
            raise
        return generator(*args, response_language=response_language)


def _persist_panel(request: DocumentLanguageRequest, current_user: User, db: Session, kind: str, result: dict, language: str) -> dict:
    persisted = dict(result)
    active = get_active_document(current_user.id)
    if request.document_id and active and request.document_id != active.get("active_doc_id"):
        raise HTTPException(status_code=409, detail="The requested document is not the active document. Reopen its conversation to reactivate it.")
    conversation_id = request.conversation_id
    if conversation_id:
        conversation = get_owned_conversation(db, current_user, conversation_id)
        validate_conversation_document(conversation, active)
    if not conversation_id:
        if active:
            conversation = create_conversation(
                db,
                current_user,
                document=active,
                mode="text",
                response_language=language,
                response_length=request.response_length,
                speech_input_language=None,
                speech_output_language=None,
                title=active.get("active_filename"),
            )
            conversation_id = conversation.id
    if conversation_id:
        save_conversation_panel(db, current_user, conversation_id, kind=kind, result=persisted, response_language=language, response_length=request.response_length, document_id=request.document_id)
        persisted["conversation_id"] = conversation_id
    return persisted


@router.post("/summary", response_model=DocumentSummaryResponse)
def document_summary(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        language = _request_language(request, current_user, "/api/document/summary")
        result = _call_generation_with_length(generate_document_summary, current_user.id, _api_key(request, current_user), response_language=language, response_length=request.response_length)
        return _persist_panel(request, current_user, db, "summary", result, language)
    except (DocumentIntelligenceGenerationError, ValueError) as exc:
        raise _generation_error(exc, request, current_user, "/api/document/summary") from exc


@router.post("/topics", response_model=DocumentTopicsResponse)
def document_topics(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        language = _request_language(request, current_user, "/api/document/topics")
        result = _call_generation_with_length(generate_document_topics, current_user.id, _api_key(request, current_user), response_language=language, response_length=request.response_length)
        return _persist_panel(request, current_user, db, "topics", result, language)
    except (DocumentIntelligenceGenerationError, ValueError) as exc:
        raise _generation_error(exc, request, current_user, "/api/document/topics") from exc


@router.post("/questions", response_model=DocumentQuestionsResponse)
def document_questions(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        language = _request_language(request, current_user, "/api/document/questions")
        active = get_active_document(current_user.id)
        logger.info(
            "document questions route diagnostics: user_id=%s document_id=%s received_response_language=%s document_language=%s",
            current_user.id, (active or {}).get("active_doc_id"), language,
            (active or {}).get("evidence_language") or (active or {}).get("input_language") or "unknown",
        )
        result = generate_document_questions(
            current_user.id,
            _api_key(request, current_user),
            response_language=language,
            response_length=request.response_length,
        )
        return _persist_panel(request, current_user, db, "questions", result, language)
    except NoGroundedQuestionsError as exc:
        # This is a valid application outcome, not malformed HTTP input.
        # Return a structured response so the UI can explain which evidence
        # and validation checks rejected the provider's candidates.
        return {
            "status": "error",
            "active_document": get_active_document(current_user.id).get("active_filename") if get_active_document(current_user.id) else None,
            "response_language": request.response_language,
            "response_length": request.response_length,
            "questions": [],
            "unsupported_items": exc.unsupported_items,
            "message": str(exc),
            "validation_diagnostics": exc.diagnostics,
        }
    except (DocumentIntelligenceGenerationError, ValueError) as exc:
        raise _generation_error(exc, request, current_user, "/api/document/questions") from exc
