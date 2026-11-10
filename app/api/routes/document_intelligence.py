from fastapi import APIRouter, Depends, HTTPException

from app.models.auth_models import User
from app.models.document_intelligence_models import (
    DocumentLanguageRequest,
    DocumentLanguageResponse,
    DocumentQuestionsResponse,
    DocumentSummaryResponse,
    DocumentTopicsResponse,
)
from app.services.document_intelligence_service import (
    generate_document_questions,
    generate_document_summary,
    generate_document_topics,
    save_response_language,
)
from app.services.document_registry_service import get_active_document
from app.services.security.auth_dependencies import get_current_user
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.language_registry import validate_language

router = APIRouter(prefix="/document", tags=["Document Intelligence"])


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


@router.post("/summary", response_model=DocumentSummaryResponse)
def document_summary(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user)):
    try:
        language = validate_language(request.response_language)
        return generate_document_summary(current_user.id, _api_key(request, current_user), response_language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/topics", response_model=DocumentTopicsResponse)
def document_topics(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user)):
    try:
        language = validate_language(request.response_language)
        return generate_document_topics(current_user.id, _api_key(request, current_user), response_language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/questions", response_model=DocumentQuestionsResponse)
def document_questions(request: DocumentLanguageRequest, current_user: User = Depends(get_current_user)):
    try:
        language = validate_language(request.response_language)
        return generate_document_questions(current_user.id, _api_key(request, current_user), response_language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
