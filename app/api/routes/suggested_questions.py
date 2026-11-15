from fastapi import APIRouter, Depends, HTTPException
import logging

from app.models.auth_models import User
from app.models.query_models import QueryRequest
from app.models.question_suggestion_models import SuggestedQuestionsResponse

from app.services.security.auth_dependencies import get_current_user
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.question_suggestion_service import generate_suggested_questions
from app.services.language_registry import validate_language

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/suggested-questions",
    response_model=SuggestedQuestionsResponse
)
def get_suggested_questions(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        # Preserve the service's persisted-language/English default when the
        # legacy client omits the selector field.
        response_language = (
            validate_language(request.response_language)
            if request.response_language
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "suggested questions request diagnostics: user_id=%s received_response_language=%s",
        current_user.id, response_language or "persisted_default",
    )
    api_key = resolve_authorized_api_key(
        user=current_user,
        access_password=request.access_password,
        user_openai_api_key=request.user_openai_api_key,
    )

    return generate_suggested_questions(
        api_key,
        user_id=current_user.id,
        response_language=response_language,
    )
