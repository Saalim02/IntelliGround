
import json
import threading

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.evaluation_models import HumanReview
from app.models.query_models import (
    AnswerEvaluationRequest,
    AnswerFeedbackRequest,
    QueryRequest,
    HumanReviewRequest,
    QueryResponse,
)
from app.models.auth_models import User
from app.models.conversation_models import ConversationTurn
from app.core.database import get_db

from app.services.evaluation.evaluation_storage_service import (
    save_evaluation,
    load_evaluation,
    save_human_review,
    append_evaluation_dataset,
    save_user_feedback,
)
from app.services.answer_service import generate_grounded_answer
from app.services.document_registry_service import get_active_document
from app.services.evaluation.evaluation_service import evaluate_rag_response
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.security.guardrails import validate_question
from app.services.security.semantic_guardrail import classify_question_semantically
from app.services.security.audit_logger import log_security_event
from app.services.security.auth_dependencies import get_current_user
from app.services.security.auth_rate_limit import check_auth_rate_limit
from app.services.security.rate_limiter import RateLimiter
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.rag_orchestration_service import run_document_query
from app.services.conversation_service import (
    add_turn_if_new,
    create_conversation,
    get_owned_conversation,
    recent_context,
    validate_conversation_document,
)


router = APIRouter()
EVALUATION_RATE_LIMITER = RateLimiter(max_requests=5, window_seconds=60)
_evaluation_lock = threading.Lock()
_evaluations_in_progress: set[tuple[int, int]] = set()


@router.post(
    "/ask",
    response_model=QueryResponse,
)
def ask_question(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    active_document = get_active_document(current_user.id)
    conversation = None
    if request.conversation_id:
        conversation = get_owned_conversation(db, current_user, request.conversation_id)
        validate_conversation_document(conversation, active_document)
        # Client-supplied context is never trusted for persisted conversations.
        request.conversation_context = recent_context(db, conversation) or None
    elif active_document and active_document.get("active_doc_id") and active_document.get("active_filename"):
        conversation = create_conversation(
            db,
            current_user,
            document=active_document,
            mode="text",
            response_language=request.response_language or active_document.get("response_language") or "en",
            speech_input_language=None,
            speech_output_language=None,
        )

    result = run_document_query(
        request,
        current_user,
        endpoint="/api/ask",
        dependencies={
            # Preserve the route-level seams used by existing integrations and
            # tests while the implementation lives in the shared service.
            "validate_question": validate_question,
            "classify_question_semantically": classify_question_semantically,
            "resolve_authorized_api_key": resolve_authorized_api_key,
            "retrieve_relevant_chunks": retrieve_relevant_chunks,
            "get_active_document": get_active_document,
            "generate_grounded_answer": generate_grounded_answer,
            "evaluate_rag_response": evaluate_rag_response,
            "save_evaluation": save_evaluation,
        },
    )
    if conversation is not None:
        turn_id = request.turn_id or None
        add_turn_if_new(
            db,
            conversation,
            transcript=request.question,
            answer=None,
            evidence=[],
            provider_request_id=turn_id,
            role="user",
        )
        assistant_text = result.get("answer") or result.get("message")
        if assistant_text:
            assistant_turn = add_turn_if_new(
                db,
                conversation,
                transcript=None,
                answer=assistant_text,
                evidence=[
                    {
                        **item,
                        "_retrieved_chunks": result.get("retrieved_chunks", []),
                    }
                    for item in result.get("evidence", [])
                ],
                evaluation_id=(result.get("evaluation") or {}).get("evaluation_id"),
                provider_request_id=f"{turn_id}:assistant" if turn_id else None,
                role="assistant",
            )
            if assistant_turn is not None:
                result["answer_id"] = assistant_turn.id
        result["conversation_id"] = conversation.id
    else:
        result.setdefault("conversation_id", None)
    return result


def _owned_answer(db: Session, user: User, request) -> tuple[ConversationTurn, str, list, list]:
    try:
        answer = (
            db.query(ConversationTurn)
            .filter(
                ConversationTurn.id == request.answer_id,
                ConversationTurn.conversation_id == request.conversation_id,
                ConversationTurn.user_id == user.id,
                ConversationTurn.role == "assistant",
            )
            .first()
        )
    except Exception:
        answer = None
    if answer is None or not answer.answer:
        raise HTTPException(status_code=404, detail="Answer not found.")
    previous = (
        db.query(ConversationTurn)
        .filter(
            ConversationTurn.conversation_id == answer.conversation_id,
            ConversationTurn.user_id == user.id,
            ConversationTurn.role == "user",
            ConversationTurn.id < answer.id,
        )
        .order_by(ConversationTurn.id.desc())
        .first()
    )
    try:
        stored_evidence = json.loads(answer.evidence_json or "[]")
    except (TypeError, ValueError):
        stored_evidence = []
    retrieved = []
    for item in stored_evidence:
        if isinstance(item, dict) and item.get("_retrieved_chunks"):
            retrieved = item["_retrieved_chunks"]
            break
    return answer, previous.transcript if previous else "", stored_evidence, retrieved


@router.post("/evaluation/answer")
def evaluate_answer(
    request: AnswerEvaluationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    answer, question, evidence, retrieved_chunks = _owned_answer(db, current_user, request)
    if answer.evaluation_id:
        evaluation = load_evaluation(answer.evaluation_id, user_id=current_user.id)
        if evaluation:
            return {"status": "completed", "reused": True, "evaluation": evaluation.model_dump(mode="json")}
    if not question or not retrieved_chunks or not evidence:
        return {"status": "unavailable", "reason": "This answer does not contain eligible PDF evidence.", "evaluation": None}
    key = (current_user.id, answer.id)
    check_auth_rate_limit(EVALUATION_RATE_LIMITER, str(current_user.id))
    with _evaluation_lock:
        if key in _evaluations_in_progress:
            raise HTTPException(status_code=409, detail="Evaluation is already in progress.")
        _evaluations_in_progress.add(key)
    try:
        api_key = resolve_authorized_api_key(
            user=current_user,
            access_password=request.access_password,
            user_openai_api_key=request.user_openai_api_key,
        )
        evaluation = evaluate_rag_response(
            question=question,
            answer=answer.answer,
            retrieved_chunks=retrieved_chunks,
            evidence=evidence,
            api_key=api_key,
            use_llm_judges=True,
        )
        evaluation.user_id = current_user.id
        save_evaluation(evaluation, user_id=current_user.id)
        answer.evaluation_id = evaluation.evaluation_id
        db.add(answer)
        db.commit()
        return {"status": "completed", "reused": False, "evaluation": evaluation.model_dump(mode="json")}
    finally:
        with _evaluation_lock:
            _evaluations_in_progress.discard(key)


@router.post("/evaluation/answer/feedback")
def answer_feedback(
    request: AnswerFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _owned_answer(db, current_user, request)
    save_user_feedback(
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        answer_id=request.answer_id,
        helpful=request.helpful,
    )
    return {"status": "saved", "answer_id": request.answer_id, "helpful": request.helpful}


# -----------------------------------
# GET EVALUATION
# -----------------------------------
@router.get("/evaluation/{evaluation_id}")
def get_evaluation(
    evaluation_id: str,
    current_user: User = Depends(get_current_user),
):
    evaluation = load_evaluation(
        evaluation_id,
        user_id=current_user.id,
    )

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found.",
        )

    return evaluation.model_dump()


# -----------------------------------
# SUBMIT HUMAN REVIEW
# -----------------------------------
@router.post("/evaluation/{evaluation_id}/review")
def submit_human_review(
    evaluation_id: str,
    request: HumanReviewRequest,
    current_user: User = Depends(get_current_user),
):
    # -----------------------------------
    # VERIFY EVALUATION EXISTS
    # -----------------------------------
    evaluation = load_evaluation(
        evaluation_id,
        user_id=current_user.id,
    )

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found.",
        )

    # -----------------------------------
    # BUILD HUMAN REVIEW
    # -----------------------------------
    review = HumanReview(
        evaluation_id=evaluation_id,
        decision=request.decision,
        feedback_category=request.feedback_category,
        feedback_text=request.feedback_text,
        corrected_answer=request.corrected_answer,
    )

    # -----------------------------------
    # SAVE HUMAN REVIEW
    # -----------------------------------
    save_human_review(
        review,
        user_id=current_user.id,
    )

    # -----------------------------------
    # APPEND TO EVALUATION DATASET
    # -----------------------------------
    append_evaluation_dataset(
        evaluation=evaluation,
        review=review,
        user_id=current_user.id,
    )

    # -----------------------------------
    # SECURITY AUDIT
    # -----------------------------------
    log_security_event(
        "human_review_submitted",
        endpoint="/api/evaluation/{evaluation_id}/review",
        client_id=str(current_user.id),
        decision=review.decision.value,
        reasons=[
            (
                f"Human review submitted with category: "
                f"{review.feedback_category.value}"
                if review.feedback_category
                else "Human review submitted without a feedback category."
            )
        ],
    )

    # -----------------------------------
    # FINAL RESPONSE
    # -----------------------------------
    return {
        "status": "success",
        "evaluation_id": evaluation_id,
        "review": review.model_dump(),
        "message": "Human review saved successfully.",
    }
